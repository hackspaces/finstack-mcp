"""MCP tool: news — datacenter-safe RSS catalyst feed.

Why RSS and not NSE/BSE APIs:
    NSE/BSE programmatic endpoints reject datacenter / cloud IPs (Akamai bot wall),
    so on a hosted MCP they simply hang or 403. Public RSS feeds (Google News,
    Yahoo Finance, SEBI) are *not* IP-blocked and are the reliable way to surface
    fresh catalysts/headlines from a server. This tool fans out across those
    sources concurrently, dedupes by title, and returns the freshest items.

Sources (each wrapped in its own try/except — one dead feed never sinks the call):
    - Google News RSS  : company-name + " stock" (if symbol) else the topic
    - Yahoo Finance RSS: per-symbol headline feed (only when a symbol is given)
    - SEBI RSS         : regulatory press releases (general / no-symbol context)

512MB host note: feeds are small (KB of XML), fetched with a short timeout and a
hard per-source byte cap; the only heavy import (feedparser) is lazy-loaded inside
the tool so importing this module stays cheap.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from mcp.server.fastmcp import FastMCP

from finstack.data.universe import UNIVERSE
from finstack.utils.respond import dumps

# --- bounds for a small host -------------------------------------------------
_HTTP_TIMEOUT = 6.0            # seconds per source
_MAX_BYTES = 1_500_000         # cap each feed body (~1.5MB) so a runaway feed can't OOM
_HARD_ITEM_CAP = 60            # never return more than this regardless of `limit`
_UA = "Mozilla/5.0 (compatible; finstack-mcp/1.0; +https://github.com/finstacklabs/finstack-mcp)"


def _fetch_xml(url: str) -> bytes:
    """Fetch raw RSS bytes with a short timeout + byte cap. Raises on failure."""
    import httpx

    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": _UA}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        body = resp.content
        if len(body) > _MAX_BYTES:
            body = body[:_MAX_BYTES]
        return body


def _published_epoch(entry) -> float:
    """Best-effort epoch seconds for sorting; 0.0 if unknown."""
    st = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if st:
        try:
            return time.mktime(st)
        except Exception:
            pass
    raw = entry.get("published") or entry.get("updated") or ""
    if raw:
        try:
            return parsedate_to_datetime(raw).timestamp()
        except Exception:
            return 0.0
    return 0.0


def _parse_feed(url: str, source: str, default_source: str) -> list:
    """Fetch + parse one RSS feed into normalized item dicts. Raises on hard failure."""
    import feedparser

    raw = _fetch_xml(url)
    parsed = feedparser.parse(raw)
    out = []
    for e in parsed.entries:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        # Google News prefixes the real publisher inside source; prefer it.
        src = default_source
        try:
            if e.get("source") and e.source.get("title"):
                src = e.source.title
        except Exception:
            pass
        out.append({
            "title": title,
            "source": src,
            "published": (e.get("published") or e.get("updated") or "").strip(),
            "link": (e.get("link") or "").strip(),
            "_epoch": _published_epoch(e),
        })
    return out


def register_news_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def news(symbol: str = "", topic: str = "", limit: int = 15) -> str:
        """
        Fresh market-moving headlines / catalysts for an NSE stock or a topic,
        aggregated from datacenter-safe RSS feeds (Google News, Yahoo Finance, SEBI).

        Use this on a hosted server where NSE/BSE APIs are IP-blocked: RSS still
        works. Fans out across sources concurrently, dedupes by title, and returns
        the freshest items first with a one-line verdict.

        Args:
            symbol: NSE symbol (e.g. "RELIANCE", "TCS"). If set, the Google News
                    query uses the company name from the universe + " stock", and
                    the Yahoo per-symbol feed (<SYMBOL>.NS) is added.
            topic:  Free-text query used when no symbol is given
                    (e.g. "RBI policy", "Nifty IT", "crude oil India").
            limit:  Max items to return (default 15, hard cap 60).

        Indian example:
            news(symbol="RELIANCE")
              -> Reliance Industries headlines from Google News + Yahoo (RELIANCE.NS),
                 newest first.
            news(topic="RBI repo rate")
              -> general catalyst feed incl. SEBI regulatory releases.

        Returns JSON:
            {
              "verdict": one-line summary (e.g. "12 fresh headlines for RELIANCE"),
              "query": the search string actually used,
              "symbol", "company", "topic",
              "count": number of items returned,
              "sources_ok":  feeds that returned items,
              "sources_failed": {feed: error} for feeds that errored,
              "items": [ {title, source, published, link}, ... ]  (deduped, newest first)
            }
        Fail-loud: each source is isolated; a dead feed appears in sources_failed
        but never blocks the others. Nothing is fabricated.
        """
        try:
            limit = max(1, min(int(limit), _HARD_ITEM_CAP))
        except Exception:
            limit = 15

        symbol = (symbol or "").strip().upper()
        topic = (topic or "").strip()

        company = UNIVERSE.get(symbol, "") if symbol else ""

        # Build the Google News query: company name + " stock" if symbol, else topic.
        if symbol:
            base = company or symbol
            gquery = f"{base} stock"
        else:
            gquery = topic

        result = {
            "symbol": symbol or None,
            "company": company or None,
            "topic": topic or None,
            "query": gquery or None,
        }

        if not gquery and not symbol:
            result["verdict"] = "No symbol or topic provided — nothing to search."
            result["count"] = 0
            result["items"] = []
            return dumps(result)

        # --- assemble the source plan -------------------------------------
        # Each: (key, url, default_source_label)
        jobs = []
        if gquery:
            g_url = ("https://news.google.com/rss/search?q="
                     + quote_plus(gquery)
                     + "&hl=en-IN&gl=IN&ceid=IN:en")
            jobs.append(("google_news", g_url, "Google News"))
        if symbol:
            y_url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(symbol)}.NS"
                     "&region=IN&lang=en-IN")
            jobs.append(("yahoo_finance", y_url, "Yahoo Finance"))
        # SEBI: regulatory context. Include when no symbol (general feed) or always
        # as a general supplement when querying a topic.
        if not symbol:
            jobs.append(("sebi", "https://www.sebi.gov.in/sebirss.xml", "SEBI"))

        items = []
        sources_ok = []
        sources_failed = {}

        # Fan out concurrently — feeds are I/O bound; small worker pool for 512MB host.
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            futs = {
                pool.submit(_parse_feed, url, key, label): key
                for (key, url, label) in jobs
            }
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    feed_items = fut.result()
                    if feed_items:
                        items.extend(feed_items)
                        sources_ok.append(key)
                    else:
                        sources_failed[key] = "no items returned"
                except Exception as exc:  # fail-loud per section, never fabricate
                    sources_failed[key] = f"{type(exc).__name__}: {str(exc)[:160]}"

        # --- dedupe by normalized title, keep newest-published copy --------
        by_title = {}
        for it in items:
            tkey = it["title"].strip().lower()
            prev = by_title.get(tkey)
            if prev is None or it["_epoch"] > prev["_epoch"]:
                by_title[tkey] = it

        merged = sorted(by_title.values(), key=lambda x: x["_epoch"], reverse=True)
        clean = [{
            "title": it["title"],
            "source": it["source"],
            "published": it["published"] or None,
            "link": it["link"] or None,
        } for it in merged[:limit]]

        # --- verdict ------------------------------------------------------
        label = symbol or (topic[:40] if topic else "query")
        if clean:
            verdict = f"{len(clean)} fresh headline(s) for {label}"
            if sources_failed:
                verdict += f" ({len(sources_failed)} source(s) unavailable)"
        elif sources_failed and not sources_ok:
            verdict = (f"All {len(sources_failed)} feed(s) failed for {label} — "
                       "no headlines available (sources may be down or blocked).")
        else:
            verdict = f"No recent headlines found for {label}."

        result["verdict"] = verdict
        result["count"] = len(clean)
        result["sources_ok"] = sources_ok
        result["sources_failed"] = sources_failed or None
        result["items"] = clean
        return dumps(result)
