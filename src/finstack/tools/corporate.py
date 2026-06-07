"""
FinStack Corporate Intelligence Tool — one configurable tool for the corporate /
governance / regulatory intel that used to live across many narrow tools.

`corporate_intel(symbol, kinds)` runs one or more per-symbol "kinds" of corporate
intelligence in a single call and returns a dict keyed by kind. Each kind reuses
the SAME data-layer function the standalone tool already wraps — no new logic,
no invented function names.

kind -> data layer call (legacy tool it absorbs):
  insider        -> get_insider_trading(symbol, days=90)        [nse_insider_trading]
  promoter       -> get_promoter_shareholding(symbol)           [promoter_shareholding]
  pledge         -> get_promoter_pledge(symbol) + get_pledge_alert(symbol)
                                                                [promoter_pledge / get_pledge_alert]
  credit         -> get_credit_ratings(symbol)                  [credit_ratings]
  esg            -> get_brsr_esg(symbol)                        [brsr_esg]
  sebi           -> get_sebi_alerts(sector=symbol)              [get_sebi_alerts]
  agm            -> get_agm_brief(symbol)                       [get_agm_brief]
  insider_signal -> get_insider_signal(symbol)                  [get_insider_signal]

Failures are isolated per kind so one failing source never fails the whole call.
"""

import json
from finstack.utils.respond import dumps as _dumps

from finstack.data.market_intelligence import (
    get_insider_trading,
    get_promoter_shareholding,
    get_promoter_pledge,
)
from finstack.data.promoter_watch import get_pledge_alert
from finstack.data.insider_pattern import get_insider_signal
from finstack.data.credit_esg import get_credit_ratings, get_brsr_esg
from finstack.data.sebi_tracker import get_sebi_alerts
from finstack.data.agm import get_agm_brief


# kind -> callable(symbol) -> dict.  Each branch mirrors exactly how the original
# single-purpose tool wrapper built its result from the data layer.
def _kind_insider(symbol: str) -> dict:
    return get_insider_trading(symbol, 90)


def _kind_promoter(symbol: str) -> dict:
    return get_promoter_shareholding(symbol)


def _kind_pledge(symbol: str) -> dict:
    # promoter_pledge tool wrapped get_promoter_pledge; the pledge early-warning
    # (get_pledge_alert) is the richer QoQ/velocity view. Surface both.
    return {
        "pledge": get_promoter_pledge(symbol),
        "pledge_alert": get_pledge_alert(symbol),
    }


def _kind_credit(symbol: str) -> dict:
    return get_credit_ratings(symbol)


def _kind_esg(symbol: str) -> dict:
    return get_brsr_esg(symbol)


def _kind_sebi(symbol: str) -> dict:
    # get_sebi_alerts is sector-keyed, not symbol-keyed; pass the symbol through
    # as the sector filter (its native arg).
    return get_sebi_alerts(symbol)


def _kind_agm(symbol: str) -> dict:
    return get_agm_brief(symbol)


def _kind_insider_signal(symbol: str) -> dict:
    return get_insider_signal(symbol)


CORPORATE_KINDS = {
    "insider": _kind_insider,
    "promoter": _kind_promoter,
    "pledge": _kind_pledge,
    "credit": _kind_credit,
    "esg": _kind_esg,
    "sebi": _kind_sebi,
    "agm": _kind_agm,
    "insider_signal": _kind_insider_signal,
}


def register_corporate_tools(mcp):
    """Register the corporate intelligence tool with the MCP server."""

    @mcp.tool()
    def corporate_intel(symbol: str, kinds: str = "all") -> str:
        """Corporate / governance / regulatory intelligence for one stock — many lenses, one call.

        Replaces a fistful of narrow tools (insider trading, promoter shareholding,
        promoter pledge, credit ratings, BRSR/ESG, SEBI alerts, AGM brief, insider
        signal) with one configurable tool. Pick the lenses you want via `kinds`.

        Args:
            symbol: NSE stock symbol (e.g. RELIANCE, ADANIENT, ZEEL, TCS).
            kinds: comma-separated list, or "all". One or more of:
                - insider         NSE SAST insider trading disclosures (last 90 days)
                - promoter        shareholding pattern (promoter / FII / DII / public)
                - pledge          promoter pledge % + pledge early-warning (QoQ velocity)
                - credit          credit ratings from SEBI-mandated exchange filings
                - esg             BRSR sustainability / ESG disclosures
                - sebi            SEBI enforcement-order alerts (filtered by symbol/sector)
                - agm             AGM/EGM resolution briefing with red-flag detection
                - insider_signal  insider buy/sell BUY/SELL/NEUTRAL conviction signal

        Returns:
            JSON string: {symbol, kinds, results: {kind: <result>}}. A kind that
            errors gets {"error": "..."} under its key; the rest still return.

        Examples:
            corporate_intel("ADANIENT")                         # all lenses
            corporate_intel("RELIANCE", kinds="credit,esg")     # ratings + ESG only
            corporate_intel("ZEEL", kinds="pledge,insider_signal,agm")
        """
        if kinds.strip().lower() == "all":
            selected = list(CORPORATE_KINDS.keys())
        else:
            raw = [k.strip().lower() for k in kinds.split(",") if k.strip()]
            # de-duplicate, preserve order
            seen: set[str] = set()
            selected = [k for k in raw if not (k in seen or seen.add(k))]

        unknown = [k for k in selected if k not in CORPORATE_KINDS]
        if unknown:
            return _dumps({
                "error": f"Unknown kind(s): {', '.join(unknown)}.",
                "valid_kinds": sorted(CORPORATE_KINDS.keys()),
            }, indent=2)

        if not selected:
            return _dumps({
                "error": "No kinds provided.",
                "valid_kinds": sorted(CORPORATE_KINDS.keys()),
            }, indent=2)

        sym = symbol.strip()
        results: dict[str, object] = {}
        for kind in selected:
            try:
                results[kind] = CORPORATE_KINDS[kind](sym)
            except Exception as e:  # isolate per-kind failures
                results[kind] = {"error": f"{type(e).__name__}: {e}"}

        return _dumps({
            "symbol": sym.upper(),
            "kinds": selected,
            "results": results,
        }, indent=2, default=str)
