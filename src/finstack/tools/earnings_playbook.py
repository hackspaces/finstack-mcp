"""
FinStack Earnings Playbook — event-trading prep around results.

One decision-grade tool, ``earnings_playbook(symbol)``, that helps you position
*before* a company's quarterly results: when is the next print, how violently has
the stock historically reacted to results (the typical post-results swing), and a
cheap expected-move band derived from recent realized vol.

Honest scope:
  - Earnings dates come from yfinance (``Ticker.get_earnings_dates()`` / ``.calendar``).
    These are frequently sparse or stale for NSE names — when unavailable we say so
    rather than guessing.
  - We DO NOT have an options chain (NSE/BSE chains are IP-blocked from this host),
    so the "expected move" is a realized-vol proxy, NOT the true options-implied
    move. We label it as such everywhere.
  - Everything is computed from yfinance OHLCV via ``_hist`` / ``_daily_returns``.

Reused primitives (no reinvention):
  - data.chart_engine._hist(symbol, period, interval)  -> yfinance OHLCV DataFrame
  - data.quant_engine._fetch_close / _daily_returns    -> close series + returns
  - data.universe.UNIVERSE                              -> symbol -> company name
  - utils.respond.dumps                                 -> compact slimmed JSON
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from finstack.data.chart_engine import _hist
from finstack.data.quant_engine import _daily_returns, _fetch_close
from finstack.data.universe import UNIVERSE
from finstack.utils.respond import dumps

# Bounded for a 512MB host: never pull huge intraday frames, cap historical
# earnings reactions scanned, and run section fan-out on a small pool.
_MAX_PAST_EARNINGS = 12          # how many past earnings reactions to study
_PRICE_PERIOD = "2y"             # daily OHLCV lookback for reactions + vol
_VOL_WINDOW = 21                 # trading days of recent returns for the move proxy
_MAX_WORKERS = 3                 # concurrent sections


def _to_nse(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.startswith("^") or "." in s:
        return s
    return f"{s}.NS"


def _round(v, nd: int = 2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _as_date(ts) -> date | None:
    """Coerce a yfinance/pandas timestamp to a plain ``date``."""
    try:
        if ts is None:
            return None
        if isinstance(ts, (pd.Timestamp, datetime)):
            return ts.date()
        if isinstance(ts, date):
            return ts
        return pd.Timestamp(ts).date()
    except Exception:
        return None


# ───────────────────────── section: earnings calendar ─────────────────────────

def _earnings_dates(symbol: str) -> dict:
    """Pull past + future earnings dates from yfinance (guarded, fail-loud).

    Returns: {"all": [date,...], "future": [date,...], "past": [date,...],
              "next_date": date|None, "source": str}
    Raises on hard failure so the caller can record the section error.
    """
    import yfinance as yf  # guarded local import; optional at module level

    tk = yf.Ticker(_to_nse(symbol))
    today = datetime.now(timezone.utc).date()
    found: list[date] = []
    source = None

    # Primary: get_earnings_dates() — gives a multi-quarter history + a few ahead.
    try:
        ed = tk.get_earnings_dates(limit=24)
        if ed is not None and len(ed) > 0:
            for idx in ed.index:
                d = _as_date(idx)
                if d is not None:
                    found.append(d)
            source = "get_earnings_dates"
    except Exception:
        pass

    # Fallback: .calendar usually carries the single next "Earnings Date".
    if not found:
        try:
            cal = tk.calendar
            cand = None
            if isinstance(cal, dict):
                cand = cal.get("Earnings Date")
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in getattr(cal, "index", []):
                cand = cal.loc["Earnings Date"].tolist()
            if cand is not None:
                items = cand if isinstance(cand, (list, tuple)) else [cand]
                for it in items:
                    d = _as_date(it)
                    if d is not None:
                        found.append(d)
                if found:
                    source = "calendar"
        except Exception:
            pass

    found = sorted(set(found))
    if not found:
        return {
            "available": False,
            "note": ("yfinance returned no earnings dates for this symbol "
                     "(common for NSE names) — cross-check the NSE corporate "
                     "calendar manually."),
            "all": [], "future": [], "past": [], "next_date": None, "source": None,
        }

    future = [d for d in found if d >= today]
    past = [d for d in found if d < today]
    next_date = future[0] if future else None
    return {
        "available": True,
        "source": source,
        "next_date": next_date.isoformat() if next_date else None,
        "days_to_next": (next_date - today).days if next_date else None,
        "future": [d.isoformat() for d in future],
        "past": [d.isoformat() for d in past[-_MAX_PAST_EARNINGS:]],
        "_past_dates": past[-_MAX_PAST_EARNINGS:],  # internal, stripped before output
    }


# ─────────────────────── section: historical reaction ─────────────────────────

def _historical_reaction(symbol: str, past_dates: list[date]) -> dict:
    """For each past earnings date, the NEXT-session % move (the post-results gap+drift).

    Reports the typical post-results swing: count, avg ABS move, median, max
    up/down moves, and up/down skew. Uses daily OHLCV; the "reaction" is the
    close-to-close % change on the first trading session on/after each date.
    """
    if not past_dates:
        return {"available": False,
                "note": "no past earnings dates available to measure reactions."}

    df = _hist(symbol, _PRICE_PERIOD, "1d")  # may raise -> section error
    close = df["Close"].dropna()
    if close.empty:
        return {"available": False, "note": "no close prices to measure reactions."}

    idx_dates = [d.date() if hasattr(d, "date") else d for d in close.index]
    closes = close.tolist()

    reactions = []
    for ed in past_dates:
        # First trading bar strictly AFTER the earnings date is the reaction bar;
        # compare it to the prior bar's close (captures the gap + session drift).
        post_i = None
        for i, dd in enumerate(idx_dates):
            if dd > ed:
                post_i = i
                break
        if post_i is None or post_i == 0:
            continue
        prev_c = closes[post_i - 1]
        post_c = closes[post_i]
        if not prev_c or prev_c <= 0:
            continue
        move = (post_c / prev_c - 1.0) * 100.0
        if math.isnan(move) or math.isinf(move):
            continue
        reactions.append({
            "earnings_date": ed.isoformat(),
            "reaction_date": idx_dates[post_i].isoformat(),
            "move_pct": _round(move),
        })

    if not reactions:
        return {"available": False,
                "note": ("past earnings dates known but none fell inside the "
                         f"{_PRICE_PERIOD} price window — cannot measure reactions.")}

    moves = np.array([r["move_pct"] for r in reactions], dtype=float)
    abs_moves = np.abs(moves)
    ups = moves[moves > 0]
    downs = moves[moves < 0]
    max_up = float(moves.max())
    max_down = float(moves.min())

    return {
        "available": True,
        "samples": len(reactions),
        "avg_abs_move_pct": _round(abs_moves.mean()),
        "median_abs_move_pct": _round(np.median(abs_moves)),
        "max_up_move_pct": _round(max_up),
        "max_down_move_pct": _round(max_down),
        "up_count": int(ups.size),
        "down_count": int(downs.size),
        "up_down_skew": ("upward" if ups.size > downs.size else
                         "downward" if downs.size > ups.size else "balanced"),
        "pct_positive": _round(ups.size / len(reactions) * 100.0, 1),
        "history": reactions[-_MAX_PAST_EARNINGS:],
        "note": ("next-session close-to-close move vs the prior close — the "
                 "typical post-results swing. Captures gap + first-session drift, "
                 "not the full multi-day move."),
    }


# ───────────────────── section: expected (proxy) move band ─────────────────────

def _expected_move(symbol: str, days_to_event: int | None) -> dict:
    """Realized-vol proxy for the options-implied move (chains are IP-blocked).

    Recent daily vol (std of last ``_VOL_WINDOW`` daily returns) scaled by
    sqrt(days_to_event or 1) gives a +/- % and a price band around spot.
    """
    close = _fetch_close(symbol, "6mo")  # may raise -> section error
    rets = _daily_returns(close)
    if len(rets) < 10:
        return {"available": False,
                "note": "insufficient return history for a vol estimate."}

    recent = rets.tail(_VOL_WINDOW)
    daily_vol = float(recent.std(ddof=1))
    if math.isnan(daily_vol) or daily_vol <= 0:
        return {"available": False, "note": "degenerate (zero) recent volatility."}

    spot = float(close.iloc[-1])
    # Horizon: time to the event if known, else a single-session move.
    horizon = max(int(days_to_event), 1) if days_to_event else 1
    move_frac = daily_vol * math.sqrt(horizon)
    move_pct = move_frac * 100.0

    return {
        "available": True,
        "method": "realized_vol_proxy",
        "spot": _round(spot),
        "recent_daily_vol_pct": _round(daily_vol * 100.0),
        "vol_window_days": int(min(_VOL_WINDOW, len(rets))),
        "horizon_days": horizon,
        "expected_move_pct": _round(move_pct),
        "price_band": {
            "low": _round(spot * (1 - move_frac)),
            "high": _round(spot * (1 + move_frac)),
        },
        "annualized_vol_pct": _round(daily_vol * math.sqrt(252) * 100.0),
        "caveat": ("PROXY ONLY — derived from recent realized daily vol, not an "
                   "options chain (NSE/BSE chains are IP-blocked from this host). "
                   "True implied move is usually richer than realized into a print."),
    }


# ───────────────────────────── positioning note ───────────────────────────────

def _positioning(react: dict, em: dict, days_to_next) -> dict:
    """Synthesize a directional-vs-volatility positioning suggestion."""
    bits = []
    swing = react.get("avg_abs_move_pct") if react.get("available") else None
    em_pct = em.get("expected_move_pct") if em.get("available") else None
    skew = react.get("up_down_skew") if react.get("available") else None

    high_swing = swing is not None and swing >= 5.0
    elevated_vol = em_pct is not None and em_pct >= 5.0

    if high_swing and elevated_vol:
        stance = "volatility / long-gamma"
        bits.append(f"high historical post-results swing (~{swing}% avg abs) and "
                    f"elevated expected move (~{em_pct}%) favour a long-vol "
                    "structure (e.g. straddle/strangle) over a directional bet.")
    elif high_swing:
        stance = "volatility-lean"
        bits.append(f"history shows large post-results swings (~{swing}% avg abs) — "
                    "size for a gap; a long-vol structure can beat picking direction.")
    elif swing is not None:
        stance = "directional-ok"
        bits.append(f"historically muted reaction (~{swing}% avg abs) — a defined-risk "
                    "directional view is reasonable if you have an edge on the print.")
    else:
        stance = "insufficient-history"
        bits.append("no measurable reaction history — treat positioning as speculative.")

    if skew and skew != "balanced" and react.get("available"):
        bits.append(f"reactions skew {skew} "
                    f"({react.get('up_count')} up / {react.get('down_count')} down).")

    if isinstance(days_to_next, int):
        if days_to_next <= 3:
            bits.append(f"event is imminent ({days_to_next}d) — vol/theta decay are live; "
                        "avoid getting run over by an IV crush after the print.")
        else:
            bits.append(f"{days_to_next}d to the event — room to scale in.")
    else:
        bits.append("next earnings date unknown — cannot time entry precisely.")

    return {"stance": stance, "rationale": " ".join(bits),
            "reminder": "Realized-vol proxy understates true pre-event IV; size accordingly."}


# ───────────────────────────────── tool ───────────────────────────────────────

def register_earnings_playbook_tools(mcp):
    """Register the earnings-playbook tool with the MCP server."""

    @mcp.tool()
    def earnings_playbook(symbol: str) -> str:
        """Event-trading prep around a stock's quarterly results.

        Builds a decision-grade playbook for positioning into an earnings print:
        when results are due, how the stock has *historically* reacted (the typical
        post-results swing), a cheap expected-move band, and a positioning stance
        (long-vol/straddle vs directional).

        Sections (each fails loud and independently — a broken section is reported,
        never fabricated):
          - earnings_calendar:   next earnings date + days away (yfinance; often
                                 sparse/stale for NSE — said honestly when missing).
          - historical_reaction: for each past earnings date, the NEXT-session
                                 close-to-close % move; avg ABS move, median, max
                                 up/down, and up/down skew.
          - expected_move:       recent realized daily vol * sqrt(days_to_event or 1)
                                 -> +/- % and a price band. PROXY for the options-
                                 implied move (chains are IP-blocked here).
          - positioning:         straddle-vs-directional suggestion from swing + vol.

        Bounded for a 512MB host: <=2y daily OHLCV, last {n} earnings reactions,
        and a small thread pool for the independent sections.

        Args:
            symbol: NSE symbol (".NS" is appended automatically). Indexes ("^NSEI")
                    and explicit suffixes (".BO") are passed through unchanged.

        Returns:
            JSON string: {symbol, company, verdict, summary, earnings_calendar,
            historical_reaction, expected_move, positioning, caveats}.

        Indian example:
            earnings_playbook("INFY")   # Infosys results playbook
            earnings_playbook("RELIANCE")
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return dumps({"error": "no symbol provided"})

        company = UNIVERSE.get(sym)

        result: dict = {
            "symbol": sym,
            "company": company or sym,
            "data_caps": {
                "price_period": _PRICE_PERIOD,
                "max_past_earnings": _MAX_PAST_EARNINGS,
                "vol_window_days": _VOL_WINDOW,
                "note": "bounded for a 512MB host.",
            },
        }

        # 1) Earnings calendar first (synchronous): the rest depend on its output.
        try:
            cal = _earnings_dates(sym)
        except Exception as e:
            cal = {"available": False,
                   "error": f"{type(e).__name__}: {e}",
                   "_past_dates": [], "days_to_next": None}

        past_dates = cal.pop("_past_dates", []) if isinstance(cal, dict) else []
        days_to_next = cal.get("days_to_next") if isinstance(cal, dict) else None
        result["earnings_calendar"] = cal

        # 2) Reaction + expected-move are independent -> run concurrently.
        sections = {}
        jobs = {
            "historical_reaction": lambda: _historical_reaction(sym, past_dates),
            "expected_move": lambda: _expected_move(sym, days_to_next),
        }
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futs = {ex.submit(fn): name for name, fn in jobs.items()}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    sections[name] = fut.result()
                except Exception as e:
                    sections[name] = {"available": False,
                                      "error": f"{type(e).__name__}: {e}"}

        react = sections.get("historical_reaction", {"available": False})
        em = sections.get("expected_move", {"available": False})
        result["historical_reaction"] = react
        result["expected_move"] = em

        # 3) Positioning synthesis (pure; depends on the two above).
        try:
            result["positioning"] = _positioning(react, em, days_to_next)
        except Exception as e:
            result["positioning"] = {"error": f"{type(e).__name__}: {e}"}

        # 4) Verdict + summary.
        next_d = cal.get("next_date") if isinstance(cal, dict) else None
        swing = react.get("avg_abs_move_pct") if react.get("available") else None
        em_pct = em.get("expected_move_pct") if em.get("available") else None
        stance = result["positioning"].get("stance") if isinstance(result["positioning"], dict) else None

        if next_d:
            verdict = f"Next results ~{next_d} ({days_to_next}d) — stance: {stance}"
        elif swing is not None:
            verdict = f"Date unknown; historical swing ~{swing}% avg abs — stance: {stance}"
        else:
            verdict = "Insufficient earnings data for a confident playbook"

        summary_parts = []
        if next_d:
            summary_parts.append(f"results due ~{next_d} ({days_to_next}d away)")
        else:
            summary_parts.append("no earnings date from yfinance")
        if swing is not None:
            summary_parts.append(f"typical post-results swing ~{swing}% "
                                 f"(over {react.get('samples')} prints, skew {react.get('up_down_skew')})")
        if em_pct is not None and em.get("available"):
            band = em.get("price_band", {})
            summary_parts.append(f"proxy expected move ~+/-{em_pct}% "
                                 f"(band {band.get('low')}-{band.get('high')})")
        result["verdict"] = verdict
        result["summary"] = "; ".join(summary_parts) + "."

        result["caveats"] = [
            "Earnings dates from yfinance are often sparse/stale for NSE — verify on NSE.",
            "expected_move is a REALIZED-vol proxy, NOT the options-implied move "
            "(chains are IP-blocked from this host); true IV is usually richer pre-print.",
            "historical_reaction is next-session close-to-close, not the full multi-day move.",
            "Not investment advice; an IV crush can lose money even on a correct direction.",
        ]
        return dumps(result)

    return earnings_playbook
