"""
FinStack `alerts` — watchlist condition scanner.

Runs a full technical read (tech_engine.technicals view="all") CONCURRENTLY across
a list of NSE symbols (or a named basket from ALL_BASKETS) and flags actionable
conditions per name: breakouts/breakdowns, RSI extremes, 50/200 cross state,
volume spikes, MACD cross, and proximity to 20d support/resistance.

Decision-grade: returns only symbols with >=1 trigger, each with its alert list and
human-readable detail, plus a summary count by condition and a market-wide verdict.
Fail-loud per symbol — a symbol that errors is reported under `errors`, never silently
dropped or fabricated.

NOTE ON "JUST CROSSED": tech_engine.technicals() exposes only CURRENT indicator values
(not the prior bar), so golden_cross / death_cross / macd_*_cross are detected as STATE
(the faster line is above/below the slower) gated to FRESH crosses by requiring the two
lines to be within a tight band (proximity) of each other. This catches recent/incipient
crosses but cannot prove the exact bar of crossover. Flagged honestly in each alert's
`detail` and in the output `methodology`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from finstack.utils.respond import dumps
from finstack.data import tech_engine as te
from finstack.data.sector_engine import ALL_BASKETS
from finstack.data.universe import UNIVERSE

# Bound work for a 512MB / shared host: cap symbols and worker fan-out.
_MAX_SYMBOLS = 25
_MAX_WORKERS = 8

# All conditions this scanner understands (order = report/summary order).
_ALL_CONDITIONS = [
    "breakout_up", "breakdown", "oversold", "overbought",
    "golden_cross", "death_cross", "volume_spike",
    "macd_bull_cross", "macd_bear_cross", "near_support", "near_resistance",
]

# Tunables (kept local + transparent in output).
_PROX = 0.02          # 2% proximity band for breakouts / support / resistance
_RVOL_BREAKOUT = 1.3  # rvol needed to confirm a breakout
_RVOL_SPIKE = 2.0     # rvol that constitutes a volume spike
_RSI_OS = 30.0
_RSI_OB = 70.0
_MA_CROSS_BAND = 0.02   # 50d/200d within 2% => treat as a FRESH (recent) cross
_MACD_CROSS_BAND = 0.15  # |macd-signal| / |macd| < 15% => FRESH macd cross


def _pct(a: float, b: float) -> float | None:
    """(a-b)/b as a fraction; None if b is falsy/None."""
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b


def _resolve_symbols(symbols: str) -> tuple[list[str], str, list[str]]:
    """Return (symbol_list, source_label, notes). Accepts a basket name OR a comma list."""
    notes: list[str] = []
    raw = symbols.strip()
    # exact basket-name match (case-sensitive registry keys)
    if raw in ALL_BASKETS:
        syms = list(ALL_BASKETS[raw]["symbols"])
        label = f"basket:{raw}"
    else:
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
        label = "custom_list"
    # de-dup, preserve order
    seen: set[str] = set()
    uniq = [s for s in syms if not (s in seen or seen.add(s))]
    if len(uniq) > _MAX_SYMBOLS:
        notes.append(f"capped to first {_MAX_SYMBOLS} of {len(uniq)} symbols (512MB host)")
        uniq = uniq[:_MAX_SYMBOLS]
    return uniq, label, notes


def _scan_one(symbol: str, wanted: set[str]) -> dict:
    """Compute technicals for one symbol and return its triggered alerts. Fail-loud."""
    res = te.technicals(symbol, views="all", period="1y", interval="1d")
    if not isinstance(res, dict) or "error" in res:
        raise RuntimeError(res.get("error", "no technicals") if isinstance(res, dict) else "no technicals")

    ind = res.get("indicators") or {}
    lvl = res.get("levels") or {}
    vol = res.get("volume") or {}

    price = ind.get("price")
    if price is None:
        raise RuntimeError("no price in technicals output")

    rsi = ind.get("rsi14")
    sma50 = ind.get("sma50")
    sma200 = ind.get("sma200")
    macd = ind.get("macd")
    macd_sig = ind.get("macd_signal")
    rvol = vol.get("rvol")
    res_20 = lvl.get("recent_resistance_20d")
    sup_20 = lvl.get("recent_support_20d")

    alerts: list[dict] = []

    def fire(cond: str, detail: str, **extra):
        if cond in wanted:
            alerts.append({"condition": cond, "detail": detail, **extra})

    # --- breakout_up / breakdown (20d high/low + rvol) ---
    if res_20 is not None:
        d = _pct(price, res_20)  # >=0 means at/above 20d high
        if d is not None and d >= -_PROX:
            if rvol is not None and rvol > _RVOL_BREAKOUT:
                fire("breakout_up",
                     f"price {price} is {round(d*100,2)}% vs 20d high {res_20} with rvol {rvol}",
                     gap_pct=round(d * 100, 2), rvol=rvol)
            else:
                # near the high but volume not confirming -> resistance test, handled below
                pass
    if sup_20 is not None:
        d = _pct(price, sup_20)  # <=0 means at/below 20d low
        if d is not None and d <= _PROX:
            fire("breakdown",
                 f"price {price} is {round(d*100,2)}% vs 20d low {sup_20}"
                 + (f" with rvol {rvol}" if rvol is not None else ""),
                 gap_pct=round(d * 100, 2), rvol=rvol)

    # --- RSI extremes ---
    if rsi is not None and rsi < _RSI_OS:
        fire("oversold", f"RSI14 {rsi} < {int(_RSI_OS)}", rsi=rsi)
    if rsi is not None and rsi > _RSI_OB:
        fire("overbought", f"RSI14 {rsi} > {int(_RSI_OB)}", rsi=rsi)

    # --- golden / death cross (STATE + freshness band; see module note) ---
    if sma50 is not None and sma200 is not None:
        spread = _pct(sma50, sma200)
        if spread is not None:
            fresh = abs(spread) < _MA_CROSS_BAND
            if sma50 > sma200:
                fire("golden_cross",
                     f"50d {sma50} {'just crossed' if fresh else 'stacked'} above 200d {sma200} "
                     f"(spread {round(spread*100,2)}%; STATE-based, exact cross bar unknown)",
                     spread_pct=round(spread * 100, 2), fresh=fresh)
            elif sma50 < sma200:
                fire("death_cross",
                     f"50d {sma50} {'just crossed' if fresh else 'stacked'} below 200d {sma200} "
                     f"(spread {round(spread*100,2)}%; STATE-based, exact cross bar unknown)",
                     spread_pct=round(spread * 100, 2), fresh=fresh)

    # --- volume spike ---
    if rvol is not None and rvol > _RVOL_SPIKE:
        fire("volume_spike", f"rvol {rvol} > {_RVOL_SPIKE}x avg20d volume", rvol=rvol)

    # --- MACD cross (STATE + freshness band; see module note) ---
    if macd is not None and macd_sig is not None:
        denom = abs(macd) if macd else None
        gap = abs(macd - macd_sig)
        fresh = (denom is not None and denom > 0 and gap / denom < _MACD_CROSS_BAND)
        if macd > macd_sig:
            fire("macd_bull_cross",
                 f"MACD {macd} above signal {macd_sig} "
                 f"({'fresh' if fresh else 'established'}; STATE-based, exact cross bar unknown)",
                 macd=macd, signal=macd_sig, fresh=fresh)
        elif macd < macd_sig:
            fire("macd_bear_cross",
                 f"MACD {macd} below signal {macd_sig} "
                 f"({'fresh' if fresh else 'established'}; STATE-based, exact cross bar unknown)",
                 macd=macd, signal=macd_sig, fresh=fresh)

    # --- near support / resistance (within 2% of 20d level) ---
    if res_20 is not None:
        d = _pct(price, res_20)
        if d is not None and -_PROX <= d <= 0:  # just below the 20d high
            fire("near_resistance",
                 f"price {price} within {round(abs(d)*100,2)}% below 20d resistance {res_20}",
                 distance_pct=round(d * 100, 2))
    if sup_20 is not None:
        d = _pct(price, sup_20)
        if d is not None and 0 <= d <= _PROX:  # just above the 20d low
            fire("near_support",
                 f"price {price} within {round(d*100,2)}% above 20d support {sup_20}",
                 distance_pct=round(d * 100, 2))

    return {
        "symbol": symbol.upper(),
        "name": UNIVERSE.get(symbol.upper(), ""),
        "price": price,
        "bias": (res.get("summary") or {}).get("bias"),
        "rsi14": rsi,
        "rvol": rvol,
        "triggered": alerts,
    }


def register_alerts_tools(mcp):
    """Register the `alerts` watchlist-scanner tool."""

    @mcp.tool()
    def alerts(symbols: str, conditions: str = "all") -> str:
        """Scan a watchlist for triggered technical alerts (breakouts, RSI extremes, crosses, volume).

        Runs a full technical read on every symbol CONCURRENTLY and flags actionable
        conditions. Returns only symbols with >=1 trigger, plus a summary count by
        condition and a market-wide verdict. Fail-loud: a symbol that errors is listed
        under `errors`, never silently dropped.

        Conditions detected:
          - breakout_up    : price at/above (or within 2% of) the 20d high AND rvol > 1.3
          - breakdown      : price at/below (or within 2% of) the 20d low
          - oversold       : RSI14 < 30
          - overbought     : RSI14 > 70
          - golden_cross   : 50d SMA above 200d SMA (fresh if within 2%)
          - death_cross    : 50d SMA below 200d SMA (fresh if within 2%)
          - volume_spike   : rvol > 2x the 20d average volume
          - macd_bull_cross: MACD above its signal line (fresh if lines are tight)
          - macd_bear_cross: MACD below its signal line (fresh if lines are tight)
          - near_support   : price within 2% above the 20d support
          - near_resistance: price within 2% below the 20d resistance

        Args:
            symbols: comma-separated NSE symbols (".NS" auto-appended; capped at 25) OR a
                single basket name from ALL_BASKETS (e.g. "nifty_it", "psu_banks").
            conditions: "all" (default) checks every condition above; otherwise a comma
                list to filter which conditions to report (e.g. "breakout_up,volume_spike").

        Indian example:
            alerts(symbols="RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK")
            alerts(symbols="nifty_it", conditions="breakout_up,overbought,volume_spike")
        """
        try:
            syms, source, notes = _resolve_symbols(symbols)
            if not syms:
                return dumps({"error": "no symbols resolved",
                              "hint": "pass a comma list or a basket name from ALL_BASKETS"})

            # which conditions to report
            if conditions.strip().lower() == "all":
                wanted = set(_ALL_CONDITIONS)
                unknown: list[str] = []
            else:
                req = [c.strip().lower() for c in conditions.split(",") if c.strip()]
                wanted = {c for c in req if c in _ALL_CONDITIONS}
                unknown = [c for c in req if c not in _ALL_CONDITIONS]
                if not wanted:
                    return dumps({"error": "no valid conditions requested",
                                  "unknown_conditions": unknown,
                                  "valid_conditions": _ALL_CONDITIONS})

            # concurrent fan-out (fail-loud per symbol)
            hits: list[dict] = []
            errors: list[dict] = []
            workers = max(1, min(_MAX_WORKERS, len(syms)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut = {ex.submit(_scan_one, s, wanted): s for s in syms}
                for f in as_completed(fut):
                    s = fut[f]
                    try:
                        r = f.result()
                        if r.get("triggered"):
                            hits.append(r)
                    except Exception as e:
                        errors.append({"symbol": s.upper(), "error": f"{type(e).__name__}: {e}"})

            # summary count by condition
            by_condition: dict[str, int] = {c: 0 for c in _ALL_CONDITIONS if c in wanted}
            for h in hits:
                for a in h["triggered"]:
                    by_condition[a["condition"]] = by_condition.get(a["condition"], 0) + 1
            by_condition = {k: v for k, v in by_condition.items() if v > 0}

            hits.sort(key=lambda x: len(x["triggered"]), reverse=True)
            scanned_ok = len(syms) - len(errors)

            # market-wide verdict
            bullish = sum(by_condition.get(c, 0) for c in
                          ("breakout_up", "golden_cross", "macd_bull_cross", "oversold", "near_support"))
            bearish = sum(by_condition.get(c, 0) for c in
                          ("breakdown", "death_cross", "macd_bear_cross", "overbought", "near_resistance"))
            if bullish > bearish * 1.3:
                verdict = "BULLISH_TILT"
            elif bearish > bullish * 1.3:
                verdict = "BEARISH_TILT"
            else:
                verdict = "MIXED"

            top = by_condition and max(by_condition, key=by_condition.get) or None
            summary_line = (
                f"{len(hits)}/{scanned_ok} symbols triggered >=1 alert; "
                f"most common: {top} ({by_condition.get(top)})" if top
                else f"no alerts across {scanned_ok} scanned symbols"
            )

            result = {
                "verdict": verdict,
                "summary": summary_line,
                "source": source,
                "symbols_requested": len(syms),
                "symbols_scanned_ok": scanned_ok,
                "symbols_triggered": len(hits),
                "conditions_checked": sorted(wanted),
                "count_by_condition": by_condition,
                "alerts": hits,
                "methodology": (
                    "Per-symbol tech_engine.technicals(views='all', period='1y') run concurrently. "
                    "Cross conditions (golden/death/macd) are STATE-based with a freshness band "
                    "(faster line above/below slower; 'fresh' if the two are tight) — the exact "
                    "crossover bar is not available from the upstream tool. Breakouts use the 20d "
                    "high/low with a 2% proximity band; breakout_up also requires rvol>1.3."
                ),
            }
            if unknown:
                result["unknown_conditions"] = unknown
            if errors:
                result["errors"] = errors
            if notes:
                result["notes"] = notes
            return dumps(result)
        except Exception as e:
            return dumps({"error": f"{type(e).__name__}: {e}", "symbols": symbols})
