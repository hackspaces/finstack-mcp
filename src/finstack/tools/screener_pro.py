"""
FinStack Screener Pro — multi-factor alpha screens over NSE baskets.

A desk-grade, Claude-first stock screener. Pick a *strategy* (a named factor
recipe), point it at a sector/thematic basket (sector_engine.ALL_BASKETS), a
custom ticker list, or a sensible default, and it fans out CONCURRENTLY across
the constituents, pulls the factor inputs it needs (key ratios + technicals +
risk metrics), builds a transparent composite score, and ranks the candidates.

This is NOT a black box: every ranked name carries the raw factor values that
earned its score plus a one-line "why", and any factor that could not be
fetched for a name is *skipped* (never faked).

Reuses (does not reinvent):
  - data.fundamentals.get_key_ratios          (PE, P/B, ROE, D/E, FCF, margins)
  - data.tech_engine.technicals               (RSI, MACD, VWAP, SMA, RVOL, CMF, 52w)
  - data.quant_engine.compute_risk_metrics    (beta, max drawdown, vol)
  - data.quant_engine.mean_reversion          (hurst regime — mean-reversion screen)
  - data.sector_engine.ALL_BASKETS / list_baskets
  - data.universe.UNIVERSE                     (name lookup)
  - utils.respond.dumps                        (compact, slim JSON)
"""

from __future__ import annotations

import concurrent.futures as _cf

from finstack.utils.respond import dumps as _dumps

from finstack.data.fundamentals import get_key_ratios
from finstack.data.tech_engine import technicals
from finstack.data.quant_engine import compute_risk_metrics, mean_reversion
from finstack.data import sector_engine as _se
from finstack.data.universe import UNIVERSE


# ----------------------------------------------------------------------------- caps
_MAX_SYMBOLS = 40          # hard cap on constituents fetched per call (512MB host)
_MAX_WORKERS = 8           # concurrent fetch fan-out
_PER_NAME_TIMEOUT = 25     # seconds budget per symbol's full fetch bundle

# Default universe when no basket / symbols given: Nifty-50-ish large-cap liquid set.
_DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC",
    "SBIN", "LT", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "MARUTI",
    "SUNPHARMA", "TITAN", "ASIANPAINT", "NTPC", "ULTRACEMCO", "POWERGRID",
    "M&M", "TATAMOTORS", "WIPRO", "HCLTECH", "NESTLEIND",
]


# ----------------------------------------------------------------------------- strategy catalog
STRATEGIES = {
    "quality_value": {
        "label": "Quality at a Reasonable Price",
        "thesis": "Decent ROE + reasonable PE + positive 6m momentum + low leverage.",
        "needs": ["ratios", "tech", "risk"],
    },
    "momentum_breakout": {
        "label": "Momentum Breakout",
        "thesis": "Near 52w high + elevated relative volume + MACD bullish + positive 3m return.",
        "needs": ["ratios", "tech"],
    },
    "mean_reversion": {
        "label": "Oversold Mean-Reversion",
        "thesis": "RSI oversold + price below VWAP/SMA50 + accumulation (CMF>=0) + mean-reverting regime.",
        "needs": ["tech", "meanrev"],
    },
    "low_vol_quality": {
        "label": "Low-Volatility Quality",
        "thesis": "Low beta + strong ROE/positive FCF + shallow drawdown.",
        "needs": ["ratios", "tech", "risk"],
    },
}


# ----------------------------------------------------------------------------- helpers
def _norm(v, lo, hi):
    """Min-max normalise v into 0..1, clamped. None-safe."""
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def _inv_norm(v, lo, hi):
    """Like _norm but lower is better (e.g. PE, beta, drawdown)."""
    n = _norm(v, lo, hi)
    return None if n is None else 1.0 - n


def _safe(d, *path):
    """Walk nested dicts; return None on any miss / non-dict."""
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _is_err(d):
    return (not isinstance(d, dict)) or bool(d.get("error"))


def _resolve_symbols(basket: str, symbols: str):
    """Return (symbol_list, source_label, note). symbols= wins, then basket=, else default."""
    note = None
    if symbols.strip():
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        return syms, "custom_list", note
    b = basket.strip()
    if b:
        if b not in _se.ALL_BASKETS:
            return [], f"basket:{b}", f"unknown basket '{b}' — call screen_pro(strategy='list') for ideas"
        return list(_se.ALL_BASKETS[b]["symbols"]), f"basket:{b}", note
    return list(_DEFAULT_SYMBOLS), "default_largecap", "no basket/symbols given — screening default large-cap set"


# ----------------------------------------------------------------------------- per-symbol fetch
def _fetch_bundle(symbol: str, needs: set) -> dict:
    """Fetch exactly the inputs a strategy needs for one symbol. Fail-loud per piece."""
    out = {"symbol": symbol.upper(), "name": UNIVERSE.get(symbol.upper(), ""), "errors": {}}
    if "ratios" in needs:
        try:
            r = get_key_ratios(symbol)
            out["ratios"] = None if _is_err(r) else r
            if _is_err(r):
                out["errors"]["ratios"] = (r or {}).get("message", "fetch failed")
        except Exception as e:
            out["errors"]["ratios"] = f"{type(e).__name__}: {e}"
    if "tech" in needs:
        try:
            t = technicals(symbol, views="all", period="1y")
            out["tech"] = None if _is_err(t) else t
            if _is_err(t):
                out["errors"]["tech"] = (t or {}).get("error", "fetch failed")
        except Exception as e:
            out["errors"]["tech"] = f"{type(e).__name__}: {e}"
    if "risk" in needs:
        try:
            rk = compute_risk_metrics(symbol, period="1y")
            out["risk"] = None if _is_err(rk) else rk
            if _is_err(rk):
                out["errors"]["risk"] = (rk or {}).get("error", "fetch failed")
        except Exception as e:
            out["errors"]["risk"] = f"{type(e).__name__}: {e}"
    if "meanrev" in needs:
        try:
            mr = mean_reversion(symbol)
            out["meanrev"] = None if _is_err(mr) else mr
            if _is_err(mr):
                out["errors"]["meanrev"] = (mr or {}).get("error", "fetch failed")
        except Exception as e:
            out["errors"]["meanrev"] = f"{type(e).__name__}: {e}"
    return out


# ----------------------------------------------------------------------------- 52w / momentum from tech levels
def _pct_off_52w_high(tech: dict):
    """Distance below the 52w high using tech levels as a proxy (50d resistance -> 1y high)."""
    price = _safe(tech, "indicators", "price")
    hi = _safe(tech, "levels", "recent_resistance_50d")
    if price is None or not hi:
        return None
    return round((price / hi - 1.0) * 100.0, 2)   # 0 = at high, negative = below


def _momentum_pct(tech: dict, ndays_key: str):
    """Approx momentum using SMA cross-overs available in tech (no extra fetch)."""
    # price vs sma50 ~ 3m proxy; price vs sma200 ~ longer proxy
    ind = _safe(tech, "indicators") or {}
    price = ind.get("price")
    ref = ind.get(ndays_key)
    if price is None or not ref:
        return None
    return round((price / ref - 1.0) * 100.0, 2)


# ----------------------------------------------------------------------------- scorers (one per strategy)
def _score_quality_value(b: dict) -> dict | None:
    ratios, tech = b.get("ratios"), b.get("tech")
    if not ratios and not tech:
        return None
    roe = _safe(ratios, "profitability", "roe")            # decimal (0.18 = 18%)
    pe = _safe(ratios, "valuation", "pe_trailing")
    de = _safe(ratios, "financial_health", "debt_to_equity")  # yfinance: percent (45 = 0.45x)
    mom6 = _momentum_pct(tech or {}, "sma200")             # ~6-12m proxy
    factors, parts, why = {}, [], []

    s_roe = _norm(roe, 0.05, 0.30)
    if s_roe is not None:
        factors["roe_pct"] = round(roe * 100, 2); parts.append(s_roe * 0.30)
        if roe >= 0.15:
            why.append(f"ROE {roe*100:.0f}%")
    s_pe = _inv_norm(pe, 8, 45)
    if s_pe is not None:
        factors["pe_trailing"] = round(pe, 2); parts.append(s_pe * 0.25)
        if pe <= 25:
            why.append(f"PE {pe:.0f}")
    s_de = _inv_norm(de, 0, 150)
    if s_de is not None:
        factors["debt_to_equity"] = round(de, 2); parts.append(s_de * 0.20)
        if de <= 60:
            why.append("low debt")
    s_mom = _norm(mom6, -10, 30)
    if s_mom is not None:
        factors["mom_vs_sma200_pct"] = mom6; parts.append(s_mom * 0.25)
        if mom6 > 0:
            why.append("+ momentum")
    if not parts:
        return None
    return {"factors": factors, "score": round(sum(parts) / 1.0 * 100, 1),
            "why": ", ".join(why) or "partial factor coverage"}


def _score_momentum_breakout(b: dict) -> dict | None:
    tech = b.get("tech")
    if not tech:
        return None
    off_high = _pct_off_52w_high(tech)
    rvol = _safe(tech, "volume", "rvol")
    macd = _safe(tech, "signals", "macd")
    mom3 = _momentum_pct(tech, "sma50")          # ~3m proxy
    factors, parts, why = {}, [], []

    s_high = _norm(off_high, -15, 0)             # closer to 0 (the high) = better
    if s_high is not None:
        factors["pct_off_52w_high"] = off_high; parts.append(s_high * 0.30)
        if off_high >= -3:
            why.append("near 52w high")
    s_rvol = _norm(rvol, 1.0, 3.0)
    if s_rvol is not None:
        factors["rvol"] = round(rvol, 2); parts.append(s_rvol * 0.25)
        if rvol >= 1.5:
            why.append(f"RVOL {rvol:.1f}x")
    if macd is not None:
        bull = (macd == "bullish")
        factors["macd"] = macd; parts.append((1.0 if bull else 0.0) * 0.20)
        if bull:
            why.append("MACD bull")
    s_mom = _norm(mom3, 0, 25)
    if s_mom is not None:
        factors["mom_vs_sma50_pct"] = mom3; parts.append(s_mom * 0.25)
        if mom3 > 0:
            why.append(f"+{mom3:.0f}% vs SMA50")
    if not parts:
        return None
    return {"factors": factors, "score": round(sum(parts) * 100, 1),
            "why": ", ".join(why) or "partial factor coverage"}


def _score_mean_reversion(b: dict) -> dict | None:
    tech, mr = b.get("tech"), b.get("meanrev")
    if not tech:
        return None
    rsi = _safe(tech, "indicators", "rsi14")
    price = _safe(tech, "indicators", "price")
    vwap = _safe(tech, "volume", "vwap_20")
    sma50 = _safe(tech, "indicators", "sma50")
    cmf = _safe(tech, "volume", "cmf_20")
    regime = _safe(mr, "regime")
    zscore = _safe(mr, "zscore_50d") if isinstance(mr, dict) else None
    factors, parts, why = {}, [], []

    s_rsi = _inv_norm(rsi, 20, 50)               # lower RSI (more oversold) = better
    if s_rsi is not None:
        factors["rsi14"] = round(rsi, 1); parts.append(s_rsi * 0.30)
        if rsi <= 35:
            why.append(f"RSI {rsi:.0f} oversold")
    # price below VWAP & SMA50 (discount)
    if price is not None and (vwap or sma50):
        below = 0.0; cnt = 0
        if vwap:
            cnt += 1; below += 1.0 if price < vwap else 0.0
            factors["price_vs_vwap20_pct"] = round((price / vwap - 1) * 100, 2)
        if sma50:
            cnt += 1; below += 1.0 if price < sma50 else 0.0
            factors["price_vs_sma50_pct"] = round((price / sma50 - 1) * 100, 2)
        if cnt:
            parts.append((below / cnt) * 0.25)
            if below == cnt:
                why.append("below VWAP & SMA50")
    if cmf is not None:
        factors["cmf_20"] = round(cmf, 3); parts.append((1.0 if cmf >= 0 else 0.0) * 0.20)
        if cmf >= 0:
            why.append("CMF accumulation")
    if regime is not None:
        factors["regime"] = regime
        if zscore is not None:
            factors["zscore"] = round(zscore, 2)
        parts.append((1.0 if regime == "mean_reverting" else 0.0) * 0.25)
        if regime == "mean_reverting":
            why.append("mean-reverting regime")
    if not parts:
        return None
    return {"factors": factors, "score": round(sum(parts) * 100, 1),
            "why": ", ".join(why) or "partial factor coverage"}


def _score_low_vol_quality(b: dict) -> dict | None:
    ratios, tech, risk = b.get("ratios"), b.get("tech"), b.get("risk")
    if not risk and not ratios:
        return None
    beta = _safe(risk, "beta")
    mdd = _safe(risk, "max_drawdown")            # negative number, e.g. -0.32
    roe = _safe(ratios, "profitability", "roe")
    fcf = _safe(ratios, "financial_health", "free_cash_flow")
    factors, parts, why = {}, [], []

    s_beta = _inv_norm(beta, 0.4, 1.6)           # lower beta = better
    if s_beta is not None:
        factors["beta"] = round(beta, 2); parts.append(s_beta * 0.30)
        if beta <= 0.9:
            why.append(f"beta {beta:.2f}")
    s_mdd = _norm(mdd, -0.60, -0.10)             # shallower (closer to 0) = better
    if s_mdd is not None:
        factors["max_drawdown_pct"] = round(mdd * 100, 1); parts.append(s_mdd * 0.25)
        if mdd >= -0.30:
            why.append("shallow drawdown")
    s_roe = _norm(roe, 0.05, 0.30)
    if s_roe is not None:
        factors["roe_pct"] = round(roe * 100, 2); parts.append(s_roe * 0.25)
        if roe >= 0.15:
            why.append(f"ROE {roe*100:.0f}%")
    if fcf is not None:
        pos = fcf > 0
        factors["free_cash_flow"] = fcf; parts.append((1.0 if pos else 0.0) * 0.20)
        if pos:
            why.append("positive FCF")
    if not parts:
        return None
    return {"factors": factors, "score": round(sum(parts) * 100, 1),
            "why": ", ".join(why) or "partial factor coverage"}


_SCORERS = {
    "quality_value": _score_quality_value,
    "momentum_breakout": _score_momentum_breakout,
    "mean_reversion": _score_mean_reversion,
    "low_vol_quality": _score_low_vol_quality,
}


# ----------------------------------------------------------------------------- verdict
def _verdict(ranked: list, strategy: str) -> dict:
    if not ranked:
        return {"read": "NO_CANDIDATES",
                "comment": "No names cleared with enough factor coverage to score."}
    top = ranked[0]
    n_strong = sum(1 for r in ranked if r["score"] >= 60)
    band = ("RICH OPPORTUNITY SET" if n_strong >= 5 else
            "SELECTIVE" if n_strong >= 1 else "THIN")
    return {
        "read": band,
        "strategy": STRATEGIES[strategy]["label"],
        "top_pick": {"symbol": top["symbol"], "name": top.get("name", ""),
                     "score": top["score"], "why": top.get("why", "")},
        "strong_count": n_strong,
        "comment": f"{n_strong} name(s) scored >=60/100 on the "
                   f"{STRATEGIES[strategy]['label']} recipe; ranked best-first below.",
    }


# ----------------------------------------------------------------------------- tool registration
def register_screener_pro_tools(mcp):
    """Register the multi-factor pro screener tool."""

    @mcp.tool()
    def screen_pro(strategy: str = "list", basket: str = "",
                   symbols: str = "", limit: int = 20) -> str:
        """Desk-grade multi-factor alpha screen across an NSE basket or custom list.

        Picks a named *strategy* (factor recipe), fans out CONCURRENTLY over the
        constituents (capped at ~40 for a 512MB host), pulls only the factor
        inputs that strategy needs (key ratios / technicals / risk metrics),
        builds a transparent 0-100 composite score, and returns the ranked
        candidates — each with its raw factor values, score, and a one-line
        "why" — plus a synthesized verdict. Factors that cannot be fetched for a
        name are skipped (never fabricated); names with zero usable factors drop.

        Args:
            strategy: which screen to run —
                - "list"              : (default) show the available strategies + this help.
                - "quality_value"     : decent ROE + reasonable PE + positive momentum + low debt.
                - "momentum_breakout" : near 52w high + elevated RVOL + MACD bullish + +3m return.
                - "mean_reversion"    : RSI oversold + below VWAP/SMA50 + CMF>=0 + mean-reverting.
                - "low_vol_quality"   : low beta + strong ROE/positive FCF + shallow drawdown.
            basket: an ALL_BASKETS name (e.g. "nse_information_technology",
                "fluorochemicals_refrigerants"). Ignored if symbols= is given.
            symbols: comma-separated custom ticker list (e.g. "TCS,INFY,WIPRO").
                Takes priority over basket=.
            limit: max ranked candidates to return (default 20).

        If neither basket nor symbols is supplied, a default large-cap set is
        screened. Universe is capped at ~40 names per call (noted in output).

        Examples:
            screen_pro(strategy="list")
            screen_pro(strategy="quality_value", basket="nse_information_technology")
            screen_pro(strategy="momentum_breakout", basket="nse_capital_goods", limit=10)
            screen_pro(strategy="mean_reversion", symbols="RELIANCE,INFY,HDFCBANK,SBIN,ITC")
            screen_pro(strategy="low_vol_quality", basket="nse_fast_moving_consumer_goods")
        """
        strat = (strategy or "list").strip().lower()

        # ---- catalog / list mode
        if strat == "list" or strat not in STRATEGIES:
            payload = {
                "tool": "screen_pro",
                "available_strategies": {
                    k: {"label": v["label"], "thesis": v["thesis"]}
                    for k, v in STRATEGIES.items()
                },
                "usage": "screen_pro(strategy=<name>, basket=<ALL_BASKETS name> OR "
                         "symbols='A,B,C', limit=20)",
                "example_baskets": [n for n in list(_se.ALL_BASKETS.keys())[:12]],
                "total_baskets": len(_se.ALL_BASKETS),
                "caps": {"max_symbols_per_call": _MAX_SYMBOLS, "workers": _MAX_WORKERS},
            }
            if strat not in STRATEGIES and strat != "list":
                payload["note"] = f"unknown strategy '{strategy}' — pick one above"
            return _dumps(payload)

        # ---- resolve universe
        syms, source, note = _resolve_symbols(basket, symbols)
        if not syms:
            return _dumps({"error": note or "no symbols resolved", "source": source,
                           "hint": "pass a valid basket= or symbols="})

        capped = False
        if len(syms) > _MAX_SYMBOLS:
            syms = syms[:_MAX_SYMBOLS]
            capped = True

        needs = set(STRATEGIES[strat]["needs"])
        scorer = _SCORERS[strat]

        # ---- concurrent fetch fan-out (one failure never sinks the call)
        bundles = []
        fetch_errors = []
        with _cf.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            fut = {ex.submit(_fetch_bundle, s, needs): s for s in syms}
            for f in _cf.as_completed(fut):
                s = fut[f]
                try:
                    bundles.append(f.result(timeout=_PER_NAME_TIMEOUT))
                except Exception as e:
                    fetch_errors.append({"symbol": s, "error": f"{type(e).__name__}: {e}"})

        # ---- score + rank
        ranked, skipped = [], []
        for b in bundles:
            try:
                sc = scorer(b)
            except Exception as e:
                skipped.append({"symbol": b.get("symbol"), "reason": f"score error: {e}"})
                continue
            if sc is None:
                skipped.append({"symbol": b.get("symbol"),
                                "reason": "no usable factors",
                                "errors": b.get("errors") or None})
                continue
            row = {"symbol": b["symbol"], "name": b.get("name", ""),
                   "score": sc["score"], "why": sc["why"], "factors": sc["factors"]}
            if b.get("errors"):
                row["skipped_factors"] = list(b["errors"].keys())
            ranked.append(row)

        ranked.sort(key=lambda r: r["score"], reverse=True)
        ranked = ranked[: max(1, int(limit))]

        notes = []
        if note:
            notes.append(note)
        if capped:
            notes.append(f"universe capped to first {_MAX_SYMBOLS} constituents "
                         "(512MB host budget)")
        notes.append("scores are composite 0-100 from normalised factors; "
                     "factors unavailable for a name are skipped, not faked")

        result = {
            "strategy": strat,
            "strategy_label": STRATEGIES[strat]["label"],
            "thesis": STRATEGIES[strat]["thesis"],
            "universe": {"source": source, "screened": len(syms),
                         "scored": len(ranked), "skipped": len(skipped)},
            "verdict": _verdict(ranked, strat),
            "candidates": ranked,
            "skipped": skipped[:15] or None,
            "fetch_errors": fetch_errors or None,
            "notes": notes,
        }
        return _dumps(result)

    return screen_pro
