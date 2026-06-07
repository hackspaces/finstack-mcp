"""
FinStack Market Regime tool — the strategist's top-down read.

One desk-grade tool, market_regime(), that fuses four independent lenses into a
single, decision-grade regime call for the Indian market:

  1. MACRO     — live official series (CPI, GDP, short-term rate, real rate) with
                 provenance/freshness stamps (finstack.data.macro_live.get_macro).
  2. NIFTY TREND — Nifty 50 (^NSEI) price vs its 50/200-day moving averages plus
                 RSI/MACD bias (finstack.data.tech_engine.technicals).
  3. VOLATILITY — India VIX (^INDIAVIX) latest level + a fear/complacency read
                 (finstack.data.quant_engine._fetch_close).
  4. BREADTH   — sector rotation leaders/laggards over 3m
                 (finstack.data.sector_engine.rotation).

It then synthesizes a RISK_ON / NEUTRAL / RISK_OFF label from a transparent,
weighted scorecard, and translates that into recommended sector tilts (rotation
leaders that are consistent with the regime) and the key risks to the thesis.

Each lens is gathered concurrently and wrapped in its own try/except, so a single
failing series never sinks the whole call — it is reported as an error note and
simply abstains from the score (fail-loud per section, never fabricate numbers).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from finstack.utils.respond import dumps as _dumps

from finstack.data import macro_live as _macro
from finstack.data import tech_engine as _tech
from finstack.data import quant_engine as _quant
from finstack.data import sector_engine as _sector


# --- VIX interpretation thresholds (India VIX convention) -----------------
_VIX_COMPLACENT = 13.0   # < 13 -> complacency / low fear
_VIX_FEAR = 20.0         # > 20 -> elevated fear / stress

# macro indicators we pull for the regime read
_MACRO_SET = ["cpi_inflation", "gdp_growth", "short_term_rate", "real_interest_rate"]


# ==========================================================================
# Lens 1: macro
# ==========================================================================
def _gather_macro() -> dict:
    """Live macro snapshot + a coarse rate-direction read for the score."""
    raw = _macro.get_macro(_MACRO_SET, country="IN")
    ind = raw.get("indicators", {}) or {}

    def _val(name):
        item = ind.get(name) or {}
        return item.get("value"), item

    cpi, cpi_i = _val("cpi_inflation")
    gdp, gdp_i = _val("gdp_growth")
    short_rate, _ = _val("short_term_rate")
    real_rate, _ = _val("real_interest_rate")

    # Coarse, honest read: a high real rate is a headwind (tight money = risk-off
    # tilt); a clearly negative real rate is accommodative (risk-on tilt).
    rate_bias = None
    if real_rate is not None:
        if real_rate >= 2.0:
            rate_bias = "tight"        # restrictive -> risk-off lean
        elif real_rate <= 0.0:
            rate_bias = "accommodative"  # loose -> risk-on lean
        else:
            rate_bias = "neutral"

    growth_bias = None
    if gdp is not None:
        growth_bias = "expanding" if gdp >= 5.0 else ("slowing" if gdp < 3.0 else "moderate")

    return {
        "cpi_inflation": cpi, "gdp_growth": gdp,
        "short_term_rate": short_rate, "real_interest_rate": real_rate,
        "rate_bias": rate_bias, "growth_bias": growth_bias,
        "fetched_at": raw.get("fetched_at"),
        "_stale_flags": {k: (ind.get(k) or {}).get("is_stale") for k in _MACRO_SET},
        "raw_indicators": ind,
        "note": "Official annual/periodic data; see each item's as_of/is_stale.",
    }


# ==========================================================================
# Lens 2: Nifty trend
# ==========================================================================
def _gather_nifty() -> dict:
    """Nifty 50 trend vs 50/200dma + RSI/MACD bias."""
    t = _tech.technicals("^NSEI", views="summary,signals,indicators", period="1y")
    if "error" in t:
        raise ValueError(t["error"])
    ind = t.get("indicators", {}) or {}
    sig = t.get("signals", {}) or {}
    summ = t.get("summary", {}) or {}

    price = ind.get("price")
    sma50 = ind.get("sma50")
    sma200 = ind.get("sma200")

    above_200 = (price is not None and sma200 is not None and price > sma200)
    above_50 = (price is not None and sma50 is not None and price > sma50)
    golden_cross = sig.get("golden_cross_50_200")

    # distance from the 200dma — the single most important regime gauge
    pct_from_200dma = None
    if price is not None and sma200:
        pct_from_200dma = round((price / sma200 - 1) * 100, 2)

    trend_bias = "up" if above_200 else "down"
    return {
        "price": price, "sma50": sma50, "sma200": sma200,
        "above_50dma": above_50, "above_200dma": above_200,
        "pct_from_200dma": pct_from_200dma,
        "golden_cross_50_200": golden_cross,
        "rsi14": ind.get("rsi14"), "rsi_state": sig.get("rsi"),
        "macd": sig.get("macd"), "ma_stack": sig.get("ma_stack"),
        "bias": summ.get("bias"), "trend": summ.get("trend"),
        "trend_bias": trend_bias,
    }


# ==========================================================================
# Lens 3: India VIX
# ==========================================================================
def _gather_vix() -> dict:
    """Latest India VIX + a complacency/fear interpretation."""
    close = _quant._fetch_close("^INDIAVIX", "3mo")
    latest = float(close.iloc[-1])
    # short trend: latest vs ~20d ago
    prev = float(close.iloc[-21]) if len(close) >= 21 else None
    change_pct = round((latest / prev - 1) * 100, 2) if prev else None

    if latest < _VIX_COMPLACENT:
        interp, vol_state = "low / complacent", "calm"
    elif latest > _VIX_FEAR:
        interp, vol_state = "elevated / fear", "stressed"
    else:
        interp, vol_state = "normal", "normal"

    return {
        "india_vix": round(latest, 2),
        "vix_20d_ago": round(prev, 2) if prev else None,
        "change_20d_pct": change_pct,
        "interpretation": interp,
        "vol_state": vol_state,
        "thresholds": {"complacent_below": _VIX_COMPLACENT, "fear_above": _VIX_FEAR},
    }


# ==========================================================================
# Lens 4: sector rotation / breadth
# ==========================================================================
def _gather_rotation() -> dict:
    """3m sector rotation: leaders, laggards, and breadth vs Nifty."""
    r = _sector.rotation(None, "3m")
    if "error" in r:
        raise ValueError(r.get("error", "rotation unavailable"))
    leaders = r.get("leaders", []) or []
    laggards = r.get("laggards", []) or []
    allb = r.get("all", []) or []
    nifty_ret = r.get("nifty_return")

    # breadth: share of baskets beating Nifty over the window
    beating = None
    if allb and nifty_ret is not None:
        rk = f"return_3m"
        n_beat = sum(1 for b in allb if b.get(rk) is not None and b[rk] > nifty_ret)
        beating = round(n_beat / len(allb) * 100, 1)

    return {
        "lookback": r.get("lookback"),
        "nifty_return_3m": nifty_ret,
        "baskets_ranked": r.get("baskets_ranked"),
        "breadth_pct_beating_nifty": beating,
        "leaders": leaders,
        "laggards": laggards,
    }


# ==========================================================================
# Synthesis
# ==========================================================================
def _synthesize(macro, nifty, vix, rotation, errors) -> dict:
    """Combine available lenses into a weighted RISK_ON/NEUTRAL/RISK_OFF score.

    Each contributing lens votes in [-1, +1]; we average the votes that are
    actually available (a failed lens abstains rather than scoring 0), then map
    the blended score onto a regime band. Weights reflect a PM's priors: trend
    and volatility dominate the tactical regime; breadth confirms; macro is the
    slow-moving backdrop.
    """
    votes = []          # (label, weight, vote, rationale)

    # --- trend vote (weight 0.35) ---
    if nifty:
        v, why = 0.0, []
        if nifty.get("above_200dma"):
            v += 0.6; why.append("Nifty > 200dma")
        else:
            v -= 0.6; why.append("Nifty < 200dma")
        if nifty.get("above_50dma"):
            v += 0.2; why.append("> 50dma")
        else:
            v -= 0.2; why.append("< 50dma")
        if nifty.get("golden_cross_50_200"):
            v += 0.2; why.append("golden cross")
        else:
            v -= 0.2; why.append("death cross")
        v = max(-1.0, min(1.0, v))
        votes.append(("trend", 0.35, v, "; ".join(why)))

    # --- volatility vote (weight 0.30) ---
    if vix:
        lvl = vix.get("india_vix")
        if lvl is not None:
            if lvl < _VIX_COMPLACENT:
                v, why = 0.7, f"VIX {lvl} < {_VIX_COMPLACENT} (calm)"
            elif lvl > _VIX_FEAR:
                v, why = -0.8, f"VIX {lvl} > {_VIX_FEAR} (fear)"
            else:
                # linear lean within the normal band
                mid = (_VIX_COMPLACENT + _VIX_FEAR) / 2
                v = round(-(lvl - mid) / (_VIX_FEAR - mid) * 0.4, 3)
                why = f"VIX {lvl} normal band"
            votes.append(("volatility", 0.30, v, why))

    # --- breadth vote (weight 0.20) ---
    if rotation:
        bp = rotation.get("breadth_pct_beating_nifty")
        if bp is not None:
            v = round((bp - 50.0) / 50.0, 3)  # >50% beating -> positive
            v = max(-1.0, min(1.0, v))
            votes.append(("breadth", 0.20, v,
                          f"{bp}% of sectors beating Nifty (3m)"))

    # --- macro/rate vote (weight 0.15) ---
    if macro:
        v, why = 0.0, []
        rb = macro.get("rate_bias")
        if rb == "accommodative":
            v += 0.5; why.append("accommodative real rate")
        elif rb == "tight":
            v -= 0.5; why.append("tight real rate")
        gb = macro.get("growth_bias")
        if gb == "expanding":
            v += 0.3; why.append("growth expanding")
        elif gb == "slowing":
            v -= 0.3; why.append("growth slowing")
        if why:
            v = max(-1.0, min(1.0, v))
            votes.append(("macro", 0.15, v, "; ".join(why)))

    if not votes:
        return {
            "regime": "UNKNOWN",
            "score": None,
            "confidence": "none",
            "rationale": ["No lens available — all data series failed. See errors."],
            "component_votes": [],
        }

    wsum = sum(w for _, w, _, _ in votes)
    score = round(sum(w * v for _, w, v, _ in votes) / wsum, 3)

    if score >= 0.30:
        regime = "RISK_ON"
    elif score <= -0.30:
        regime = "RISK_OFF"
    else:
        regime = "NEUTRAL"

    # confidence scales with how many lenses voted (coverage)
    coverage = round(wsum, 2)  # max 1.0 if all four present
    if coverage >= 0.85:
        conf = "high"
    elif coverage >= 0.5:
        conf = "medium"
    else:
        conf = "low"

    rationale = [f"{lbl}: vote {v:+.2f} (w{w}) — {why}" for lbl, w, v, why in votes]

    return {
        "regime": regime,
        "score": score,
        "score_scale": "-1 (max risk-off) .. +1 (max risk-on); bands: <=-0.30 OFF, >=+0.30 ON",
        "confidence": conf,
        "coverage": coverage,
        "rationale": rationale,
        "component_votes": [
            {"lens": lbl, "weight": w, "vote": v, "why": why}
            for lbl, w, v, why in votes
        ],
    }


def _build_tilts(verdict, rotation) -> dict:
    """Recommended sector tilts: rotation leaders consistent with the regime.

    RISK_ON  -> lean into momentum leaders, underweight laggards.
    RISK_OFF -> keep only the strongest leaders, flag a defensive posture, and
                underweight the high-beta laggards.
    NEUTRAL  -> selective: top 3 leaders only.
    """
    regime = verdict.get("regime")
    if not rotation:
        return {"note": "rotation unavailable — no tilt guidance"}

    leaders = rotation.get("leaders", []) or []
    laggards = rotation.get("laggards", []) or []
    lead_names = [b.get("basket") for b in leaders]
    lag_names = [b.get("basket") for b in laggards]

    if regime == "RISK_ON":
        overweight = lead_names[:4]
        posture = ("Lean into momentum: overweight the 3m rotation leaders; "
                   "underweight the laggards.")
        underweight = lag_names[:3]
    elif regime == "RISK_OFF":
        overweight = lead_names[:2]
        posture = ("Defensive posture: raise cash / quality, keep only the "
                   "strongest leaders, avoid high-beta laggards. Trim gross.")
        underweight = lag_names[:4]
    else:  # NEUTRAL / UNKNOWN
        overweight = lead_names[:3]
        posture = ("Selective: stay close to benchmark; rotate into the top "
                   "leaders only, fund from the deepest laggards.")
        underweight = lag_names[:2]

    return {
        "posture": posture,
        "overweight": [b for b in leaders if b.get("basket") in overweight],
        "underweight": [b for b in laggards if b.get("basket") in underweight],
    }


def _key_risks(macro, nifty, vix, verdict) -> list:
    """Honest list of what could break the call."""
    risks = []
    if vix and vix.get("india_vix") is not None:
        if vix["india_vix"] > _VIX_FEAR:
            risks.append("Volatility already elevated — sharp drawdowns / gap risk; size down.")
        elif vix["india_vix"] < _VIX_COMPLACENT:
            risks.append("Complacency (low VIX) — crowded longs vulnerable to a vol shock.")
    if nifty and nifty.get("pct_from_200dma") is not None:
        d = nifty["pct_from_200dma"]
        if d is not None and d > 12:
            risks.append(f"Nifty extended {d}% above 200dma — mean-reversion / correction risk.")
        elif d is not None and d < -10:
            risks.append(f"Nifty {abs(d)}% below 200dma — falling-knife risk; trend not yet repaired.")
    if macro:
        if macro.get("rate_bias") == "tight":
            risks.append("Tight real rates — liquidity headwind; rate path is the key swing factor.")
        cpi = macro.get("cpi_inflation")
        if cpi is not None and cpi > 6:
            risks.append(f"CPI {cpi}% above comfort band — policy could stay restrictive.")
        if any(macro.get("_stale_flags", {}).values()):
            risks.append("Some macro series are stale (annual/lagged) — treat backdrop as slow-moving context, not a trigger.")
    if verdict.get("confidence") in ("low", "none"):
        risks.append("Low data coverage — regime read is tentative; confirm before sizing up.")
    if not risks:
        risks.append("No acute risk flag from available lenses; watch VIX and the 200dma for regime change.")
    return risks


def register_regime_tools(mcp):
    """Register the market_regime strategist tool."""

    @mcp.tool()
    def market_regime() -> str:
        """Top-down market-regime read for India — RISK_ON / NEUTRAL / RISK_OFF.

        The strategist's tactical dashboard. Concurrently fuses four independent
        lenses and synthesizes a single, decision-grade regime call with a
        transparent weighted scorecard, recommended sector tilts, and key risks:

          - MACRO      : live CPI, GDP growth, short-term & real interest rates
                         (World Bank / DBnomics, with as_of + is_stale stamps).
          - NIFTY TREND: ^NSEI vs its 50/200-day MAs, RSI, MACD, golden/death cross.
          - VOLATILITY : ^INDIAVIX latest level (low<13 complacent, >20 fear) + 20d change.
          - BREADTH    : 3-month sector rotation leaders/laggards + % beating Nifty.

        The verdict blends per-lens votes in [-1,+1] (weights: trend 0.35,
        volatility 0.30, breadth 0.20, macro 0.15). A lens that fails to fetch
        ABSTAINS (it does not score 0), and its error is reported — so the call
        is fail-loud per section and never fabricates numbers. Confidence scales
        with how many lenses were available.

        Takes no arguments (the market is the input). Bounded for a 512MB host:
        sector rotation samples the first 15 constituents per basket and the
        VIX/Nifty pulls use <=1y of daily data.

        Returns (JSON): {regime, score, confidence, verdict{...},
        sector_tilts{overweight,underweight,posture}, key_risks[],
        components{macro,nifty_trend,india_vix,sector_rotation}, errors{}}.

        Example:
            market_regime()
            # -> {"regime":"RISK_ON","score":0.41,"confidence":"high",
            #     "sector_tilts":{"overweight":[{"basket":"nse_realty",...}]...},
            #     "components":{"india_vix":{"india_vix":11.8,...}, ...}}
        """
        lenses = {
            "macro": _gather_macro,
            "nifty": _gather_nifty,
            "vix": _gather_vix,
            "rotation": _gather_rotation,
        }
        results = {}
        errors = {}

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(fn): name for name, fn in lenses.items()}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:
                    errors[name] = f"{type(e).__name__}: {e}"

        macro = results.get("macro")
        nifty = results.get("nifty")
        vix = results.get("vix")
        rotation = results.get("rotation")

        verdict = _synthesize(macro, nifty, vix, rotation, errors)
        tilts = _build_tilts(verdict, rotation)
        risks = _key_risks(macro or {}, nifty or {}, vix or {}, verdict)

        out = {
            "as_of": (macro or {}).get("fetched_at"),
            "market": "India (Nifty 50)",
            "regime": verdict.get("regime"),
            "score": verdict.get("score"),
            "confidence": verdict.get("confidence"),
            "verdict": verdict,
            "sector_tilts": tilts,
            "key_risks": risks,
            "components": {
                "macro": macro,
                "nifty_trend": nifty,
                "india_vix": vix,
                "sector_rotation": rotation,
            },
            "caps": "rotation: first 15 constituents/basket; trend/vix: <=1y daily.",
        }
        if errors:
            out["errors"] = errors
            out["data_health"] = (
                f"{len(results)}/4 lenses available; "
                f"{', '.join(errors)} unavailable (abstained from score)."
            )
        return _dumps(out)
