"""
FinStack `trade_setup` — a trader's actionable setup card for an NSE stock.

One configurable tool that fuses the technicals engine (ATR, support/resistance,
pivots, trend, RSI, RVOL/accumulation) with the mean-reversion model (z-score,
half-life, Hurst regime) into a desk-grade plan: direction, a current-price AND a
better pullback entry, an ATR/swing-based stop, R-multiple + level-based targets,
position sizing off a fixed risk budget, and a confluence-driven conviction read.

Reuses (does NOT reinvent):
  - data.tech_engine.technicals(symbol, views="all", period=...)
  - data.quant_engine.mean_reversion(symbol)
Both heavy lifts are wrapped in their own try/except so one failure degrades the
card gracefully rather than sinking the whole call (fail-loud per section).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from finstack.utils.respond import dumps
from finstack.data import tech_engine as te
from finstack.data.quant_engine import mean_reversion


def _f(x):
    """Best-effort float, else None."""
    try:
        if x is None:
            return None
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def _r(x, d=2):
    v = _f(x)
    return round(v, d) if v is not None else None


def register_trade_tools(mcp):
    """Register the trade_setup tool."""

    @mcp.tool()
    def trade_setup(symbol: str, capital: float = 100000,
                    risk_pct: float = 1.0, direction: str = "auto",
                    horizon: str = "swing") -> str:
        """Actionable swing/positional trade setup card for an NSE stock.

        Fuses the technicals engine (ATR, support/resistance, pivots, trend, RSI,
        RVOL/accumulation) with the mean-reversion model (z-score, half-life, Hurst)
        into a single executable plan: direction read, entries (current + pullback),
        an ATR/swing-based stop, R-multiple AND level-based targets, position sizing
        off a fixed risk budget, and a confluence-scored conviction note.

        Sizing math (fixed-fractional risk):
          per_share_risk = abs(entry - stop)
          shares         = floor((capital * risk_pct/100) / per_share_risk)
          capital_deployed = shares * entry   (flagged if it exceeds `capital`)
          risk_amount    = shares * per_share_risk
        Targets T1/T2/T3 are 1R/2R/3R from the chosen entry; the card also reports
        the nearest structural resistance (long) / support (short) so you can trail
        to real levels. risk_reward is measured to T2.

        Args:
            symbol: NSE symbol, e.g. "RELIANCE", "HDFCBANK" (".NS" appended downstream).
            capital: total trading capital in INR for sizing (default 100000).
            risk_pct: percent of capital risked on this trade (default 1.0 = 1%).
            direction: "auto" (infer from trend+momentum), or force "long" / "short".

        Example:
            trade_setup(symbol="RELIANCE")
            trade_setup(symbol="TATAMOTORS", capital=500000, risk_pct=0.75)
            trade_setup(symbol="INFY", direction="short")

        Returns a JSON setup card: verdict, direction, entry, stop, targets,
        position_sizing, risk_reward, confluence + conviction, and the raw evidence
        (technicals snapshot + mean-reversion read). Fails loud per section.
        """
        sym = (symbol or "").strip().upper()
        result: dict = {"tool": "trade_setup", "symbol": sym,
                        "capital_inr": _r(capital), "risk_pct": _r(risk_pct),
                        "requested_direction": (direction or "auto").lower()}
        errors: list[str] = []

        # ---- validate inputs -------------------------------------------------
        if not sym:
            return dumps({"error": "symbol is required"})
        cap = _f(capital)
        rpct = _f(risk_pct)
        if cap is None or cap <= 0:
            return dumps({"error": "capital must be a positive number", "symbol": sym})
        if rpct is None or rpct <= 0 or rpct > 100:
            return dumps({"error": "risk_pct must be in (0, 100]", "symbol": sym})
        d_req = (direction or "auto").lower()
        if d_req not in ("auto", "long", "short"):
            return dumps({"error": "direction must be auto|long|short", "symbol": sym})

        # ---- gather evidence concurrently (each wrapped) ---------------------
        tech: dict | None = None
        mr: dict | None = None

        def _do_tech():
            return te.technicals(sym, views="all", period="1y")

        def _do_mr():
            return mean_reversion(sym)

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(_do_tech): "tech", ex.submit(_do_mr): "mr"}
            for fut in as_completed(futs):
                tag = futs[fut]
                try:
                    out = fut.result()
                    if isinstance(out, dict) and out.get("error"):
                        errors.append(f"{tag}: {out.get('error')}")
                    elif tag == "tech":
                        tech = out
                    else:
                        mr = out
                except Exception as e:
                    errors.append(f"{tag}: {type(e).__name__}: {e}")

        if not tech:
            return dumps({"error": "insufficient technical data — cannot build setup",
                          "symbol": sym, "errors": errors})

        ind = tech.get("indicators", {}) or {}
        sig = tech.get("signals", {}) or {}
        lvl = tech.get("levels", {}) or {}
        summ = tech.get("summary", {}) or {}
        vol = tech.get("volume", {}) or {}

        price = _f(ind.get("price"))
        atr = _f(ind.get("atr14"))
        if price is None or atr is None or atr <= 0:
            return dumps({"error": "missing price/ATR — cannot size a setup",
                          "symbol": sym, "errors": errors})

        rsi = _f(ind.get("rsi14"))
        vwap = _f(vol.get("vwap_20"))
        sma20 = _f(ind.get("sma20"))
        sup20 = _f(lvl.get("recent_support_20d"))
        res20 = _f(lvl.get("recent_resistance_20d"))
        sup50 = _f(lvl.get("recent_support_50d"))
        res50 = _f(lvl.get("recent_resistance_50d"))
        s1, r1 = _f(lvl.get("s1")), _f(lvl.get("r1"))

        # ---- direction -------------------------------------------------------
        trend_dir = sig.get("trend_direction")  # up/down
        bias = summ.get("bias")                  # BULLISH/BEARISH/NEUTRAL
        macd = sig.get("macd")                   # bullish/bearish
        if d_req in ("long", "short"):
            side = d_req
            dir_basis = "forced by caller"
        else:
            up_votes = sum([trend_dir == "up", bias == "BULLISH", macd == "bullish",
                            rsi is not None and rsi >= 50])
            dn_votes = sum([trend_dir == "down", bias == "BEARISH", macd == "bearish",
                            rsi is not None and rsi < 50])
            side = "long" if up_votes >= dn_votes else "short"
            dir_basis = f"auto: long_votes={up_votes} short_votes={dn_votes} (trend={trend_dir}, bias={bias}, macd={macd}, rsi={_r(rsi)})"

        long = side == "long"

        # ---- entries: current + a better pullback near support/VWAP ----------
        # Pullback long: nearest meaningful level below price (VWAP/SMA20/S1/support).
        # Pullback short: nearest meaningful level above price (VWAP/SMA20/R1/resistance).
        if long:
            below = [x for x in (vwap, sma20, s1, sup20) if x is not None and x < price]
            pullback = max(below) if below else _r(price - 0.5 * atr)
        else:
            above = [x for x in (vwap, sma20, r1, res20) if x is not None and x > price]
            pullback = min(above) if above else _r(price + 0.5 * atr)
        pullback = _r(pullback)

        entry = price  # primary actionable entry = current price (chosen for sizing)

        # ---- stop: tighter of ATR-stop vs recent swing level -----------------
        atr_mult = 1.5
        if long:
            atr_stop = entry - atr_mult * atr
            swing = sup20 if (sup20 is not None and sup20 < entry) else None
            # pick the sensible TIGHTER stop = the higher of the two (smaller risk),
            # but never above entry.
            cands = [x for x in (atr_stop, swing) if x is not None and x < entry]
            stop = max(cands) if cands else atr_stop
        else:
            atr_stop = entry + atr_mult * atr
            swing = res20 if (res20 is not None and res20 > entry) else None
            cands = [x for x in (atr_stop, swing) if x is not None and x > entry]
            stop = min(cands) if cands else atr_stop
        stop = _r(stop)
        stop_basis = "1.5*ATR" if stop == _r(atr_stop) else "recent 20d swing (tighter than 1.5*ATR)"

        per_share_risk = _f(abs(entry - stop))
        if per_share_risk is None or per_share_risk <= 0:
            return dumps({"error": "degenerate stop (zero risk per share)", "symbol": sym,
                          "entry": _r(entry), "stop": stop, "errors": errors})

        # ---- targets: R-multiples + nearest structural level -----------------
        if long:
            t1, t2, t3 = entry + per_share_risk, entry + 2 * per_share_risk, entry + 3 * per_share_risk
            struct = [x for x in (res20, res50) if x is not None and x > entry]
            nearest_level = min(struct) if struct else None
        else:
            t1, t2, t3 = entry - per_share_risk, entry - 2 * per_share_risk, entry - 3 * per_share_risk
            struct = [x for x in (sup20, sup50) if x is not None and x < entry]
            nearest_level = max(struct) if struct else None

        # ---- position sizing -------------------------------------------------
        risk_budget = cap * rpct / 100.0
        shares = int(math.floor(risk_budget / per_share_risk))
        capital_deployed = _r(shares * entry)
        risk_amount = _r(shares * per_share_risk)
        deploy_warn = None
        if shares <= 0:
            deploy_warn = "risk budget too small for one share at this stop distance"
        elif capital_deployed and capital_deployed > cap:
            deploy_warn = (f"position notional ₹{capital_deployed:,.0f} exceeds capital "
                           f"₹{cap:,.0f} — would need leverage; size down or widen capital")

        rr_t2 = _r((2 * per_share_risk) / per_share_risk)  # always 2.0 by construction
        # RR to nearest structural level (real-world, not the symmetric R-target)
        rr_to_level = None
        if nearest_level is not None:
            reward = abs(nearest_level - entry)
            rr_to_level = _r(reward / per_share_risk)

        # ---- confluence + conviction ----------------------------------------
        trend_strong = sig.get("trend_strength") == "strong"
        rvol = _f(vol.get("rvol"))
        cmf = _f(vol.get("cmf_20"))
        accumulating = (cmf is not None and cmf > 0)
        distributing = (cmf is not None and cmf < 0)
        mr_signal = (mr or {}).get("signal")
        mr_z = _f((mr or {}).get("zscore_50d"))
        mr_regime = (mr or {}).get("regime")

        confluence = []
        score = 0
        # trend agreement
        if (long and trend_dir == "up") or (not long and trend_dir == "down"):
            confluence.append(f"trend {trend_dir} agrees ({sig.get('trend_strength')})")
            score += 2 if trend_strong else 1
        else:
            confluence.append(f"trend {trend_dir} fights the {side} setup")
            score -= 1
        # momentum
        if (long and macd == "bullish") or (not long and macd == "bearish"):
            confluence.append(f"MACD {macd} confirms")
            score += 1
        # volume / flow
        if rvol is not None and rvol > 1.5:
            confluence.append(f"RVOL {rvol} — unusually active")
            score += 1
        if long and accumulating:
            confluence.append(f"CMF {cmf} > 0 — accumulation")
            score += 1
        elif (not long) and distributing:
            confluence.append(f"CMF {cmf} < 0 — distribution")
            score += 1
        elif long and distributing:
            confluence.append(f"CMF {cmf} < 0 — distribution fights a long")
            score -= 1
        elif (not long) and accumulating:
            confluence.append(f"CMF {cmf} > 0 — accumulation fights a short")
            score -= 1
        # mean-reversion agreement (dip-buy / rip-sell)
        if long and mr_signal == "BUY_DIP":
            confluence.append(f"mean-reversion BUY_DIP (z={mr_z}, {mr_regime})")
            score += 1
        elif (not long) and mr_signal == "SELL_RIP":
            confluence.append(f"mean-reversion SELL_RIP (z={mr_z}, {mr_regime})")
            score += 1
        elif mr_z is not None:
            # extended against entry = caution
            if long and mr_z > 1.5:
                confluence.append(f"stretched: z={mr_z} above mean — chasing a long")
                score -= 1
            elif (not long) and mr_z < -1.5:
                confluence.append(f"stretched: z={mr_z} below mean — chasing a short")
                score -= 1

        conviction = "HIGH" if score >= 4 else ("MEDIUM" if score >= 2 else "LOW")

        # ---- verdict ---------------------------------------------------------
        # ---- expected PnL at each level (₹ and % of capital) ----
        sgn = 1 if long else -1

        def _pnl(level):
            if level is None or shares <= 0:
                return None
            inr = round(shares * sgn * (level - entry))
            return {"price": _r(level), "pnl_inr": inr, "pnl_pct_of_capital": _r(inr / cap * 100)}

        expected_pnl = {
            "stop": _pnl(stop), "T1_1R": _pnl(t1), "T2_2R": _pnl(t2), "T3_3R": _pnl(t3),
            "nearest_level": _pnl(nearest_level),
        }

        # ---- exit plan (horizon-aware) ----
        hz = (horizon or "swing").strip().lower()
        if hz not in ("intraday", "swing", "positional"):
            hz = "swing"
        hl = _f((mr or {}).get("half_life_days"))
        time_stop = {
            "intraday": "Square off by today's close — do not carry overnight.",
            "swing": f"Time-stop: exit if thesis hasn't worked in ~{int(min(hl, 15)) if hl else 10} sessions.",
            "positional": "Hold while trend/thesis intact; review on results or a 200DMA break.",
        }[hz]
        exit_plan = [
            f"HARD STOP ₹{stop} ({stop_basis}) → exit ALL; loss ≈ ₹{risk_amount} ({_r(rpct)}% of capital).",
            f"T1 ₹{_r(t1)} (1R) → book ~50%, trail stop to entry ₹{_r(entry)} (lock breakeven).",
            f"T2 ₹{_r(t2)} (2R, primary) → book ~30%"
            + (f"; PnL ≈ ₹{expected_pnl['T2_2R']['pnl_inr']}." if expected_pnl.get('T2_2R') else "."),
            f"T3 ₹{_r(t3)} (3R) → trail the rest with ATR/SMA20; exit into strength.",
            time_stop,
        ]

        verdict = (f"{conviction} conviction {side.upper()} — entry ~₹{_r(entry)}, "
                   f"pullback ₹{pullback}, stop ₹{stop} ({stop_basis}), "
                   f"T2 ₹{_r(t2)} (2R). Risk ₹{risk_amount} on {shares} sh.")
        if deploy_warn:
            verdict += f" ⚠ {deploy_warn}"

        result.update({
            "verdict": verdict,
            "direction": side,
            "direction_basis": dir_basis,
            "current_price": _r(price),
            "entry": {
                "primary": _r(entry),
                "note": "primary = current price (used for sizing)",
                "better_pullback": pullback,
                "pullback_note": ("buy the dip toward VWAP/SMA20/support"
                                  if long else "sell the rip toward VWAP/SMA20/resistance"),
            },
            "stop": {
                "price": stop,
                "basis": stop_basis,
                "atr14": _r(atr),
                "atr_multiple": atr_mult,
                "per_share_risk": _r(per_share_risk),
            },
            "targets": {
                "T1_1R": _r(t1),
                "T2_2R": _r(t2),
                "T3_3R": _r(t3),
                "nearest_structural_level": _r(nearest_level),
                "rr_to_T2": rr_t2,
                "rr_to_nearest_level": rr_to_level,
            },
            "position_sizing": {
                "capital_inr": _r(cap),
                "risk_pct": _r(rpct),
                "risk_budget_inr": _r(risk_budget),
                "per_share_risk_inr": _r(per_share_risk),
                "shares": shares,
                "capital_deployed_inr": capital_deployed,
                "risk_amount_inr": risk_amount,
                "warning": deploy_warn,
            },
            "horizon": hz,
            "expected_pnl": expected_pnl,
            "exit_plan": exit_plan,
            "conviction": {
                "rating": conviction,
                "score": score,
                "confluence": confluence,
            },
            "evidence": {
                "technicals": {
                    "bias": bias, "trend": summ.get("trend"),
                    "rsi14": _r(rsi), "macd": macd, "adx14": _r(ind.get("adx14")),
                    "sma20": _r(sma20), "sma50": _r(ind.get("sma50")),
                    "sma200": _r(ind.get("sma200")), "vwap_20": _r(vwap),
                    "rvol": rvol, "cmf_20": cmf,
                    "support_20d": _r(sup20), "resistance_20d": _r(res20),
                    "pivot": _r(lvl.get("pivot")),
                },
                "mean_reversion": (mr or {"note": "unavailable"}),
            },
            "caps_and_disclaimer": (
                "Setup uses 1y daily technicals + 50d z-score; ATR-stop = 1.5*ATR vs 20d "
                "swing (tighter wins). Targets are mechanical R-multiples — manage to real "
                "levels. Sizing is fixed-fractional; notional may exceed capital on tight "
                "stops (flagged). NOT investment advice."
            ),
        })
        if errors:
            result["partial_errors"] = errors
        return dumps(result)
