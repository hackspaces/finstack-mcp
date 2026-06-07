"""
FinStack `day_trade` — an intraday scalp / day-trade card for an NSE stock.

Builds a desk-grade intraday plan from 15-minute bars: session VWAP, the opening
range (first 30 min) and its ORB breakout/breakdown triggers, intraday
support/resistance (today's high/low + prior-day high/low), an ATR(14)-on-15m stop
gauge, and RVOL of the current 15m bar. From those it derives a directional bias,
an ORB-based entry, a stop (other side of the opening range OR 1.5x intraday ATR,
whichever is tighter to control risk), 1R/2R + level-based targets, fixed-fractional
position sizing off `capital * risk_pct%`, and an explicit square-off-by-close note.

Reuses (does NOT reinvent):
  - data.chart_engine._hist(symbol, period="5d", interval="15m")   # intraday OHLCV
  - data.nse.get_nse_quote(symbol)                                  # live LTP / day H-L
  - utils.respond.dumps(obj)

Design notes / honesty:
  - Pure pandas/numpy on 15m bars; bounded to a 5-day intraday pull (~130 bars), so
    memory stays trivial on a 512MB host. No multi-symbol fan-out here (single
    instrument), so ThreadPoolExecutor would add no value — the two I/O calls
    (intraday history + live quote) are run concurrently instead.
  - VWAP is a TRUE session VWAP recomputed from the latest day's bars only (cum
    typical*vol / cum vol), NOT a rolling VWAP.
  - Intraday data on free yfinance can lag / be unavailable pre-open or on holidays;
    that path is fail-loud (returns an error envelope, never a fabricated card).
  - The live quote is best-effort enrichment; if it fails the card still builds off
    the 15m bars, with a noted degradation.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from finstack.utils.respond import dumps
from finstack.data.chart_engine import _hist
from finstack.data.nse import get_nse_quote

# Bound the intraday pull. 5 trading days of 15m bars (~25 bars/day, ~125 total)
# is plenty for opening range + prior-day levels + ATR(14) and stays tiny in RAM.
_INTRADAY_PERIOD = "5d"
_INTRADAY_INTERVAL = "15m"
_ATR_N = 14
_OPENING_RANGE_BARS = 2  # first two 15m bars == first 30 minutes


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


def _atr_15m(df: pd.DataFrame, n: int = _ATR_N) -> float | None:
    """Wilder-style ATR on the 15m frame (whole window, not session-reset)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return _f(atr.iloc[-1])


def register_day_trade_tools(mcp):
    """Register the day_trade tool."""

    @mcp.tool()
    def day_trade(symbol: str, capital: float = 100000, risk_pct: float = 0.5) -> str:
        """Intraday scalp / day-trade card for an NSE stock (15-minute bars).

        Computes session VWAP, the opening range (first 30 min) with ORB
        breakout/breakdown triggers, intraday + prior-day support/resistance, an
        ATR(14)-on-15m volatility read, and current-bar RVOL — then assembles a
        complete intraday plan: bias, entry trigger, stop, 1R/2R + level targets,
        position size, and a square-off-by-close note.

        Sizing math (fixed-fractional intraday risk):
          risk_budget    = capital * (risk_pct / 100)        # rupees at risk
          per_share_risk = abs(entry - stop)
          shares         = floor(risk_budget / per_share_risk)
          notional       = shares * entry  (flagged if > capital -> needs MIS leverage)

        Args:
            symbol:   NSE symbol, e.g. "RELIANCE", "TCS", "HDFCBANK" (no .NS needed).
            capital:  Trading capital in INR for sizing. Default 100000 (₹1L).
            risk_pct: Percent of capital to risk on the trade, e.g. 0.5 == 0.5%.
                      Default 0.5 (intraday risk is kept small vs swing).

        Returns:
            dumps(...) JSON: a structured intraday card with a clear `verdict`,
            `bias`, `setup` (entry/stop/targets/sizing), `levels`, `vwap`,
            `opening_range`, `atr`, `rvol`, and `exit_note`. Fail-loud per section.

        Indian example:
            day_trade("RELIANCE", capital=200000, risk_pct=0.5)
            -> a 15m ORB plan for RELIANCE.NS: VWAP ~₹X, opening range high/low,
               breakout entry above OR-high with stop at OR-low or 1.5xATR, 1R/2R
               targets, and shares sized to risk ₹1,000 (0.5% of ₹2L).
        """
        out: dict = {
            "tool": "day_trade",
            "symbol": symbol.upper().replace(".NS", ""),
            "timeframe": _INTRADAY_INTERVAL,
            "capital": _r(capital),
            "risk_pct": _r(risk_pct, 3),
            "caps": {
                "intraday_lookback": _INTRADAY_PERIOD,
                "note": ("Bounded to a 5d/15m pull (~125 bars) for a 512MB host. "
                         "Intraday data is delayed/free-tier; not for live execution."),
            },
            "warnings": [],
        }

        # --- 1. Concurrent I/O: intraday bars (required) + live quote (best-effort) ---
        df = None
        quote = None
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_hist = ex.submit(_hist, symbol, _INTRADAY_PERIOD, _INTRADAY_INTERVAL)
            f_quote = ex.submit(get_nse_quote, symbol)
            try:
                df = f_hist.result()
            except Exception as e:
                f_quote.cancel()
                out["error"] = f"intraday data unavailable for {symbol}: {e}"
                out["verdict"] = "NO-TRADE — no intraday 15m data (market closed / holiday / bad symbol)."
                return dumps(out)
            try:
                q = f_quote.result()
                if isinstance(q, dict) and not q.get("error"):
                    quote = q
            except Exception:
                quote = None

        if df is None or df.empty or len(df) < (_OPENING_RANGE_BARS + 2):
            out["error"] = f"insufficient intraday bars for {symbol} (got {0 if df is None else len(df)})"
            out["verdict"] = "NO-TRADE — not enough 15m bars yet (too early in session?)."
            return dumps(out)

        # Normalize index -> calendar dates for session splitting.
        try:
            idx = df.index
            day_keys = pd.Index([ts.date() for ts in idx])
            unique_days = list(dict.fromkeys(day_keys))  # ordered unique
            latest_day = unique_days[-1]
            prior_day = unique_days[-2] if len(unique_days) >= 2 else None
            today = df[day_keys == latest_day]
            prior = df[day_keys == prior_day] if prior_day is not None else None
        except Exception as e:
            out["error"] = f"could not split intraday sessions: {e}"
            out["verdict"] = "NO-TRADE — session parsing failed."
            return dumps(out)

        c = df["Close"]
        last = _f(c.iloc[-1])
        ltp = _f(quote.get("price")) if quote else None
        ref_price = ltp if ltp is not None else last  # prefer live LTP for triggers

        out["session"] = {
            "latest_day": str(latest_day),
            "prior_day": str(prior_day) if prior_day is not None else None,
            "bars_today": int(len(today)),
            "last_bar_close": _r(last),
            "live_ltp": _r(ltp),
            "reference_price": _r(ref_price),
        }
        if quote is None:
            out["warnings"].append("live quote unavailable; using last 15m bar close as reference.")

        # --- 2. Session VWAP (cum typical*vol / cum vol over the LATEST day) ---
        try:
            th = today["High"]; tl = today["Low"]; tc = today["Close"]; tv = today["Volume"]
            typ = (th + tl + tc) / 3.0
            cum_pv = (typ * tv).cumsum()
            cum_v = tv.cumsum().replace(0, np.nan)
            vwap_series = cum_pv / cum_v
            vwap = _f(vwap_series.iloc[-1])
            out["vwap"] = {
                "session_vwap": _r(vwap),
                "price_vs_vwap": ("above" if (ref_price is not None and vwap is not None and ref_price > vwap)
                                  else ("below" if (ref_price is not None and vwap is not None) else None)),
                "dist_pct": _r((ref_price / vwap - 1) * 100) if (ref_price and vwap) else None,
                "note": "Session VWAP = cum(typical*vol)/cum(vol) for today only; the intraday fair-value anchor.",
            }
        except Exception as e:
            vwap = None
            out["vwap"] = {"error": f"vwap failed: {e}"}

        # --- 3. Opening range (first 2 bars = first 30 min) + ORB triggers ---
        try:
            or_bars = today.iloc[:_OPENING_RANGE_BARS]
            or_high = _f(or_bars["High"].max())
            or_low = _f(or_bars["Low"].min())
            or_complete = len(today) > _OPENING_RANGE_BARS
            out["opening_range"] = {
                "or_high": _r(or_high),
                "or_low": _r(or_low),
                "or_width": _r((or_high - or_low)) if (or_high is not None and or_low is not None) else None,
                "bars_used": int(min(_OPENING_RANGE_BARS, len(today))),
                "complete": bool(or_complete),
                "orb_breakout_level": _r(or_high),
                "orb_breakdown_level": _r(or_low),
                "note": "ORB = first-30min high/low. Long on a 15m close above OR-high; short below OR-low.",
            }
            if not or_complete:
                out["warnings"].append("opening range still forming (<3 bars today); ORB triggers tentative.")
        except Exception as e:
            or_high = or_low = None
            out["opening_range"] = {"error": f"opening range failed: {e}"}

        # --- 4. Intraday + prior-day support/resistance ---
        try:
            today_high = _f(today["High"].max())
            today_low = _f(today["Low"].min())
            pdh = _f(prior["High"].max()) if prior is not None and not prior.empty else None
            pdl = _f(prior["Low"].min()) if prior is not None and not prior.empty else None
            out["levels"] = {
                "today_high": _r(today_high),
                "today_low": _r(today_low),
                "prior_day_high": _r(pdh),
                "prior_day_low": _r(pdl),
                "note": "Prior-day high/low (PDH/PDL) are key intraday magnets and target/stop references.",
            }
        except Exception as e:
            pdh = pdl = today_high = today_low = None
            out["levels"] = {"error": f"levels failed: {e}"}

        # --- 5. Intraday ATR(14) on 15m ---
        try:
            atr = _atr_15m(df, _ATR_N)
            out["atr"] = {
                "atr14_15m": _r(atr),
                "atr_pct_of_price": _r((atr / ref_price) * 100) if (atr and ref_price) else None,
                "note": "ATR(14) on 15m bars — per-bar volatility used for the volatility stop (1.5xATR).",
            }
        except Exception as e:
            atr = None
            out["atr"] = {"error": f"atr failed: {e}"}

        # --- 6. RVOL of current 15m bar vs average 15m bar ---
        try:
            v = df["Volume"]
            cur_vol = _f(v.iloc[-1])
            avg_bar_vol = _f(v.tail(40).mean())  # ~last 1.5 sessions of 15m bars
            rvol = (cur_vol / avg_bar_vol) if (cur_vol is not None and avg_bar_vol) else None
            out["rvol"] = {
                "current_bar_volume": int(cur_vol) if cur_vol is not None else None,
                "avg_15m_bar_volume": int(avg_bar_vol) if avg_bar_vol is not None else None,
                "rvol": _r(rvol),
                "state": ("hot" if (rvol and rvol > 1.5) else ("active" if (rvol and rvol > 1.0) else "quiet"))
                         if rvol is not None else None,
                "note": "RVOL>1.5 on the breakout bar confirms participation; <1 is a fade-risk breakout.",
            }
        except Exception as e:
            rvol = None
            out["rvol"] = {"error": f"rvol failed: {e}"}

        # --- 7. Bias (above/below VWAP & opening range) ---
        bull_pts = 0
        bear_pts = 0
        reasons = []
        if vwap is not None and ref_price is not None:
            if ref_price > vwap:
                bull_pts += 1; reasons.append("price above session VWAP")
            else:
                bear_pts += 1; reasons.append("price below session VWAP")
        if or_high is not None and or_low is not None and ref_price is not None:
            if ref_price > or_high:
                bull_pts += 1; reasons.append("price above opening range")
            elif ref_price < or_low:
                bear_pts += 1; reasons.append("price below opening range")
            else:
                reasons.append("price inside opening range (no ORB yet)")
        bias = "LONG" if bull_pts > bear_pts else ("SHORT" if bear_pts > bull_pts else "NEUTRAL")
        out["bias"] = {
            "direction": bias,
            "bull_points": bull_pts,
            "bear_points": bear_pts,
            "reasons": reasons,
        }

        # --- 8. Build the setup: entry, stop, targets, sizing ---
        setup = {"direction": bias}
        try:
            entry = stop = None
            if bias == "LONG" and or_high is not None:
                entry = or_high  # ORB long trigger
                # stop: other side of opening range OR 1.5*ATR below entry — tighter wins
                stop_or = or_low
                stop_atr = entry - 1.5 * atr if atr else None
                stop_candidates = [s for s in (stop_or, stop_atr) if s is not None and s < entry]
                stop = max(stop_candidates) if stop_candidates else None  # tighter = closer to entry
                setup["stop_logic"] = "max(opening-range-low, entry - 1.5*ATR) — tighter of the two"
            elif bias == "SHORT" and or_low is not None:
                entry = or_low  # ORB short trigger
                stop_or = or_high
                stop_atr = entry + 1.5 * atr if atr else None
                stop_candidates = [s for s in (stop_or, stop_atr) if s is not None and s > entry]
                stop = min(stop_candidates) if stop_candidates else None  # tighter = closer to entry
                setup["stop_logic"] = "min(opening-range-high, entry + 1.5*ATR) — tighter of the two"
            else:
                setup["note"] = ("No clean ORB trigger (NEUTRAL / inside opening range). "
                                 "Wait for a 15m close beyond the opening range, or stand aside.")

            if entry is not None and stop is not None and entry != stop:
                per_share_risk = abs(entry - stop)
                risk_budget = _f(capital) * (_f(risk_pct) / 100.0) if (capital and risk_pct is not None) else None
                shares = int(math.floor(risk_budget / per_share_risk)) if (risk_budget and per_share_risk) else None

                if bias == "LONG":
                    t1 = entry + per_share_risk          # 1R
                    t2 = entry + 2 * per_share_risk      # 2R
                    level_targets = [x for x in (today_high, pdh) if x is not None and x > entry]
                else:  # SHORT
                    t1 = entry - per_share_risk
                    t2 = entry - 2 * per_share_risk
                    level_targets = [x for x in (today_low, pdl) if x is not None and x < entry]

                notional = shares * entry if shares else None
                setup.update({
                    "entry": _r(entry),
                    "entry_trigger": (f"15m close above OR-high {_r(or_high)}" if bias == "LONG"
                                      else f"15m close below OR-low {_r(or_low)}"),
                    "stop": _r(stop),
                    "per_share_risk": _r(per_share_risk),
                    "risk_budget_inr": _r(risk_budget),
                    "shares": shares,
                    "notional_inr": _r(notional),
                    "target_1R": _r(t1),
                    "target_2R": _r(t2),
                    "level_targets": [_r(x) for x in level_targets] if level_targets else [],
                    "reward_risk_at_2R": 2.0,
                })
                if notional is not None and capital and notional > _f(capital):
                    out["warnings"].append(
                        f"notional ₹{_r(notional)} exceeds capital ₹{_r(capital)} — requires MIS intraday leverage.")
            elif "note" not in setup:
                setup["note"] = "Could not derive entry/stop (missing opening range or ATR)."
        except Exception as e:
            setup["error"] = f"setup build failed: {e}"
        out["setup"] = setup

        # --- 9. Intraday exit note ---
        out["exit_note"] = ("Intraday only: scale out at 1R, trail the rest to 2R / PDH-PDL. "
                            "Hard square-off before market close (~15:20 IST) — do NOT carry MIS overnight. "
                            "Invalidate the bias if price loses session VWAP against your direction.")

        # --- 10. Verdict / summary ---
        if setup.get("entry") is not None and setup.get("shares"):
            verdict = (f"{bias} ORB on {out['symbol']}: enter {setup['entry']} "
                       f"(stop {setup['stop']}, {setup['shares']} sh), targets "
                       f"{setup.get('target_1R')}/{setup.get('target_2R')}. Square off by close.")
        elif bias == "NEUTRAL":
            verdict = (f"NO-TRADE on {out['symbol']}: price is inside the opening range / "
                       f"mixed vs VWAP. Wait for a 15m ORB close.")
        else:
            verdict = (f"{bias} bias on {out['symbol']} but no executable trigger yet — "
                       f"wait for a clean ORB break with RVOL>1.5.")
        out["verdict"] = verdict

        return dumps(out)
