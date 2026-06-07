"""
FinStack Technicals Engine

Full technical-analysis read from yfinance OHLCV: indicators (current values),
interpreted signals (overbought/cross/squeeze/trend), support/resistance + pivots,
and a composite bull/bear bias. Pure pandas/numpy — no TA-Lib dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finstack.data.chart_engine import _hist  # OHLCV fetch (appends .NS)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, n: int = 14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return atr, plus_di, minus_di, dx.ewm(alpha=1 / n, adjust=False).mean()


def _r(x, d=2):
    try:
        return round(float(x), d)
    except Exception:
        return None


def _volume_metrics(df: pd.DataFrame) -> dict:
    """Volume/flow analytics: RVOL, CMF, MFI, A/D, up-down ratio, divergence, spikes."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    typ = (h + l + c) / 3
    avg20 = v.rolling(20).mean().iloc[-1]
    avg50 = v.rolling(50).mean().iloc[-1]
    cur = float(v.iloc[-1])
    rvol = cur / avg20 if avg20 else None

    # Chaikin Money Flow (20)
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    mfv = (mfm * v).fillna(0.0)
    cmf = mfv.rolling(20).sum().iloc[-1] / v.rolling(20).sum().iloc[-1]
    adl = mfv.cumsum()

    # Money Flow Index (14)
    rmf = typ * v
    pos = rmf.where(typ > typ.shift(), 0.0).rolling(14).sum()
    neg = rmf.where(typ < typ.shift(), 0.0).rolling(14).sum()
    mfi = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).iloc[-1]

    # up/down volume ratio (20d) + VWAP(20)
    up_v = v.where(c > c.shift(), 0.0).tail(20).sum()
    dn_v = v.where(c < c.shift(), 0.0).tail(20).sum()
    udr = float(up_v / dn_v) if dn_v else None
    vwap20 = (typ * v).rolling(20).sum().iloc[-1] / v.rolling(20).sum().iloc[-1]

    # price/volume divergence (20d)
    price_chg = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0.0
    vol_trend_up = bool(avg20 > avg50) if (avg20 and avg50) else None
    diverg = "none"
    if price_chg > 0.02 and vol_trend_up is False:
        diverg = "bearish (price up on fading volume)"
    elif price_chg < -0.02 and vol_trend_up is False:
        diverg = "bullish (selloff on fading volume)"
    elif price_chg > 0.02 and vol_trend_up:
        diverg = "confirmed_uptrend (price up on rising volume)"

    # recent volume spikes (last 20 days, vol > 2x avg20)
    spikes = []
    a20 = v.rolling(20).mean()
    for d, vol in v.tail(20).items():
        a = a20.loc[d]
        if a and vol > 2 * a:
            move = float(c.loc[d] / c.shift().loc[d] - 1) * 100 if d in c.shift().index else None
            spikes.append({"date": d.strftime("%Y-%m-%d"), "rvol": _r(vol / a),
                           "day_move_pct": _r(move)})

    return {
        "current_volume": int(cur), "avg_volume_20d": int(avg20) if avg20 else None,
        "avg_volume_50d": int(avg50) if avg50 else None, "rvol": _r(rvol),
        "cmf_20": _r(cmf, 3), "mfi_14": _r(mfi), "vwap_20": _r(vwap20),
        "up_down_vol_ratio_20d": _r(udr),
        "obv_trend": "up" if adl.iloc[-1] > adl.iloc[-20] else "down",
        "volume_trend": ("rising" if vol_trend_up else "falling") if vol_trend_up is not None else None,
        "price_volume_divergence": diverg,
        "recent_spikes": spikes[-5:],
        "note": "RVOL>1.5 = unusually active; CMF>0 accumulation; MFI>80 ob / <20 os. "
                "(Total traded volume — NSE delivery% needs broker data, unavailable here.)",
    }


def technicals(symbol: str, views: str = "summary", period: str = "6mo", interval: str = "1d") -> dict:
    """Compute technicals; `views` is a comma list of indicators/signals/levels/trend/summary."""
    df = _hist(symbol, period, interval)
    if len(df) < 60:
        return {"error": f"insufficient history for {symbol}"}
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(c.iloc[-1])

    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    rsi = _rsi(c)
    mid = sma20
    std = c.rolling(20).std()
    bb_up, bb_dn = mid + 2 * std, mid - 2 * std
    bb_bw = ((bb_up - bb_dn) / mid)
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    atr, plus_di, minus_di, adx = _adx(df)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()

    def cur(s):
        return _r(s.iloc[-1]) if len(s.dropna()) else None

    indicators = {
        "price": _r(last), "sma20": cur(sma20), "sma50": cur(sma50), "sma200": cur(sma200),
        "ema12": cur(ema12), "ema26": cur(ema26),
        "rsi14": cur(rsi), "macd": cur(macd), "macd_signal": cur(macd_sig),
        "macd_hist": _r(macd.iloc[-1] - macd_sig.iloc[-1]),
        "bb_upper": cur(bb_up), "bb_lower": cur(bb_dn), "bb_bandwidth": _r(bb_bw.iloc[-1], 4),
        "atr14": cur(atr), "stoch_k": cur(stoch_k), "stoch_d": cur(stoch_d),
        "adx14": cur(adx), "plus_di": cur(plus_di), "minus_di": cur(minus_di),
    }

    # signals
    rsi_v = indicators["rsi14"]
    adx_v = indicators["adx14"]
    sig = {
        "rsi": "overbought" if rsi_v and rsi_v > 70 else ("oversold" if rsi_v and rsi_v < 30 else "neutral"),
        "macd": "bullish" if macd.iloc[-1] > macd_sig.iloc[-1] else "bearish",
        "ma_stack": "bullish" if (sma50.iloc[-1] > sma200.iloc[-1] and last > sma50.iloc[-1]) else
                    ("bearish" if (sma50.iloc[-1] < sma200.iloc[-1] and last < sma50.iloc[-1]) else "mixed"),
        "golden_cross_50_200": bool(sma50.iloc[-1] > sma200.iloc[-1]),
        "bollinger": "above_upper" if last > bb_up.iloc[-1] else ("below_lower" if last < bb_dn.iloc[-1] else "inside"),
        "bb_squeeze": bool(bb_bw.iloc[-1] <= bb_bw.tail(120).quantile(0.2)),
        "stochastic": "overbought" if (indicators["stoch_k"] or 0) > 80 else ("oversold" if (indicators["stoch_k"] or 100) < 20 else "neutral"),
        "trend_strength": "strong" if adx_v and adx_v > 25 else ("weak" if adx_v and adx_v < 20 else "moderate"),
        "trend_direction": "up" if plus_di.iloc[-1] > minus_di.iloc[-1] else "down",
    }

    # support/resistance + classic pivots (from last bar)
    H, L, C = float(h.iloc[-1]), float(l.iloc[-1]), last
    p = (H + L + C) / 3
    levels = {
        "recent_support_20d": _r(l.tail(20).min()), "recent_resistance_20d": _r(h.tail(20).max()),
        "recent_support_50d": _r(l.tail(50).min()), "recent_resistance_50d": _r(h.tail(50).max()),
        "pivot": _r(p), "r1": _r(2 * p - L), "s1": _r(2 * p - H),
        "r2": _r(p + (H - L)), "s2": _r(p - (H - L)),
    }

    # composite bull/bear score
    bull = sum([sig["macd"] == "bullish", sig["ma_stack"] == "bullish", sig["golden_cross_50_200"],
                sig["trend_direction"] == "up", rsi_v is not None and 50 < rsi_v < 70,
                last > (indicators["sma200"] or last)])
    bear = sum([sig["macd"] == "bearish", sig["ma_stack"] == "bearish", not sig["golden_cross_50_200"],
                sig["trend_direction"] == "down", rsi_v is not None and 30 < rsi_v < 50,
                last < (indicators["sma200"] or last)])
    bias = "BULLISH" if bull - bear >= 2 else ("BEARISH" if bear - bull >= 2 else "NEUTRAL")
    summary = {"bias": bias, "bull_signals": bull, "bear_signals": bear,
               "trend": f"{sig['trend_strength']} {sig['trend_direction']}",
               "rsi_state": sig["rsi"], "vs_sma200": "above" if last > (indicators["sma200"] or last) else "below"}

    want = [w.strip() for w in views.split(",") if w.strip()] or ["summary"]
    out = {"symbol": symbol.upper(), "period": period, "interval": interval, "obv_trend": "up" if obv.iloc[-1] > obv.iloc[-20] else "down"}
    if "summary" in want or "all" in want:
        out["summary"] = summary
    if "indicators" in want or "all" in want:
        out["indicators"] = indicators
    if "signals" in want or "all" in want:
        out["signals"] = sig
    if "levels" in want or "all" in want:
        out["levels"] = levels
    if "trend" in want:
        out["trend"] = {"adx": adx_v, "plus_di": indicators["plus_di"], "minus_di": indicators["minus_di"],
                        "strength": sig["trend_strength"], "direction": sig["trend_direction"]}
    if "volume" in want or "all" in want:
        out["volume"] = _volume_metrics(df)
    return out
