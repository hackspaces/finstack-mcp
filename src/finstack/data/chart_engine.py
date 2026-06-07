"""
FinStack Chart Engine

Returns CLEAN, render-agnostic, plot-ready data — the server fetches + shapes,
the LLM renders the interactive chart (Chart.js / Recharts / plotly / HTML
artifact). Every function returns a uniform envelope:

    {
      "chart": <type>,
      "render_hint": "line|candlestick|bar|heatmap|histogram|scatter",
      "title": str, "x_label": str, "y_label": str,
      "labels": [...]                 # x-axis categories (line/bar/heatmap)
      "series": [{"name": str, "data": [...]}],   # one per line/bar group
      ...payload varies by render_hint (points / matrix / candles)...
      "meta": {...}
    }
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yfinance as yf


def _to_nse(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.endswith((".NS", ".BO")) or "." in s or s.startswith("^"):
        return s
    return f"{s}.NS"


def _hist(symbol: str, period: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(_to_nse(symbol)).history(period=period, interval=interval)
    if df is None or df.empty:
        raise ValueError(f"no price data for {symbol}")
    return df


def _close_frame(symbols: list[str], period: str, interval: str) -> pd.DataFrame:
    cols = {}
    for s in symbols:
        try:
            cols[s.strip().upper()] = _hist(s, period, interval)["Close"]
        except Exception:
            continue
    if not cols:
        raise ValueError("no price data for any symbol")
    return pd.DataFrame(cols).dropna(how="all")


def _dates(idx) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def _clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), 4)


def price(symbols: list[str], period: str = "1y", interval: str = "1d") -> dict:
    """Close-price line(s)."""
    df = _close_frame(symbols, period, interval)
    return {
        "chart": "price", "render_hint": "line",
        "title": f"Price — {', '.join(df.columns)}", "x_label": "Date", "y_label": "Price",
        "labels": _dates(df.index),
        "series": [{"name": c, "data": [_clean(x) for x in df[c].tolist()]} for c in df.columns],
        "meta": {"period": period, "interval": interval, "points": len(df)},
    }


def candlestick(symbol: str, period: str = "6mo", interval: str = "1d") -> dict:
    """OHLC(V) candles for one symbol."""
    df = _hist(symbol, period, interval)
    candles = [{"t": d.strftime("%Y-%m-%d"), "o": _clean(r.Open), "h": _clean(r.High),
                "l": _clean(r.Low), "c": _clean(r.Close), "v": int(r.Volume) if not math.isnan(r.Volume) else None}
               for d, r in df.iterrows()]
    return {
        "chart": "candlestick", "render_hint": "candlestick",
        "title": f"{symbol.upper()} OHLC", "x_label": "Date", "y_label": "Price",
        "candles": candles, "meta": {"period": period, "interval": interval, "points": len(candles)},
    }


def comparison(symbols: list[str], period: str = "1y", interval: str = "1d") -> dict:
    """Multi-symbol performance rebased to 100 (relative comparison)."""
    df = _close_frame(symbols, period, interval).dropna()
    if df.empty:
        return {"error": "no overlapping dates across symbols"}
    rebased = df / df.iloc[0] * 100.0
    return {
        "chart": "comparison", "render_hint": "line",
        "title": f"Relative performance (rebased to 100) — {period}",
        "x_label": "Date", "y_label": "Indexed to 100",
        "labels": _dates(rebased.index),
        "series": [{"name": c, "data": [_clean(x) for x in rebased[c].tolist()]} for c in rebased.columns],
        "meta": {"period": period, "final_return_pct": {c: _clean(rebased[c].iloc[-1] - 100) for c in rebased.columns}},
    }


def drawdown(symbols: list[str], period: str = "2y") -> dict:
    """Underwater (drawdown-from-peak) curve(s)."""
    df = _close_frame(symbols, period, "1d").dropna()
    series = []
    maxdd = {}
    for c in df.columns:
        run_max = df[c].cummax()
        dd = (df[c] / run_max - 1.0) * 100.0
        series.append({"name": c, "data": [_clean(x) for x in dd.tolist()]})
        maxdd[c] = _clean(dd.min())
    return {
        "chart": "drawdown", "render_hint": "line",
        "title": f"Drawdown from peak — {period}", "x_label": "Date", "y_label": "Drawdown %",
        "labels": _dates(df.index), "series": series, "meta": {"max_drawdown_pct": maxdd},
    }


def returns_histogram(symbol: str, period: str = "2y", bins: int = 30) -> dict:
    """Histogram of daily % returns for one symbol."""
    df = _hist(symbol, period, "1d")
    rets = df["Close"].pct_change().dropna() * 100.0
    counts, edges = np.histogram(rets.values, bins=int(bins))
    labels = [f"{edges[i]:.2f}" for i in range(len(edges) - 1)]
    return {
        "chart": "returns_histogram", "render_hint": "histogram",
        "title": f"{symbol.upper()} daily return distribution", "x_label": "Daily return %", "y_label": "Frequency",
        "labels": labels, "series": [{"name": "frequency", "data": [int(x) for x in counts]}],
        "meta": {"mean_pct": _clean(rets.mean()), "std_pct": _clean(rets.std()),
                 "skew": _clean(rets.skew()), "kurtosis": _clean(rets.kurtosis()), "n": int(rets.shape[0])},
    }


def rolling_volatility(symbols: list[str], period: str = "2y", window: int = 21) -> dict:
    """Annualized rolling volatility line(s)."""
    df = _close_frame(symbols, period, "1d")
    rets = df.pct_change()
    vol = rets.rolling(int(window)).std() * math.sqrt(252) * 100.0
    vol = vol.dropna(how="all")
    return {
        "chart": "rolling_volatility", "render_hint": "line",
        "title": f"Rolling {window}d annualized volatility", "x_label": "Date", "y_label": "Volatility %",
        "labels": _dates(vol.index),
        "series": [{"name": c, "data": [_clean(x) for x in vol[c].tolist()]} for c in vol.columns],
        "meta": {"window": int(window), "period": period},
    }


def correlation_heatmap(symbols: list[str], period: str = "1y") -> dict:
    """Return-correlation matrix as heatmap data."""
    df = _close_frame(symbols, period, "1d").pct_change().dropna()
    if df.shape[1] < 2:
        return {"error": "need >=2 symbols with overlapping data"}
    corr = df.corr()
    return {
        "chart": "correlation_heatmap", "render_hint": "heatmap",
        "title": f"Return correlation — {period}",
        "labels": list(corr.columns),
        "matrix": [[_clean(corr.iloc[i, j]) for j in range(corr.shape[1])] for i in range(corr.shape[0])],
        "meta": {"avg_pairwise_corr": _clean(corr.values[np.triu_indices_from(corr.values, k=1)].mean())},
    }


def efficient_frontier(symbols: list[str], period: str = "2y", n_portfolios: int = 3000) -> dict:
    """Random-portfolio cloud + max-Sharpe / min-vol markers (scatter)."""
    df = _close_frame(symbols, period, "1d").dropna()
    if df.shape[1] < 2:
        return {"error": "need >=2 symbols"}
    rets = df.pct_change().dropna()
    mean = rets.mean() * 252
    cov = rets.cov() * 252
    n = df.shape[1]
    rf = 0.065
    pts = []
    best_sharpe = {"sharpe": -1e9}
    min_vol = {"vol": 1e9}
    rng = np.random.default_rng(42)
    for _ in range(int(n_portfolios)):
        w = rng.random(n); w /= w.sum()
        r = float(np.dot(w, mean))
        v = float(math.sqrt(np.dot(w, np.dot(cov, w))))
        s = (r - rf) / v if v else 0.0
        pts.append({"vol": _clean(v * 100), "ret": _clean(r * 100), "sharpe": _clean(s)})
        if s > best_sharpe["sharpe"]:
            best_sharpe = {"vol": _clean(v * 100), "ret": _clean(r * 100), "sharpe": _clean(s),
                           "weights": {c: _clean(wi) for c, wi in zip(df.columns, w)}}
        if v < min_vol["vol"]:
            min_vol = {"vol": _clean(v * 100), "ret": _clean(r * 100),
                       "weights": {c: _clean(wi) for c, wi in zip(df.columns, w)}}
    return {
        "chart": "efficient_frontier", "render_hint": "scatter",
        "title": f"Efficient frontier — {', '.join(df.columns)}",
        "x_label": "Annual volatility %", "y_label": "Annual return %",
        "points": pts, "markers": {"max_sharpe": best_sharpe, "min_volatility": min_vol},
        "meta": {"period": period, "n_portfolios": int(n_portfolios), "risk_free_rate": rf},
    }


def volume(symbol: str, period: str = "6mo", interval: str = "1d") -> dict:
    """Volume bars + 20d average overlay, colored by up/down day."""
    df = _hist(symbol, period, interval)
    colors = ["#3CDE66" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#FF5C5C"
              for i in range(len(df))]
    avg = df["Volume"].rolling(20).mean()
    return {
        "chart": "volume", "render_hint": "bar",
        "title": f"{symbol.upper()} volume", "x_label": "Date", "y_label": "Volume",
        "labels": _dates(df.index),
        "series": [{"name": "volume", "data": [int(x) if not math.isnan(x) else None for x in df["Volume"]],
                    "colors": colors},
                   {"name": "avg_20d", "data": [_clean(x) for x in avg], "type": "line"}],
        "meta": {"period": period, "rvol_latest": _clean(df["Volume"].iloc[-1] / avg.iloc[-1]) if avg.iloc[-1] else None},
    }


def volume_profile(symbol: str, period: str = "1y", bins: int = 24) -> dict:
    """Volume-by-price histogram → Point of Control (POC) + 70% value area."""
    df = _hist(symbol, period, "1d")
    typ = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].values
    lo, hi = float(typ.min()), float(typ.max())
    edges = np.linspace(lo, hi, int(bins) + 1)
    idx = np.clip(np.digitize(typ.values, edges) - 1, 0, int(bins) - 1)
    prof = np.zeros(int(bins))
    for i, vv in zip(idx, vol):
        if not math.isnan(vv):
            prof[i] += vv
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(int(bins))]
    poc_i = int(np.argmax(prof))
    # 70% value area around POC
    total = prof.sum(); target = total * 0.70
    lo_i = hi_i = poc_i; acc = prof[poc_i]
    while acc < target and (lo_i > 0 or hi_i < int(bins) - 1):
        down = prof[lo_i - 1] if lo_i > 0 else -1
        up = prof[hi_i + 1] if hi_i < int(bins) - 1 else -1
        if up >= down:
            hi_i += 1; acc += prof[hi_i]
        else:
            lo_i -= 1; acc += prof[lo_i]
    return {
        "chart": "volume_profile", "render_hint": "bar_horizontal",
        "title": f"{symbol.upper()} volume profile ({period})",
        "x_label": "Volume", "y_label": "Price",
        "price_levels": [_clean(x) for x in centers],
        "series": [{"name": "volume_at_price", "data": [int(x) for x in prof]}],
        "meta": {"current_price": _clean(df["Close"].iloc[-1]),
                 "poc_price": _clean(centers[poc_i]),
                 "value_area_high": _clean(centers[hi_i]), "value_area_low": _clean(centers[lo_i]),
                 "note": "POC = price with most traded volume (strongest S/R); value area = 70% of volume."},
    }


def seasonality(symbol: str, years: int = 6) -> dict:
    """Average return by calendar month (bar) over the lookback."""
    df = _hist(symbol, f"{int(years)}y", "1d")
    m = df["Close"].resample("ME").last().pct_change().dropna() * 100.0
    by_month = m.groupby(m.index.month).mean()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    labels = [names[i - 1] for i in by_month.index]
    return {
        "chart": "seasonality", "render_hint": "bar",
        "title": f"{symbol.upper()} avg monthly return ({years}y)", "x_label": "Month", "y_label": "Avg return %",
        "labels": labels, "series": [{"name": "avg_monthly_return_pct", "data": [_clean(x) for x in by_month.tolist()]}],
        "meta": {"years": int(years), "best_month": names[int(by_month.idxmax()) - 1],
                 "worst_month": names[int(by_month.idxmin()) - 1]},
    }
