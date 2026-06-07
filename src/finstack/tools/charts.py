"""
FinStack Chart Tools — Claude-first "data for charts".

One configurable tool returns clean, render-agnostic, plot-ready data. The
server fetches + shapes; you (the LLM) render the interactive chart (Chart.js,
Recharts, plotly, or an HTML artifact) using the `render_hint` + series.
"""

import json
from finstack.utils.respond import dumps as _dumps

from finstack.data import chart_engine as ce


def register_charts_tools(mcp):
    """Register the chart-data tool with the MCP server."""

    @mcp.tool()
    def chart_data(symbols: str, chart: str = "price",
                   period: str = "1y", interval: str = "1d",
                   window: int = 21, bins: int = 30, years: int = 6,
                   max_points: int = 150) -> str:
        """Get clean, plot-ready data to render an interactive chart.

        Returns a render-agnostic envelope: {chart, render_hint, title, x_label,
        y_label, labels, series/candles/points/matrix, meta}. Build the chart from
        it (Chart.js / Recharts / plotly / an HTML artifact).

        Args:
            symbols: comma-separated symbols (NSE assumed; use .BO / ^NSEI / AAPL
                     for BSE / index / global). Some charts use only the first.
            chart: which chart's data to return —
                - price               line of close prices (multi-symbol)
                - candlestick         OHLCV candles (first symbol)
                - comparison          multi-symbol performance rebased to 100
                - drawdown            underwater drawdown-from-peak curve(s)
                - returns_histogram   daily-return distribution (first symbol; uses `bins`)
                - rolling_volatility  annualized rolling vol (uses `window`)
                - correlation_heatmap return-correlation matrix (>=2 symbols)
                - efficient_frontier  risk/return cloud + max-Sharpe & min-vol (>=2 symbols)
                - seasonality         avg return by month (first symbol; uses `years`)
                - volume              volume bars + 20d avg (first symbol)
                - volume_profile      volume-by-price + POC & value area (first symbol; uses `bins`)
            period: history window (e.g. "6mo","1y","2y","5y").
            interval: bar size ("1d","1wk","1mo").
            window: rolling window in days (rolling_volatility).
            bins: histogram bins (returns_histogram).
            years: lookback years (seasonality).

        Examples:
            chart_data(symbols="RELIANCE,TCS,INFY", chart="comparison", period="1y")
            chart_data(symbols="HDFCBANK", chart="candlestick", period="6mo")
            chart_data(symbols="RELIANCE,TCS,INFY,ITC", chart="efficient_frontier")
        """
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        if not syms:
            return _dumps({"error": "no symbols provided"}, indent=2)
        c = chart.strip().lower()
        try:
            if c == "price":
                res = ce.price(syms, period, interval)
            elif c == "candlestick":
                res = ce.candlestick(syms[0], period, interval)
            elif c == "comparison":
                res = ce.comparison(syms, period, interval)
            elif c == "drawdown":
                res = ce.drawdown(syms, period)
            elif c == "returns_histogram":
                res = ce.returns_histogram(syms[0], period, bins)
            elif c == "rolling_volatility":
                res = ce.rolling_volatility(syms, period, window)
            elif c == "correlation_heatmap":
                res = ce.correlation_heatmap(syms, period)
            elif c == "efficient_frontier":
                res = ce.efficient_frontier(syms, period)
            elif c == "seasonality":
                res = ce.seasonality(syms[0], years)
            elif c == "volume":
                res = ce.volume(syms[0], period, interval)
            elif c == "volume_profile":
                res = ce.volume_profile(syms[0], period, bins)
            else:
                return _dumps({"error": f"unknown chart '{chart}'",
                                   "valid_charts": ["price", "candlestick", "comparison", "drawdown",
                                                    "returns_histogram", "rolling_volatility",
                                                    "correlation_heatmap", "efficient_frontier",
                                                    "seasonality", "volume", "volume_profile"]}, indent=2)
        except Exception as e:
            return _dumps({"error": f"{type(e).__name__}: {e}", "chart": c}, indent=2)

        # context-engineering: thin long series so a chart stays cheap to read
        from finstack.utils.respond import downsample
        if isinstance(res, dict) and max_points and "error" not in res:
            if "labels" in res:
                res["labels"] = downsample(res["labels"], max_points)
            for s in res.get("series", []):
                s["data"] = downsample(s["data"], max_points)
            for key in ("candles", "points"):
                if key in res:
                    res[key] = downsample(res[key], max_points)
        return _dumps(res, indent=2, default=str)
