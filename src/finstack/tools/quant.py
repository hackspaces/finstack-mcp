"""
FinStack Quant Analytics Tools — consolidated, configurable.

ONE configurable tool (`quant`) replaces the five former narrow quant tools.
Each `analysis` branch reuses the exact same pure-computation function the old
wrappers called, from finstack.data.quant_engine (and the SMA-crossover
backtest from finstack.data.analytics). Each branch returns
json.dumps(result, indent=2, default=str).

analysis values:
  - risk         -> compute_risk_metrics(symbol)        (per-symbol; runs each)
  - optimize     -> optimize_portfolio(symbol_list)
  - vol_forecast -> forecast_volatility(symbol)         (per-symbol; runs each)
  - correlation  -> correlation_matrix(symbol_list)
  - pairs        -> pairs_cointegration(symbol1, symbol2)
  - backtest     -> backtest_sma_crossover(symbol)      (per-symbol; runs each)
"""

import json

from finstack.data.quant_engine import (
    compute_risk_metrics,
    optimize_portfolio,
    forecast_volatility,
    correlation_matrix,
    pairs_cointegration,
)
from finstack.data.analytics import backtest_sma_crossover

VALID_ANALYSES = [
    "risk",
    "optimize",
    "vol_forecast",
    "correlation",
    "pairs",
    "backtest",
]


def register_quant_tools(mcp):
    """Register the consolidated quant analytics tool with the MCP server."""

    @mcp.tool()
    def quant(
        symbols: str,
        analysis: str = "risk",
        benchmark: str = "^NSEI",
        period: str = "",
        objective: str = "max_sharpe",
        horizon: int = 5,
        symbol1: str = "",
        symbol2: str = "",
        short_window: int = 20,
        long_window: int = 50,
        initial_capital: float = 100000,
    ) -> str:
        """Configurable quantitative analytics on NSE equities (one tool, many modes).

        Built on numpy/pandas/scipy/statsmodels/arch. Pass `symbols` (comma-separated)
        and pick an `analysis`. Per-symbol analyses (risk, vol_forecast, backtest)
        run over every symbol with failures isolated per ticker. Basket analyses
        (optimize, correlation) take the full list. `pairs` uses symbol1/symbol2
        (falling back to the first two of `symbols`).

        Args:
            symbols: comma-separated NSE symbols, e.g. "RELIANCE,TCS,HDFCBANK"
                     (".NS" is appended automatically).
            analysis: which analysis to run. One of:
                - risk         risk profile (annualized return/vol, Sharpe, Sortino,
                               max drawdown, VaR/CVaR, beta & alpha vs benchmark)
                - optimize     long-only mean-variance portfolio optimization
                - vol_forecast GARCH(1,1) volatility forecast
                - correlation  return-correlation matrix + diversification note
                - pairs        cointegration test + pairs-trading signal
                - backtest     SMA crossover backtest vs buy-and-hold
            benchmark: benchmark ticker for `risk` (default Nifty 50 "^NSEI").
            period: history window (e.g. "1y", "2y"). Empty -> sensible per-analysis
                    default ("1y" for risk/optimize/correlation, "2y" for
                    vol_forecast/pairs/backtest).
            objective: for `optimize` — "max_sharpe" or "min_vol".
            horizon: for `vol_forecast` — forecast horizon in trading days (default 5).
            symbol1: for `pairs` — dependent leg (else symbols[0]).
            symbol2: for `pairs` — hedge leg (else symbols[1]).
            short_window: for `backtest` — short SMA window (default 20).
            long_window: for `backtest` — long SMA window (default 50).
            initial_capital: for `backtest` — starting capital (default 100000).

        Returns:
            JSON string. Unknown `analysis` -> {"error": ..., "valid_analyses": [...]}.
            Per-symbol analyses return {analysis, count, results: {symbol: <result>}};
            a ticker that errors gets {"error": "..."} under its key.

        Examples:
            quant(symbols="RELIANCE", analysis="risk", benchmark="^NSEI", period="1y")
            quant(symbols="RELIANCE,TCS,HDFCBANK", analysis="optimize", objective="max_sharpe")
            quant(symbols="HDFCBANK", analysis="vol_forecast", horizon=5)
            quant(symbols="RELIANCE,TCS,HDFCBANK", analysis="correlation", period="1y")
            quant(symbols="HDFCBANK,ICICIBANK", analysis="pairs", period="2y")
            quant(symbols="RELIANCE", analysis="backtest", short_window=20, long_window=50)
        """
        op = analysis.strip().lower()
        if op not in VALID_ANALYSES:
            return json.dumps({
                "error": f"Unknown analysis '{analysis}'.",
                "valid_analyses": VALID_ANALYSES,
            }, indent=2)

        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

        # Per-analysis default period (empty string -> mode default).
        def _period(default: str) -> str:
            return period.strip() if period.strip() else default

        # --- Basket analyses (take the whole list) ---
        if op == "optimize":
            result = optimize_portfolio(
                symbol_list, objective=objective, period=_period("1y")
            )
            return json.dumps(result, indent=2, default=str)

        if op == "correlation":
            result = correlation_matrix(symbol_list, period=_period("1y"))
            return json.dumps(result, indent=2, default=str)

        # --- Pair analysis ---
        if op == "pairs":
            s1 = symbol1.strip() or (symbol_list[0] if len(symbol_list) >= 1 else "")
            s2 = symbol2.strip() or (symbol_list[1] if len(symbol_list) >= 2 else "")
            if not s1 or not s2:
                return json.dumps({
                    "error": "pairs needs two symbols (symbol1/symbol2, or two in `symbols`).",
                }, indent=2)
            result = pairs_cointegration(s1, s2, period=_period("2y"))
            return json.dumps(result, indent=2, default=str)

        # --- Per-symbol analyses (run for each symbol, isolate failures) ---
        if not symbol_list:
            return json.dumps({"error": "No symbols provided."}, indent=2)

        if op == "risk":
            def _one(sym):
                return compute_risk_metrics(
                    sym, benchmark=benchmark, period=_period("1y")
                )
        elif op == "vol_forecast":
            def _one(sym):
                return forecast_volatility(
                    sym, horizon=horizon, period=_period("2y")
                )
        elif op == "backtest":
            def _one(sym):
                return backtest_sma_crossover(
                    sym,
                    short_window=short_window,
                    long_window=long_window,
                    period=_period("2y"),
                    initial_capital=initial_capital,
                )
        else:  # defensive — should be unreachable given the guard above
            return json.dumps({
                "error": f"Unhandled analysis '{analysis}'.",
                "valid_analyses": VALID_ANALYSES,
            }, indent=2)

        results: dict[str, object] = {}
        for sym in symbol_list:
            try:
                results[sym] = _one(sym)
            except Exception as e:  # isolate per-symbol failures
                results[sym] = {"error": f"{type(e).__name__}: {e}"}

        out = {
            "analysis": op,
            "count": len(results),
            "results": results,
        }
        return json.dumps(out, indent=2, default=str)
