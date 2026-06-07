"""
FinStack Quant Analytics Tools

Quant toolset built on real Python quant libraries (numpy, pandas, scipy,
statsmodels, arch). Each tool wraps a pure computation function from
finstack.data.quant_engine and returns a JSON string.

Tools:
  - quant_risk_metrics        - risk profile (Sharpe/Sortino/VaR/beta) vs a benchmark
  - quant_optimize_portfolio  - long-only mean-variance optimization
  - quant_volatility_forecast - GARCH(1,1) volatility forecast
  - quant_correlation_matrix  - correlation matrix + diversification note
  - quant_pairs_trade         - cointegration test + pairs-trading signal
"""

import json

from finstack.data.quant_engine import (
    compute_risk_metrics,
    optimize_portfolio,
    forecast_volatility,
    correlation_matrix,
    pairs_cointegration,
)


def register_quant_tools(mcp):
    """Register quant analytics tools with the MCP server."""

    @mcp.tool()
    def quant_risk_metrics(
        symbol: str,
        benchmark: str = "^NSEI",
        period: str = "1y",
    ) -> str:
        """Compute a quantitative risk profile for an NSE stock vs a benchmark.

        Returns annualized return & volatility, Sharpe and Sortino ratios, max
        drawdown, historical 95% VaR & CVaR, and beta & alpha vs the benchmark
        (OLS regression of daily returns).

        Args:
            symbol: NSE symbol (e.g. RELIANCE, TCS, HDFCBANK); ".NS" is appended.
            benchmark: benchmark ticker, default Nifty 50 ("^NSEI").
            period: history window (e.g. "1y", "2y").

        Returns:
            JSON string of metrics, or an object with an "error" key on failure.

        Example:
            quant_risk_metrics(symbol="RELIANCE", benchmark="^NSEI", period="1y")
        """
        result = compute_risk_metrics(symbol, benchmark=benchmark, period=period)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quant_optimize_portfolio(
        symbols: str,
        objective: str = "max_sharpe",
        period: str = "1y",
    ) -> str:
        """Optimize a long-only equity portfolio (mean-variance, scipy).

        Computes optimal weights (summing to 1, each 0..1) for the given objective
        and reports expected annual return, annual volatility and Sharpe ratio.

        Args:
            symbols: comma-separated NSE symbols (e.g. "RELIANCE,TCS,HDFCBANK").
            objective: "max_sharpe" or "min_vol".
            period: history window (e.g. "1y", "2y").

        Returns:
            JSON string with weights and portfolio stats, or an "error" object.

        Example:
            quant_optimize_portfolio(symbols="RELIANCE,TCS,HDFCBANK",
                                     objective="max_sharpe", period="1y")
        """
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        result = optimize_portfolio(symbol_list, objective=objective, period=period)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quant_volatility_forecast(symbol: str, horizon: int = 5) -> str:
        """Forecast volatility for an NSE stock using a GARCH(1,1) model (arch).

        Reports the current annualized volatility and the annualized volatility
        forecast `horizon` trading days ahead.

        Args:
            symbol: NSE symbol (e.g. RELIANCE, TCS, HDFCBANK).
            horizon: forecast horizon in trading days (default 5).

        Returns:
            JSON string with current and forecast volatility, or an "error" object.

        Example:
            quant_volatility_forecast(symbol="HDFCBANK", horizon=5)
        """
        result = forecast_volatility(symbol, horizon=horizon)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quant_correlation_matrix(symbols: str, period: str = "1y") -> str:
        """Compute the return-correlation matrix for a basket of NSE stocks.

        Returns the full correlation matrix, the average pairwise correlation and
        a plain-language diversification note.

        Args:
            symbols: comma-separated NSE symbols (e.g. "RELIANCE,TCS,HDFCBANK").
            period: history window (e.g. "1y", "2y").

        Returns:
            JSON string with the matrix and diversification note, or an "error".

        Example:
            quant_correlation_matrix(symbols="RELIANCE,TCS,HDFCBANK", period="1y")
        """
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        result = correlation_matrix(symbol_list, period=period)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quant_pairs_trade(symbol1: str, symbol2: str, period: str = "2y") -> str:
        """Test two NSE stocks for cointegration and emit a pairs-trading signal.

        Runs an Engle-Granger cointegration test (statsmodels), estimates the OLS
        hedge ratio, computes the current spread z-score and returns a signal
        (LONG_SPREAD / SHORT_SPREAD / NEUTRAL) plus a "cointegrated" boolean
        (p < 0.05).

        Args:
            symbol1: first NSE symbol, the dependent leg (e.g. HDFCBANK).
            symbol2: second NSE symbol, the hedge leg (e.g. ICICIBANK).
            period: history window (e.g. "2y").

        Returns:
            JSON string with p-value, hedge ratio, z-score and signal, or an
            "error" object on failure.

        Example:
            quant_pairs_trade(symbol1="HDFCBANK", symbol2="ICICIBANK", period="2y")
        """
        result = pairs_cointegration(symbol1, symbol2, period=period)
        return json.dumps(result, indent=2, default=str)
