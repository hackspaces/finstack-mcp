"""
FinStack `technicals` — full technical-analysis read for an NSE stock:
indicators + interpreted signals + support/resistance/pivots + a composite
bull/bear bias. One configurable tool.
"""

from finstack.utils.respond import dumps
from finstack.data import tech_engine as te


def register_technicals_tools(mcp):
    """Register the technicals tool."""

    @mcp.tool()
    def technicals(symbol: str, views: str = "summary",
                   period: str = "6mo", interval: str = "1d") -> str:
        """Technical analysis for an NSE stock — indicators, signals, levels, bias.

        Args:
            symbol: NSE symbol (e.g. RELIANCE; ".NS" appended).
            views: comma list of what to return —
                - summary    composite bias (BULLISH/BEARISH/NEUTRAL) + trend + RSI state (default)
                - indicators current RSI/MACD/SMA20-50-200/EMA/Bollinger/ATR/Stochastic/ADX/+DI/-DI
                - signals    interpreted: rsi ob/os, macd cross, MA stack, golden/death cross,
                             bollinger position + squeeze, stochastic, trend strength/direction
                - levels     support/resistance (20d/50d) + classic pivot points (P/R1/R2/S1/S2)
                - trend      ADX + directional movement detail
                - all        everything
            period: history window (default "6mo"; use "1y"+ for SMA200).
            interval: bar size ("1d","1wk").

        Examples:
            technicals(symbol="RELIANCE")                       # composite read
            technicals(symbol="HDFCBANK", views="indicators,signals")
            technicals(symbol="TCS", views="all", period="1y")
        """
        try:
            res = te.technicals(symbol, views=views, period=period, interval=interval)
        except Exception as e:
            return dumps({"error": f"{type(e).__name__}: {e}", "symbol": symbol})
        return dumps(res)
