"""
FinStack History Tool — one configurable historical-OHLCV tool over many symbols.

Absorbs three single-symbol historical wrappers into a single tool that loops over
a comma-separated list of tickers and auto-routes each one to the right data layer:

  - nse_historical     -> finstack.data.nse.get_historical_data        (NSE / India)
  - stock_historical   -> finstack.data.global_markets.get_global_historical (global)
  - crypto_historical  -> finstack.data.global_markets.get_crypto_historical (crypto)

`market="auto"` (default) detects per symbol whether it is a crypto, an NSE Indian
stock, or a global ticker. Pass market="nse"|"global"|"crypto" to force a route.
Failures are isolated per symbol so one bad ticker never fails the whole call.
"""

import json
from finstack.utils.respond import dumps as _dumps

from finstack.data.nse import get_historical_data
from finstack.data.global_markets import get_global_historical, get_crypto_historical

# market name -> callable(symbol, period, interval) -> dict
HISTORY_OPS = {
    "nse": get_historical_data,
    "global": get_global_historical,
    "crypto": get_crypto_historical,
}

VALID_MARKETS = ["auto", "nse", "global", "crypto"]

MAX_SYMBOLS = 25

# Common crypto tickers used to auto-detect the crypto route.
_CRYPTO_TICKERS = {
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "MATIC", "DOT", "BNB", "LTC",
    "AVAX", "LINK", "TRX", "BCH", "XLM", "ATOM", "ETC", "FIL", "NEAR", "ALGO",
    "SHIB", "UNI", "APT", "ARB", "OP", "USDT", "USDC",
}


def _detect_market(symbol: str) -> str:
    """Auto-detect which historical route a symbol belongs to.

    crypto  -> ends with -USD, or a well-known crypto ticker
    global  -> contains a dot suffix (AAPL has none; HSBA.L, 7203.T do) or no
               obvious India marker but is clearly foreign — handled via .NS too
    nse     -> default for plain Indian-style symbols
    """
    sym = symbol.strip().upper()

    # explicit crypto pair
    if sym.endswith("-USD") or sym in _CRYPTO_TICKERS:
        return "crypto"

    # explicit yfinance exchange suffix
    if "." in sym:
        if sym.endswith(".NS") or sym.endswith(".BO"):
            return "nse"
        return "global"

    # default: treat plain symbols as NSE (the India-first default)
    return "nse"


def register_history_tools(mcp):
    """Register the configurable historical-data tool with the MCP server."""

    @mcp.tool()
    def history(
        symbols: str,
        period: str = "1y",
        interval: str = "1d",
        market: str = "auto",
    ) -> str:
        """Historical OHLCV for MANY tickers across NSE, global, and crypto in ONE call.

        One configurable tool that replaces nse_historical, stock_historical, and
        crypto_historical. Pass a comma-separated list; each symbol is auto-routed
        to the correct data source (or force it with `market`). Per-symbol failures
        are isolated.

        Args:
            symbols: comma-separated tickers, e.g. "RELIANCE,AAPL,BTC".
                     Up to 25 per call; extras are reported in `dropped`.
            period: time span. Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
            interval: candle size. Options: 1m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo.
                      Note: 1m data is only available for the last ~7 days.
            market: routing mode. One of:
                - auto    detect per symbol (default): crypto (BTC, ETH, *-USD),
                          NSE (plain symbols, .NS/.BO), else global (.L, .T, .HK ...)
                - nse     force NSE / India historical (get_historical_data)
                - global  force global-equity historical (get_global_historical)
                - crypto  force crypto historical (get_crypto_historical)

        Returns:
            JSON string: {period, interval, market, count, results: {symbol: <ohlcv>}}.
            Each symbol's value also carries the `market` route used. A ticker that
            errors gets {"error": "..."} under its key.

        Examples:
            history(symbols="RELIANCE,TCS,INFY", period="1y", interval="1d")
            history(symbols="AAPL,MSFT", period="6mo", interval="1wk", market="global")
            history(symbols="BTC,ETH,SOL", period="3mo", market="crypto")
            history(symbols="RELIANCE,AAPL,BTC")  # mixed, auto-routed
        """
        mode = market.strip().lower()
        if mode not in VALID_MARKETS:
            return _dumps({
                "error": f"Unknown market '{market}'.",
                "valid_markets": VALID_MARKETS,
            }, indent=2)

        raw = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        # de-duplicate while preserving order
        seen: set[str] = set()
        tickers = [s for s in raw if not (s in seen or seen.add(s))]
        dropped = tickers[MAX_SYMBOLS:]
        tickers = tickers[:MAX_SYMBOLS]

        if not tickers:
            return _dumps({"error": "No symbols provided."}, indent=2)

        results: dict[str, object] = {}
        for sym in tickers:
            route = _detect_market(sym) if mode == "auto" else mode
            fn = HISTORY_OPS[route]
            try:
                res = fn(sym, period, interval)
                if isinstance(res, dict):
                    res = {"market": route, **res}
                results[sym] = res
            except Exception as e:  # isolate per-symbol failures
                results[sym] = {"market": route, "error": f"{type(e).__name__}: {e}"}

        out = {
            "period": period,
            "interval": interval,
            "market": mode,
            "count": len(results),
            "results": results,
        }
        if dropped:
            out["dropped"] = dropped
            out["note"] = f"Capped at {MAX_SYMBOLS} symbols per call; {len(dropped)} ignored."
        return _dumps(out, indent=2, default=str)
