"""
FinStack Quote Tools — one configurable market-quote tool.

A single `market_quote` umbrella tool that replaces a fistful of narrow
single-purpose quote tools. The `view` arg selects what to do; for plain
price quotes it auto-detects (or you force) the market — NSE / BSE / global /
crypto / FX — and supports a comma-separated batch of symbols.

Each branch reuses the EXACT data-layer function the original per-tool wrapper
called (copied, not reinvented):
  view=quote              -> get_nse_quote / get_bse_quote (finstack.data.nse)
                             get_global_quote / get_crypto_price / get_forex_rate
                             (finstack.data.global_markets)
  view=index              -> get_index_data (finstack.data.nse)  [nifty_index]
  view=compare            -> compare_stocks (finstack.data.analytics)
  view=support_resistance -> compute_support_resistance (finstack.data.analytics)
"""

import json

from finstack.data.nse import get_nse_quote, get_bse_quote, get_index_data
from finstack.data.global_markets import (
    get_global_quote,
    get_crypto_price,
    get_forex_rate,
)
from finstack.data.analytics import compare_stocks, compute_support_resistance

VALID_VIEWS = ["quote", "index", "compare", "support_resistance"]
VALID_MARKETS = ["auto", "nse", "bse", "global", "crypto", "fx"]

# common crypto tickers — used only to help "auto" route a single bare symbol
_CRYPTO_HINTS = {
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "MATIC", "DOT", "BNB",
    "LTC", "AVAX", "LINK", "TRX", "SHIB", "USDT", "USDC",
}


def _detect_market(symbol: str) -> str:
    """Best-effort market routing for `market="auto"` quotes.

    Rules (in order):
      - "FROM/TO" or "FROM-TO" (e.g. USD/INR) -> fx
      - has a dot suffix (AAPL? no; HSBA.L, 7203.T, RELIANCE.NS) -> global
      - a known crypto ticker -> crypto
      - otherwise a plain Indian ticker -> nse
    """
    s = symbol.strip().upper()
    if "/" in s or (s.count("-") == 1 and len(s) <= 8):
        return "fx"
    if "." in s:
        return "global"
    if s in _CRYPTO_HINTS:
        return "crypto"
    return "nse"


def _split_fx_pair(symbol: str) -> tuple[str, str]:
    """Parse 'USD/INR' or 'USD-INR' or 'USDINR' into (from, to)."""
    s = symbol.strip().upper()
    for sep in ("/", "-"):
        if sep in s:
            a, _, b = s.partition(sep)
            return a.strip(), (b.strip() or "INR")
    if len(s) == 6:  # e.g. USDINR
        return s[:3], s[3:]
    return s, "INR"


def _one_quote(symbol: str, market: str) -> dict:
    """Fetch a single quote, routing to the right data-layer function."""
    mkt = market if market in VALID_MARKETS else "auto"
    if mkt == "auto":
        mkt = _detect_market(symbol)

    if mkt == "nse":
        return get_nse_quote(symbol)
    if mkt == "bse":
        return get_bse_quote(symbol)
    if mkt == "global":
        return get_global_quote(symbol)
    if mkt == "crypto":
        return get_crypto_price(symbol)
    if mkt == "fx":
        frm, to = _split_fx_pair(symbol)
        return get_forex_rate(frm, to)
    # should be unreachable, but fail loud per-item
    return {"error": f"unknown market '{market}' for symbol '{symbol}'",
            "valid_markets": VALID_MARKETS}


def register_quote_tools(mcp):
    """Register the configurable market-quote tool with the MCP server."""

    @mcp.tool()
    def market_quote(symbols: str, view: str = "quote", market: str = "auto") -> str:
        """One configurable quote tool — prices, indices, comparison, key levels.

        Replaces several narrow tools (nse_quote, bse_quote, stock_quote,
        crypto_price, forex_rate, nifty_index, compare_stocks, support_resistance)
        with a single `view`-driven entry point.

        Args:
            symbols: what to look up. Meaning depends on `view`:
                - quote:   one OR a comma-separated batch, e.g. "RELIANCE"
                           or "RELIANCE,TCS,INFY". For FX use a pair like
                           "USD/INR" or "USD-EUR".
                - index:   an index name, e.g. "NIFTY50", "SENSEX", "BANKNIFTY",
                           "NIFTYIT", or "ALL".
                - compare: 2-5 comma-separated symbols, e.g. "RELIANCE,TCS,INFY".
                - support_resistance: a single ticker, e.g. "RELIANCE".
            view: one of:
                - quote               live price quote(s) (default)
                - index               index value (Nifty/Sensex/BankNifty/...)
                - compare             side-by-side comparison of 2-5 stocks
                - support_resistance  pivot levels + key support/resistance
            market: only used by view=quote. One of:
                - auto    (default) detect per symbol — plain Indian tickers->NSE,
                          dotted tickers (AAPL? no; HSBA.L/7203.T/RELIANCE.NS)->global,
                          known crypto tickers->crypto, "X/Y" pairs->fx
                - nse     force NSE quote
                - bse     force BSE quote
                - global  force global (US/EU/Asia) quote
                - crypto  force crypto price (USD)
                - fx      force forex rate (symbols is the pair, e.g. "USD/INR")

        Returns:
            JSON string. For batched quotes: {view, market, count, results:
            {symbol: <quote or {"error":...}>}}. For a single symbol/index/etc.
            the underlying result dict. Per-symbol errors are isolated.

        Examples:
            market_quote("RELIANCE")                                  # NSE quote
            market_quote("RELIANCE,TCS,HDFCBANK")                     # batch NSE
            market_quote("AAPL", market="global")                    # US quote
            market_quote("BTC", market="crypto")                     # crypto
            market_quote("USD/INR", market="fx")                     # forex
            market_quote("RELIANCE", market="bse")                   # BSE quote
            market_quote("NIFTY50", view="index")                    # index
            market_quote("RELIANCE,TCS,INFY", view="compare")        # compare
            market_quote("RELIANCE", view="support_resistance")      # key levels
        """
        v = view.strip().lower()
        if v not in VALID_VIEWS:
            return json.dumps({
                "error": f"unknown view '{view}'",
                "valid_views": VALID_VIEWS,
            }, indent=2)

        mkt = market.strip().lower()
        if mkt not in VALID_MARKETS:
            return json.dumps({
                "error": f"unknown market '{market}'",
                "valid_markets": VALID_MARKETS,
            }, indent=2)

        # --- index: nifty_index ---
        if v == "index":
            index_name = symbols.strip() or "NIFTY50"
            result = get_index_data(index_name)
            return json.dumps(result, indent=2, default=str)

        # --- compare: compare_stocks_tool ---
        if v == "compare":
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
            if not sym_list:
                return json.dumps({"error": "No symbols provided for compare."}, indent=2)
            result = compare_stocks(sym_list)
            return json.dumps(result, indent=2, default=str)

        # --- support_resistance ---
        if v == "support_resistance":
            sym = symbols.strip().split(",")[0].strip()
            if not sym:
                return json.dumps({"error": "No symbol provided for support_resistance."}, indent=2)
            result = compute_support_resistance(sym)
            return json.dumps(result, indent=2, default=str)

        # --- quote (possibly batched) ---
        tickers = [s.strip() for s in symbols.split(",") if s.strip()]
        if not tickers:
            return json.dumps({"error": "No symbols provided."}, indent=2)

        # single symbol -> return the raw quote dict (mirrors old single tools)
        if len(tickers) == 1:
            try:
                result = _one_quote(tickers[0], mkt)
            except Exception as e:  # fail loud, but as JSON
                result = {"error": f"{type(e).__name__}: {e}", "symbol": tickers[0]}
            return json.dumps(result, indent=2, default=str)

        # batch -> isolate per-symbol failures
        results: dict[str, object] = {}
        for sym in tickers:
            try:
                results[sym] = _one_quote(sym, mkt)
            except Exception as e:
                results[sym] = {"error": f"{type(e).__name__}: {e}"}

        return json.dumps({
            "view": "quote",
            "market": mkt,
            "count": len(results),
            "results": results,
        }, indent=2, default=str)
