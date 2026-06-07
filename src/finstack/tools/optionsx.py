"""
FinStack Options-X Tool — one configurable options tool over many views.

A single `options(symbol, views)` tool that fans a symbol out across several
options-analytics views in ONE call, instead of calling each narrow options
tool separately. Each view maps to an existing per-symbol data-layer function
(the exact same call the standalone tool makes); per-view failures are isolated
so one bad view never fails the whole request.

Views and the underlying calls they reuse:
  - chain    -> get_options_chain(symbol)            [from nse_options_chain]
  - oi       -> get_options_oi_analytics(symbol)     [from options_oi_analytics]
  - greeks   -> get_options_greeks(symbol)           [from options_greeks]
  - pcr      -> get_nifty_pcr_trend()                [from nifty_pcr_trend; NIFTY-only]
  - maxpain  -> derived from get_options_oi_analytics(symbol) (max_pain per expiry)
"""

import json
from finstack.utils.respond import dumps as _dumps

from finstack.data.nse_advanced import get_options_chain
from finstack.data.market_intelligence import (
    get_options_oi_analytics,
    get_options_greeks,
    get_nifty_pcr_trend,
)


def _view_chain(symbol: str) -> dict:
    # Mirrors nse_options_chain tool (tools/analytics.py).
    return get_options_chain(symbol)


def _view_oi(symbol: str) -> dict:
    # Mirrors options_oi_analytics tool (tools/market_intelligence.py).
    return get_options_oi_analytics(symbol)


def _view_greeks(symbol: str) -> dict:
    # Mirrors options_greeks tool (tools/market_intelligence.py).
    return get_options_greeks(symbol)


def _view_pcr(symbol: str) -> dict:
    # Mirrors nifty_pcr_trend tool (tools/market_intelligence.py).
    # That data layer is NIFTY-only (^NSEI) and ignores `symbol`.
    return get_nifty_pcr_trend()


def _view_maxpain(symbol: str) -> dict:
    # Max Pain is exposed by get_options_oi_analytics: each entry in `analysis`
    # carries `max_pain` and `max_pain_vs_spot`. We project just those out.
    oi = get_options_oi_analytics(symbol)
    if isinstance(oi, dict) and oi.get("error"):
        return oi
    analysis = oi.get("analysis", []) if isinstance(oi, dict) else []
    per_expiry = [
        {
            "expiry": a.get("expiry"),
            "max_pain": a.get("max_pain"),
            "max_pain_vs_spot": a.get("max_pain_vs_spot"),
        }
        for a in analysis
    ]
    return {
        "symbol": oi.get("symbol") if isinstance(oi, dict) else symbol,
        "underlying_price": oi.get("underlying_price") if isinstance(oi, dict) else None,
        "max_pain_by_expiry": per_expiry,
        "interpretation": "Price gravitates toward max pain at expiry (option writer hedging effect)",
        "data_source": oi.get("data_source") if isinstance(oi, dict) else None,
    }


# view name -> callable(symbol) -> dict
OPTIONS_VIEWS = {
    "chain": _view_chain,
    "oi": _view_oi,
    "greeks": _view_greeks,
    "pcr": _view_pcr,
    "maxpain": _view_maxpain,
}


def register_optionsx_tools(mcp):
    """Register the configurable options-X tool with the MCP server."""

    @mcp.tool()
    def options(symbol: str, views: str = "chain") -> str:
        """Options analytics for a symbol across one or more views in a single call.

        Pass a comma-separated list of views and get each one back under its own
        key. Per-view failures are isolated — one failing view never breaks the rest.

        Args:
            symbol: NSE stock or index symbol (e.g., NIFTY, BANKNIFTY, RELIANCE, TCS).
            views: comma-separated list of views (default "chain"). One or more of:
                - chain    full options chain + PCR (calls/puts, OI, IV)
                - oi       OI analytics: Max Pain, PCR trend, IV summary, top OI strikes
                - greeks   Black-Scholes Delta/Gamma/Theta/Vega/Rho for the chain
                - pcr      Nifty PCR trend across expiries (NIFTY index only — ignores symbol)
                - maxpain  Max Pain per expiry (derived from the OI analytics view)

        Returns:
            JSON string: {symbol, views, results: {view: <result>}, ...}.
            A view that errors gets {"error": "..."} under its key.

        Examples:
            options(symbol="NIFTY", views="oi,maxpain,greeks")
            options(symbol="RELIANCE", views="chain")
            options(symbol="NIFTY", views="pcr")
        """
        requested = [v.strip().lower() for v in views.split(",") if v.strip()]
        if not requested:
            requested = ["chain"]

        unknown = [v for v in requested if v not in OPTIONS_VIEWS]
        if unknown:
            return _dumps({
                "error": f"Unknown view(s): {', '.join(unknown)}.",
                "valid_views": sorted(OPTIONS_VIEWS.keys()),
            }, indent=2)

        # de-duplicate while preserving order
        seen: set[str] = set()
        ordered_views = [v for v in requested if not (v in seen or seen.add(v))]

        results: dict[str, object] = {}
        for view in ordered_views:
            fn = OPTIONS_VIEWS[view]
            try:
                results[view] = fn(symbol)
            except Exception as e:  # isolate per-view failures
                results[view] = {"error": f"{type(e).__name__}: {e}"}

        out = {
            "symbol": symbol,
            "views": ordered_views,
            "results": results,
        }
        if "pcr" in ordered_views:
            out["note"] = "The 'pcr' view is NIFTY-index only and ignores `symbol`."
        return _dumps(out, indent=2, default=str)
