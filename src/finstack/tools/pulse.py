"""
FinStack Pulse Tools — configurable market-pulse, screening, and fund tools.

Three umbrella tools that absorb many narrow ones (vs calling each separately):

  - market_pulse(views) : one call returns several whole-market reads at once
                          (status, movers, circuit, 52-week, FII/DII, deals,
                          corporate actions, sectors, calendars, VIX, GIFT
                          Nifty, quarterly results, PCR). Each view maps to its
                          existing data-layer function; per-view failures are
                          isolated so one broken feed never sinks the call.
  - screen(...)         : stock_screener — multi-criteria fundamental screen.
  - funds(action, ...)  : mutual-fund actions — latest NAV, two-fund overlap,
                          and AMFI industry flows.

Each branch reuses the SAME data-layer call its single-purpose wrapper uses
(indian.py, analytics.py, market_intelligence.py), so behaviour matches the
originals exactly. Unknown action/view -> a loud error JSON listing the valid
choices. Every tool returns json.dumps(result, indent=2, default=str).
"""

import json

from finstack.config import config
from finstack.utils.helpers import tier_locked_error

# market-pulse data-layer functions (same imports the narrow wrappers use)
from finstack.data.nse import get_market_status, get_market_movers
from finstack.data.nse_advanced import (
    get_circuit_breakers,
    get_52week_scanner,
    get_fii_dii_data,
    get_bulk_deals,
    get_corporate_actions,
    get_quarterly_results,
    get_earnings_calendar,
    get_ipo_calendar,
)
from finstack.data.analytics import get_sector_performance, screen_stocks
from finstack.data.market_intelligence import (
    get_india_vix,
    get_gift_nifty,
    get_nifty_pcr_trend,
    get_amfi_fund_flows,
)
from finstack.data.mf_overlap import get_mf_overlap as _get_mf_overlap


# view name -> zero-arg callable() -> dict (matches each single-purpose wrapper)
PULSE_VIEWS = {
    "status": get_market_status,            # nse_market_status
    "movers": get_market_movers,            # nse_top_movers (default "gainers")
    "circuit": get_circuit_breakers,        # nse_circuit_breakers (default "both")
    "fiftytwo_week": get_52week_scanner,    # nse_52week_scanner (default near_high/5)
    "fii_dii": get_fii_dii_data,            # nse_fii_dii_data
    "bulk_deals": get_bulk_deals,           # nse_bulk_deals
    "corporate_actions": get_corporate_actions,  # nse_corporate_actions (needs symbol)
    "sector": get_sector_performance,       # sector_performance
    "earnings_calendar": get_earnings_calendar,  # earnings_calendar (optional symbol)
    "ipo": get_ipo_calendar,                # ipo_calendar
    "vix": get_india_vix,                   # india_vix (default 30 days)
    "gift_nifty": get_gift_nifty,           # gift_nifty
    "quarterly_results": get_quarterly_results,  # nse_quarterly_results (needs symbol)
    "pcr": get_nifty_pcr_trend,             # nifty_pcr_trend (default 5 expiries)
}

# views that REQUIRE a `symbol` argument to mean anything
_SYMBOL_REQUIRED_VIEWS = {"corporate_actions", "quarterly_results"}


def register_pulse_tools(mcp):
    """Register the pulse (configurable market) tools with the MCP server."""

    @mcp.tool()
    def market_pulse(views: str = "status", symbol: str = "") -> str:
        """Whole-market pulse — fetch SEVERAL market-wide reads in ONE call.

        Pass a comma-separated list of views instead of calling a dozen narrow
        tools. Each view is computed independently and a failing feed is isolated
        under its own key, so one broken source never fails the whole call.

        Args:
            views: comma-separated view names (default "status"). One or more of:
                - status             market open/closed + IST time
                - movers             top gainers/losers/active (Nifty 50)
                - circuit            stocks locked at upper/lower circuit
                - fiftytwo_week      stocks near 52-week high/low
                - fii_dii            FII/DII institutional flows today
                - bulk_deals         recent bulk/block deals
                - corporate_actions  dividends/splits/bonuses (needs `symbol`)
                - sector             Nifty sectoral index performance
                - earnings_calendar  upcoming earnings (uses `symbol` if given)
                - ipo                upcoming & recent IPOs
                - vix                India VIX fear index + history
                - gift_nifty         pre-market global + Nifty reference
                - quarterly_results  latest quarterly financials (needs `symbol`)
                - pcr                Nifty Put-Call Ratio trend across expiries
            symbol: NSE symbol for views that need one (corporate_actions,
                    quarterly_results, and optionally earnings_calendar).

        Returns:
            JSON: {views, count, results: {view: <result>}, ...}. A view that
            errors gets {"error": "..."} under its key.

        Examples:
            market_pulse("status,movers,fii_dii,vix")
            market_pulse("corporate_actions,quarterly_results", symbol="RELIANCE")
            market_pulse("sector,gift_nifty,pcr")
        """
        raw = [v.strip().lower() for v in views.split(",") if v.strip()]
        # de-duplicate, preserve caller order
        seen: set[str] = set()
        wanted = [v for v in raw if not (v in seen or seen.add(v))]

        if not wanted:
            return json.dumps({
                "error": "No views provided.",
                "valid_views": sorted(PULSE_VIEWS.keys()),
            }, indent=2)

        unknown = [v for v in wanted if v not in PULSE_VIEWS]
        if unknown:
            return json.dumps({
                "error": f"Unknown view(s): {', '.join(unknown)}.",
                "valid_views": sorted(PULSE_VIEWS.keys()),
            }, indent=2)

        sym = symbol.strip().upper()
        results: dict[str, object] = {}

        for view in wanted:
            fn = PULSE_VIEWS[view]
            try:
                if view in _SYMBOL_REQUIRED_VIEWS:
                    if not sym:
                        results[view] = {
                            "error": f"view '{view}' requires a `symbol` argument."
                        }
                        continue
                    results[view] = fn(sym)
                elif view == "earnings_calendar":
                    # optional symbol — matches earnings_calendar(symbol="")
                    results[view] = fn(sym)
                else:
                    results[view] = fn()
            except Exception as e:  # isolate per-view failures
                results[view] = {"error": f"{type(e).__name__}: {e}"}

        out = {
            "views": wanted,
            "count": len(wanted),
            "results": results,
        }
        if sym:
            out["symbol"] = sym
        return json.dumps(out, indent=2, default=str)

    @mcp.tool()
    def screen(
        exchange: str = "NSE",
        pe_max: float = 0,
        pe_min: float = 0,
        roe_min: float = 0,
        market_cap_min: float = 0,
        dividend_yield_min: float = 0,
        debt_equity_max: float = 0,
        sector: str = "",
        limit: int = 15,
    ) -> str:
        """Screen stocks by multiple financial criteria. [PRO]

        Filter Nifty 50 or S&P 500 stocks by P/E ratio, ROE, market cap,
        dividend yield, debt/equity ratio, and sector.

        Args:
            exchange: "NSE" for Indian stocks, "US" for US stocks (default: NSE)
            pe_max: Maximum P/E ratio (e.g., 15 for value stocks). 0 = no filter.
            pe_min: Minimum P/E ratio. 0 = no filter.
            roe_min: Minimum Return on Equity in % (e.g., 20). 0 = no filter.
            market_cap_min: Minimum market cap in USD (e.g., 1000000000 for $1B). 0 = no filter.
            dividend_yield_min: Minimum dividend yield in % (e.g., 2). 0 = no filter.
            debt_equity_max: Maximum debt-to-equity ratio (e.g., 50). 0 = no filter.
            sector: Filter by sector name (e.g., "Technology", "Financial"). Empty = all.
            limit: Max number of results (default: 15)

        Examples:
            screen("NSE", pe_max=15, roe_min=20) → Value stocks with high ROE
            screen("NSE", dividend_yield_min=3) → High dividend yield stocks
            screen("US", pe_max=20, sector="Technology") → Cheap US tech stocks
        """
        if not config.is_tool_allowed("stock_screener"):
            return json.dumps(tier_locked_error("stock_screener"), indent=2)

        result = screen_stocks(
            exchange=exchange,
            pe_max=pe_max if pe_max > 0 else None,
            pe_min=pe_min if pe_min > 0 else None,
            roe_min=roe_min if roe_min > 0 else None,
            market_cap_min=market_cap_min if market_cap_min > 0 else None,
            dividend_yield_min=dividend_yield_min if dividend_yield_min > 0 else None,
            debt_equity_max=debt_equity_max if debt_equity_max > 0 else None,
            sector=sector if sector else None,
            limit=min(limit, 25),
        )
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def funds(
        action: str = "nav",
        scheme_code: str = "",
        scheme_name: str = "",
        symbol1: str = "",
        symbol2: str = "",
    ) -> str:
        """Mutual-fund toolkit — NAV lookup, two-fund overlap, and AMFI flows.

        One tool for the common fund questions, routed by `action`.

        Args:
            action: which fund operation to run. One of:
                - nav      latest NAV + details for a fund. Uses `scheme_code`
                           (e.g. "119598") or `scheme_name` (e.g. "SBI Bluechip").
                - overlap  % holdings overlap between two funds. Needs `symbol1`
                           and `symbol2` (fund names, e.g. "HDFC Flexi Cap" and
                           "Mirae Asset Large Cap").
                - flows    AMFI mutual-fund industry AUM, SIP flows, scheme mix.
                           No other arguments needed.
            scheme_code: numeric AMFI scheme code (for action="nav").
            scheme_name: fund name (for action="nav").
            symbol1: first fund name (for action="overlap").
            symbol2: second fund name (for action="overlap").

        Returns:
            JSON for the chosen action. Unknown action -> error JSON with the
            valid actions listed.

        Examples:
            funds(action="nav", scheme_name="SBI Bluechip")
            funds(action="nav", scheme_code="119598")
            funds(action="overlap", symbol1="HDFC Flexi Cap", symbol2="Mirae Asset Large Cap")
            funds(action="flows")
        """
        act = action.strip().lower()

        if act == "nav":
            # mirror indian.py mutual_fund_nav -> get_mutual_fund_nav(query)
            query = scheme_code.strip() or scheme_name.strip()
            if not query:
                return json.dumps({
                    "error": "action 'nav' needs `scheme_code` or `scheme_name`.",
                }, indent=2)
            from finstack.data.nse_advanced import get_mutual_fund_nav
            return json.dumps(get_mutual_fund_nav(query), indent=2, default=str)

        if act == "overlap":
            # mirror intelligence.py get_mf_overlap -> get_mf_overlap(fund1, fund2)
            f1, f2 = symbol1.strip(), symbol2.strip()
            if not f1 or not f2:
                return json.dumps({
                    "error": "action 'overlap' needs both `symbol1` and `symbol2` (fund names).",
                }, indent=2)
            return json.dumps(_get_mf_overlap(f1, f2), indent=2, default=str)

        if act == "flows":
            # mirror market_intelligence.py amfi_fund_flows -> get_amfi_fund_flows()
            return json.dumps(get_amfi_fund_flows(), indent=2, default=str)

        return json.dumps({
            "error": f"unknown action '{action}'",
            "valid_actions": ["nav", "overlap", "flows"],
        }, indent=2)
