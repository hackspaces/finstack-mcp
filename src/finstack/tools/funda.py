"""
FinStack Fundamentals (umbrella) Tool

A single configurable tool that returns one or more "statements" for a company
in ONE call. Instead of calling income_statement / balance_sheet / cash_flow /
key_ratios / company_profile / dividend_history / technical_indicators
separately, pass a comma-separated list (or "all") and get a dict keyed by the
requested statements.

Each branch reuses the EXACT data-layer function that the existing single-purpose
tool wrappers call (see tools/fundamentals.py and tools/analytics.py):

    income      -> get_income_statement(symbol)        (annual)
    balance     -> get_balance_sheet(symbol)           (annual)
    cashflow    -> get_cash_flow(symbol)               (annual)
    ratios      -> get_key_ratios(symbol)
    profile     -> get_company_profile(symbol)
    dividends   -> get_dividend_history(symbol)
    technicals  -> compute_technical_indicators(symbol) (all indicators, 6mo)

Failures are isolated per statement so one bad branch never fails the whole call.
"""

import json
from finstack.utils.respond import dumps as _dumps

from finstack.data.fundamentals import (
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_key_ratios,
    get_company_profile,
    get_dividend_history,
)
from finstack.data.analytics import compute_technical_indicators


# statement name -> callable(symbol) -> dict
# Each lambda mirrors how the original single-purpose tool wrapper invoked the
# data-layer function (default args preserved: annual statements, all indicators).
STATEMENT_OPS = {
    "income": lambda symbol: get_income_statement(symbol),
    "balance": lambda symbol: get_balance_sheet(symbol),
    "cashflow": lambda symbol: get_cash_flow(symbol),
    "ratios": lambda symbol: get_key_ratios(symbol),
    "profile": lambda symbol: get_company_profile(symbol),
    "dividends": lambda symbol: get_dividend_history(symbol),
    "technicals": lambda symbol: compute_technical_indicators(symbol, "6mo", None),
}


def register_funda_tools(mcp):
    """Register the fundamentals umbrella tool with the MCP server."""

    @mcp.tool()
    def fundamentals(symbol: str, statements: str = "all") -> str:
        """Get one or MANY fundamental statements for a company in a single call.

        Instead of calling income_statement / balance_sheet / cash_flow /
        key_ratios / company_profile / dividend_history / technical_indicators
        separately, pass a comma-separated list (or "all") and get a dict keyed
        by the requested statements. Each statement is fetched independently and
        failures are isolated per statement.

        Args:
            symbol: Stock ticker (e.g., RELIANCE, TCS, AAPL, MSFT). Works for
                    both Indian (NSE/BSE) and US stocks.
            statements: comma-separated list, or "all" (default) for every one. One or more of:
                - income      annual income statement (P&L)
                - balance     annual balance sheet
                - cashflow    annual cash flow statement
                - ratios      key valuation/profitability/health ratios
                - profile     company overview (sector, industry, description)
                - dividends   historical dividend payments
                - technicals  technical indicators (RSI, MACD, SMA, etc.; 6mo, all)

        Returns:
            JSON string: {symbol, statements, results: {statement: <result>}, ...}.
            A statement that errors gets {"error": "..."} under its key.

        Examples:
            fundamentals(symbol="RELIANCE") → all statements for Reliance
            fundamentals(symbol="TCS", statements="income,ratios") → P&L + ratios
            fundamentals(symbol="AAPL", statements="profile,technicals")
        """
        raw = statements.strip().lower()
        if raw in ("", "all"):
            requested = list(STATEMENT_OPS.keys())
        else:
            requested = [s.strip().lower() for s in statements.split(",") if s.strip()]

        # validate up front; unknown -> fail loud listing valid statements
        unknown = [s for s in requested if s not in STATEMENT_OPS]
        if unknown:
            return _dumps({
                "error": f"Unknown statement(s): {unknown}.",
                "valid_statements": sorted(STATEMENT_OPS.keys()),
            }, indent=2)

        # de-duplicate while preserving requested order
        seen: set[str] = set()
        ordered = [s for s in requested if not (s in seen or seen.add(s))]

        if not ordered:
            return _dumps({
                "error": "No statements requested.",
                "valid_statements": sorted(STATEMENT_OPS.keys()),
            }, indent=2)

        results: dict[str, object] = {}
        for name in ordered:
            try:
                results[name] = STATEMENT_OPS[name](symbol)
            except Exception as e:  # isolate per-statement failures
                results[name] = {"error": f"{type(e).__name__}: {e}"}

        out = {
            "symbol": symbol,
            "statements": ordered,
            "count": len(ordered),
            "results": results,
        }
        return _dumps(out, indent=2, default=str)
