"""
FinStack Pro Tools — Claude-first, configurable analytics.

Three high-leverage tools (vs many narrow ones):
  - macro                : live, key-free macro with provenance/freshness stamps
  - valuation            : computation-first intrinsic valuation (data from caller)
  - forensic_diagnostics : computation-first forensic/distress screens

The computation tools take an `inputs` dict (or current/prior dicts) supplied by
the caller/LLM — they never fetch data, so they can't be rate-limited or blocked.
"""

import json

from finstack.data.macro_live import get_macro, WB_INDICATORS, DBNOMICS_RATES
from finstack.data import valuation_models as vm
from finstack.data import forensic as fz


def register_pro_tools(mcp):
    """Register the pro analytics tools with the MCP server."""

    @mcp.tool()
    def macro(indicators: str = "", country: str = "IN") -> str:
        """Live macroeconomic indicators (key-free, official, provenance-stamped).

        Replaces stale hardcoded macro with live World Bank + DBnomics data. Every
        value carries `as_of`, `source`, and `is_stale` so nothing looks fresher
        than it is.

        Args:
            indicators: comma-separated indicator names, or "" for a default set.
                Available: cpi_inflation, gdp_growth, gdp_usd,
                current_account_pct_gdp, unemployment, real_interest_rate,
                lending_rate, broad_money_growth, fdi_pct_gdp,
                gross_capital_formation_pct, gni_per_capita_usd, govt_debt_pct_gdp,
                short_term_rate, policy_rate.
            country: ISO code (default "IN" for India; also e.g. "US", "CN").

        Example:
            macro(indicators="cpi_inflation,gdp_growth,short_term_rate", country="IN")
        """
        names = [s.strip() for s in indicators.split(",") if s.strip()] or None
        return json.dumps(get_macro(names, country=country), indent=2, default=str)

    @mcp.tool()
    def valuation(method: str, inputs: dict | None = None) -> str:
        """Intrinsic-valuation engines (computation-first — YOU supply the inputs).

        Args:
            method: one of:
                - dcf            inputs: fcf0, growth_high, years_high,
                                 growth_terminal, discount_rate, shares, net_debt
                - reverse_dcf    inputs: price, fcf0, shares, discount_rate,
                                 [years_high, growth_terminal, net_debt]
                                 -> solves for the growth the price already implies
                - graham         inputs: eps, book_value_per_share
                - owner_earnings inputs: net_income, depreciation_amortization,
                                 maintenance_capex, [working_capital_change]
                - epv            inputs: normalized_ebit, tax_rate, wacc, shares,
                                 [net_debt]
            inputs: dict of the named numbers above (rates as decimals, 0.12 = 12%).

        Example:
            valuation(method="reverse_dcf", inputs={"price":1290,"fcf0":70000,
                      "shares":6766,"discount_rate":0.12})
        """
        data = inputs or {}
        fns = {
            "dcf": vm.two_stage_dcf,
            "reverse_dcf": vm.reverse_dcf,
            "graham": vm.graham_number,
            "owner_earnings": vm.owner_earnings,
            "epv": vm.earnings_power_value,
        }
        if method not in fns:
            return json.dumps({"error": f"unknown method '{method}'",
                               "valid_methods": sorted(fns)}, indent=2)
        try:
            result = fns[method](**data)
        except TypeError as e:
            return json.dumps({"error": f"missing/invalid inputs for '{method}': {e}"}, indent=2)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def forensic_diagnostics(model: str, inputs: dict | None = None,
                             current: dict | None = None, prior: dict | None = None) -> str:
        """Forensic / distress screens (computation-first — YOU supply financials).

        Args:
            model: one of:
                - beneish_m    earnings-manipulation (uses `current` + `prior`, each:
                               sales, cogs, net_receivables, current_assets, ppe_net,
                               total_assets, depreciation, sga, net_income, cfo,
                               current_liabilities, long_term_debt)
                - piotroski_f  9-pt quality (uses `current` + `prior`, each:
                               net_income, cfo, total_assets, long_term_debt,
                               current_assets, current_liabilities, shares,
                               gross_profit, sales)
                - altman_z     inputs: working_capital, retained_earnings, ebit,
                               market_value_equity, total_liabilities, sales,
                               total_assets, [model="manufacturing"|"emerging"]
                - sloan_accruals inputs: net_income, cfo, cfi, total_assets
                - dupont       inputs: net_income, sales, total_assets, equity,
                               [ebit, pretax_income]
                - merton_dd    inputs: equity_value, equity_volatility, debt_face,
                               [risk_free_rate, horizon_years]
            inputs / current / prior: dicts of the named numbers above.

        Example:
            forensic_diagnostics(model="altman_z", inputs={"working_capital":5e9,
              "retained_earnings":2e10,"ebit":8e9,"market_value_equity":1.5e11,
              "total_liabilities":4e10,"sales":9e10,"total_assets":1.2e11,"model":"emerging"})
        """
        data = inputs or {}
        try:
            if model == "beneish_m":
                result = fz.beneish_m_score(current or {}, prior or {})
            elif model == "piotroski_f":
                result = fz.piotroski_f(current or {}, prior or {})
            elif model == "altman_z":
                result = fz.altman_z(**data)
            elif model == "sloan_accruals":
                result = fz.sloan_accruals(**data)
            elif model == "dupont":
                result = fz.dupont(**data)
            elif model == "merton_dd":
                result = fz.merton_distance_to_default(**data)
            else:
                return json.dumps({"error": f"unknown model '{model}'",
                                   "valid_models": ["beneish_m", "piotroski_f", "altman_z",
                                                    "sloan_accruals", "dupont", "merton_dd"]}, indent=2)
        except TypeError as e:
            return json.dumps({"error": f"missing/invalid inputs for '{model}': {e}"}, indent=2)
        return json.dumps(result, indent=2, default=str)
