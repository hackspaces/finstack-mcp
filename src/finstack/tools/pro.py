"""
FinStack Pro Tools — live, key-free `macro`.

(The former `valuation` + `forensic_diagnostics` tools moved into the unified
`models` tool in tools/models.py.)
"""

from finstack.utils.respond import dumps

from finstack.data.macro_live import get_macro


def register_pro_tools(mcp):
    """Register the macro tool with the MCP server."""

    @mcp.tool()
    def macro(indicators: str = "", country: str = "IN") -> str:
        """Live macroeconomic indicators (key-free, official, provenance-stamped).

        Live World Bank + DBnomics data; every value carries `as_of`, `source`,
        and `is_stale` so nothing looks fresher than it is.

        Args:
            indicators: comma-separated names, or "" for a default set.
                Available: cpi_inflation, gdp_growth, gdp_usd,
                current_account_pct_gdp, unemployment, real_interest_rate,
                lending_rate, broad_money_growth, fdi_pct_gdp,
                gross_capital_formation_pct, gni_per_capita_usd, govt_debt_pct_gdp,
                short_term_rate, policy_rate.
            country: ISO code (default "IN"; also e.g. "US", "CN").

        Example:
            macro(indicators="cpi_inflation,gdp_growth,short_term_rate", country="IN")
        """
        names = [s.strip() for s in indicators.split(",") if s.strip()] or None
        return dumps(get_macro(names, country=country))
