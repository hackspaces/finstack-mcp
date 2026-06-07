"""
FinStack `models` — one computation-first tool for all deterministic financial
models. You supply the data (inputs/current/prior), the tool computes — so it's
never rate-limited or IP-blocked, and the math is exact (not LLM arithmetic).

Merges the former `valuation` + `forensic_diagnostics` tools.
"""

from finstack.utils.respond import dumps
from finstack.data import valuation_models as vm
from finstack.data import forensic as fz


def register_models_tools(mcp):
    """Register the unified financial-models tool."""

    @mcp.tool()
    def models(model: str, inputs: dict | None = None,
               current: dict | None = None, prior: dict | None = None) -> str:
        """Deterministic financial models (computation-first — YOU supply the data).

        Valuation:
            dcf            inputs: fcf0, growth_high, years_high, growth_terminal,
                           discount_rate, shares, net_debt
            reverse_dcf    inputs: price, fcf0, shares, discount_rate,
                           [years_high, growth_terminal, net_debt]  -> growth the price implies
            graham         inputs: eps, book_value_per_share
            owner_earnings inputs: net_income, depreciation_amortization,
                           maintenance_capex, [working_capital_change]
            epv            inputs: normalized_ebit, tax_rate, wacc, shares, [net_debt]

        Forensic / distress:
            beneish_m      earnings manipulation — uses `current` + `prior` dicts
                           (sales, cogs, net_receivables, current_assets, ppe_net,
                            total_assets, depreciation, sga, net_income, cfo,
                            current_liabilities, long_term_debt)
            piotroski_f    9-pt quality — uses `current` + `prior` dicts
                           (net_income, cfo, total_assets, long_term_debt,
                            current_assets, current_liabilities, shares,
                            gross_profit, sales)
            altman_z       inputs: working_capital, retained_earnings, ebit,
                           market_value_equity, total_liabilities, sales,
                           total_assets, [model="manufacturing"|"emerging"]
            sloan_accruals inputs: net_income, cfo, cfi, total_assets
            dupont         inputs: net_income, sales, total_assets, equity,
                           [ebit, pretax_income]
            merton_dd      inputs: equity_value, equity_volatility, debt_face,
                           [risk_free_rate, horizon_years]

        Rates are decimals (0.12 = 12%). Returns the result + interpretation.

        Examples:
            models(model="reverse_dcf", inputs={"price":1290,"fcf0":70000,"shares":6766,"discount_rate":0.12})
            models(model="altman_z", inputs={"working_capital":5e9,"retained_earnings":2e10,"ebit":8e9,
                   "market_value_equity":1.5e11,"total_liabilities":4e10,"sales":9e10,"total_assets":1.2e11,"model":"emerging"})
            models(model="beneish_m", current={...}, prior={...})
        """
        data = inputs or {}
        val = {"dcf": vm.two_stage_dcf, "reverse_dcf": vm.reverse_dcf, "graham": vm.graham_number,
               "owner_earnings": vm.owner_earnings, "epv": vm.earnings_power_value}
        try:
            if model in val:
                result = val[model](**data)
            elif model == "beneish_m":
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
                return dumps({"error": f"unknown model '{model}'",
                              "valuation": list(val),
                              "forensic": ["beneish_m", "piotroski_f", "altman_z",
                                           "sloan_accruals", "dupont", "merton_dd"]})
        except TypeError as e:
            return dumps({"error": f"missing/invalid inputs for '{model}': {e}"})
        return dumps(result)
