"""
FinStack Valuation Engines (computation-first)

Pure-Python intrinsic-valuation models. These take explicit financial inputs
(supplied by the caller / LLM) and return a valuation — they do NOT fetch data,
so they can never be rate-limited or IP-blocked and run anywhere.

Models:
  - two_stage_dcf       : high-growth stage + Gordon terminal value
  - reverse_dcf         : solve for the FCF growth the current price implies
  - graham_number       : Benjamin Graham's fair-value ceiling
  - owner_earnings      : Buffett owner-earnings (NI + D&A - maint capex - dWC)
  - earnings_power_value: Greenwald EPV (no-growth value of current earnings)

All money values should be in the same currency/unit; per-share outputs follow.
"""

from __future__ import annotations

import math


def _f(x) -> float:
    return float(x)


def two_stage_dcf(
    fcf0: float,
    growth_high: float,
    years_high: int,
    growth_terminal: float,
    discount_rate: float,
    shares: float,
    net_debt: float = 0.0,
) -> dict:
    """Two-stage free-cash-flow DCF -> intrinsic value per share.

    growth_high / growth_terminal / discount_rate are decimals (0.12 = 12%).
    net_debt = total debt - cash (subtracted from enterprise value).
    """
    if discount_rate <= growth_terminal:
        return {"error": "discount_rate must exceed growth_terminal for a finite terminal value."}
    if shares <= 0:
        return {"error": "shares must be > 0."}

    pv_stage1 = 0.0
    fcf = _f(fcf0)
    projected = []
    for yr in range(1, int(years_high) + 1):
        fcf = fcf * (1 + growth_high)
        pv = fcf / (1 + discount_rate) ** yr
        pv_stage1 += pv
        projected.append({"year": yr, "fcf": round(fcf, 2), "pv": round(pv, 2)})

    # terminal value at end of stage 1 (Gordon growth), discounted back
    fcf_terminal = fcf * (1 + growth_terminal)
    terminal_value = fcf_terminal / (discount_rate - growth_terminal)
    pv_terminal = terminal_value / (1 + discount_rate) ** int(years_high)

    enterprise_value = pv_stage1 + pv_terminal
    equity_value = enterprise_value - net_debt
    per_share = equity_value / shares

    return {
        "model": "two_stage_dcf",
        "intrinsic_value_per_share": round(per_share, 2),
        "equity_value": round(equity_value, 2),
        "enterprise_value": round(enterprise_value, 2),
        "pv_high_growth_stage": round(pv_stage1, 2),
        "pv_terminal_value": round(pv_terminal, 2),
        "terminal_value_pct_of_ev": round(100 * pv_terminal / enterprise_value, 1) if enterprise_value else None,
        "assumptions": {
            "fcf0": fcf0, "growth_high": growth_high, "years_high": int(years_high),
            "growth_terminal": growth_terminal, "discount_rate": discount_rate,
            "shares": shares, "net_debt": net_debt,
        },
        "projection": projected,
    }


def reverse_dcf(
    price: float,
    fcf0: float,
    shares: float,
    discount_rate: float,
    years_high: int = 10,
    growth_terminal: float = 0.04,
    net_debt: float = 0.0,
) -> dict:
    """Solve for the high-growth FCF rate the CURRENT price already implies.

    Instead of guessing growth to get a value, this inverts the DCF: given what
    the market pays today, what FCF growth must it believe? Compare that implied
    growth to what you think is achievable to judge over/under-valuation.
    """
    if discount_rate <= growth_terminal:
        return {"error": "discount_rate must exceed growth_terminal."}
    if shares <= 0 or price <= 0:
        return {"error": "price and shares must be > 0."}

    target_equity = price * shares

    def equity_for_growth(g: float) -> float:
        return (two_stage_dcf(fcf0, g, years_high, growth_terminal, discount_rate,
                              shares, net_debt)["equity_value"])

    # bisection on growth in [-50%, +100%]
    lo, hi = -0.50, 1.00
    f_lo = equity_for_growth(lo) - target_equity
    f_hi = equity_for_growth(hi) - target_equity
    if f_lo * f_hi > 0:
        return {
            "model": "reverse_dcf",
            "error": "Implied growth is outside the searchable range (-50%..+100%).",
            "hint": "Price may be unsupported by FCF at this discount rate, or inputs are off.",
        }
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = equity_for_growth(mid) - target_equity
        if abs(f_mid) < 1e-6 or (hi - lo) < 1e-7:
            break
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid

    implied = (lo + hi) / 2
    return {
        "model": "reverse_dcf",
        "implied_growth_high": round(implied, 4),
        "implied_growth_high_pct": f"{implied * 100:.2f}%",
        "interpretation": (
            f"At ₹{price}, the market is pricing ~{implied*100:.1f}% annual FCF growth "
            f"for {int(years_high)} years (then {growth_terminal*100:.1f}% terminal). "
            "If you believe the company can beat that, it's cheap; if not, it's dear."
        ),
        "assumptions": {
            "price": price, "fcf0": fcf0, "shares": shares,
            "discount_rate": discount_rate, "years_high": int(years_high),
            "growth_terminal": growth_terminal, "net_debt": net_debt,
        },
    }


def graham_number(eps: float, book_value_per_share: float) -> dict:
    """Benjamin Graham's fair-value ceiling: sqrt(22.5 * EPS * BVPS)."""
    if eps <= 0 or book_value_per_share <= 0:
        return {"error": "Graham number needs positive EPS and book value per share."}
    val = math.sqrt(22.5 * eps * book_value_per_share)
    return {
        "model": "graham_number",
        "graham_number": round(val, 2),
        "note": "Graham's max price for a defensive investor (22.5 = 15 P/E × 1.5 P/B).",
        "inputs": {"eps": eps, "book_value_per_share": book_value_per_share},
    }


def owner_earnings(
    net_income: float,
    depreciation_amortization: float,
    maintenance_capex: float,
    working_capital_change: float = 0.0,
) -> dict:
    """Buffett owner earnings = NI + D&A - maintenance capex - ΔWorking Capital."""
    oe = net_income + depreciation_amortization - maintenance_capex - working_capital_change
    return {
        "model": "owner_earnings",
        "owner_earnings": round(oe, 2),
        "note": "Cash an owner could extract without impairing the business.",
        "inputs": {
            "net_income": net_income, "depreciation_amortization": depreciation_amortization,
            "maintenance_capex": maintenance_capex, "working_capital_change": working_capital_change,
        },
    }


def earnings_power_value(
    normalized_ebit: float,
    tax_rate: float,
    wacc: float,
    shares: float,
    net_debt: float = 0.0,
) -> dict:
    """Greenwald EPV: value of current earnings assuming ZERO growth.

    EPV(enterprise) = NOPAT / WACC ; equity = EPV - net_debt.
    Below the DCF value because it credits no growth — a conservative floor.
    """
    if wacc <= 0 or shares <= 0:
        return {"error": "wacc and shares must be > 0."}
    nopat = normalized_ebit * (1 - tax_rate)
    epv_enterprise = nopat / wacc
    epv_equity = epv_enterprise - net_debt
    return {
        "model": "earnings_power_value",
        "epv_per_share": round(epv_equity / shares, 2),
        "epv_equity": round(epv_equity, 2),
        "epv_enterprise": round(epv_enterprise, 2),
        "nopat": round(nopat, 2),
        "note": "No-growth value of current earnings — a conservative floor vs DCF.",
        "inputs": {
            "normalized_ebit": normalized_ebit, "tax_rate": tax_rate,
            "wacc": wacc, "shares": shares, "net_debt": net_debt,
        },
    }
