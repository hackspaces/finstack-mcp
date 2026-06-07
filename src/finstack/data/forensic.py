"""
FinStack Forensic & Distress Diagnostics (computation-first)

Under-the-radar quantitative screens that institutions use but retail tools
rarely expose. All are pure functions over explicit financial inputs (supplied
by the caller / LLM) — no data fetching, so they run anywhere.

  - beneish_m_score   : earnings-manipulation likelihood (8-factor)
  - altman_z          : bankruptcy/distress (manufacturing + emerging-market)
  - piotroski_f       : 9-point fundamental quality
  - sloan_accruals    : earnings quality via accruals (high = low quality)
  - dupont            : ROE decomposition (3- and 5-step)
  - merton_distance_to_default : structural credit risk from equity + vol
"""

from __future__ import annotations

import math


def beneish_m_score(cur: dict, prev: dict) -> dict:
    """Beneish M-score — likelihood that earnings are manipulated.

    `cur` and `prev` are dicts for the current and prior year, each with:
      sales, cogs, net_receivables, current_assets, ppe_net, total_assets,
      depreciation, sga, net_income, cfo, current_liabilities, long_term_debt
    M > -1.78  => likely manipulator.
    """
    try:
        # 1. DSRI - Days Sales in Receivables Index
        dsri = (cur["net_receivables"] / cur["sales"]) / (prev["net_receivables"] / prev["sales"])
        # 2. GMI - Gross Margin Index (deterioration > 1)
        gm_c = (cur["sales"] - cur["cogs"]) / cur["sales"]
        gm_p = (prev["sales"] - prev["cogs"]) / prev["sales"]
        gmi = gm_p / gm_c
        # 3. AQI - Asset Quality Index
        aq_c = 1 - (cur["current_assets"] + cur["ppe_net"]) / cur["total_assets"]
        aq_p = 1 - (prev["current_assets"] + prev["ppe_net"]) / prev["total_assets"]
        aqi = aq_c / aq_p if aq_p != 0 else 1.0
        # 4. SGI - Sales Growth Index
        sgi = cur["sales"] / prev["sales"]
        # 5. DEPI - Depreciation Index
        dr_c = cur["depreciation"] / (cur["depreciation"] + cur["ppe_net"])
        dr_p = prev["depreciation"] / (prev["depreciation"] + prev["ppe_net"])
        depi = dr_p / dr_c if dr_c != 0 else 1.0
        # 6. SGAI - SG&A Index
        sgai = (cur["sga"] / cur["sales"]) / (prev["sga"] / prev["sales"])
        # 7. LVGI - Leverage Index
        lev_c = (cur["long_term_debt"] + cur["current_liabilities"]) / cur["total_assets"]
        lev_p = (prev["long_term_debt"] + prev["current_liabilities"]) / prev["total_assets"]
        lvgi = lev_c / lev_p if lev_p != 0 else 1.0
        # 8. TATA - Total Accruals to Total Assets
        tata = (cur["net_income"] - cur["cfo"]) / cur["total_assets"]
    except (KeyError, ZeroDivisionError, TypeError) as e:
        return {"error": f"Beneish needs full current+prior financials: {e}"}

    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    return {
        "model": "beneish_m_score",
        "m_score": round(m, 3),
        "threshold": -1.78,
        "verdict": "LIKELY_MANIPULATOR" if m > -1.78 else "unlikely_manipulator",
        "components": {k: round(v, 3) for k, v in
                       dict(DSRI=dsri, GMI=gmi, AQI=aqi, SGI=sgi, DEPI=depi,
                            SGAI=sgai, LVGI=lvgi, TATA=tata).items()},
        "note": "M > -1.78 flags possible earnings manipulation. Screen, not proof.",
    }


def altman_z(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_value_equity: float,
    total_liabilities: float,
    sales: float,
    total_assets: float,
    model: str = "manufacturing",
) -> dict:
    """Altman Z-score — distress/bankruptcy risk.

    model: "manufacturing" (original Z) or "emerging" (Z'' = 3.25 + 6.56 X1 +
    3.26 X2 + 6.72 X3 + 1.05 X4) — Altman's emerging-market variant, better
    for Indian non-manufacturers.
    """
    if total_assets <= 0:
        return {"error": "total_assets must be > 0."}
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_value_equity / total_liabilities if total_liabilities else 0.0

    if model == "emerging":
        z = 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        safe, distress = 2.60, 1.10
    else:
        x5 = sales / total_assets
        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        safe, distress = 2.99, 1.81

    zone = "safe" if z > safe else ("distress" if z < distress else "grey")
    return {
        "model": f"altman_z_{model}",
        "z_score": round(z, 3),
        "zone": zone,
        "bands": {"safe_above": safe, "distress_below": distress},
        "components": {"X1_wc_ta": round(x1, 3), "X2_re_ta": round(x2, 3),
                       "X3_ebit_ta": round(x3, 3), "X4_mve_tl": round(x4, 3)},
    }


def piotroski_f(cur: dict, prev: dict) -> dict:
    """Piotroski F-score (0-9) — fundamental quality. 9 binary tests.

    Each dict needs: net_income, cfo, total_assets, long_term_debt,
    current_assets, current_liabilities, shares, gross_profit, sales.
    """
    try:
        roa_c = cur["net_income"] / cur["total_assets"]
        roa_p = prev["net_income"] / prev["total_assets"]
        cr_c = cur["current_assets"] / cur["current_liabilities"]
        cr_p = prev["current_assets"] / prev["current_liabilities"]
        gm_c = cur["gross_profit"] / cur["sales"]
        gm_p = prev["gross_profit"] / prev["sales"]
        at_c = cur["sales"] / cur["total_assets"]
        at_p = prev["sales"] / prev["total_assets"]
        lev_c = cur["long_term_debt"] / cur["total_assets"]
        lev_p = prev["long_term_debt"] / prev["total_assets"]
        tests = {
            "positive_net_income": cur["net_income"] > 0,
            "positive_cfo": cur["cfo"] > 0,
            "roa_improving": roa_c > roa_p,
            "cfo_gt_net_income": cur["cfo"] > cur["net_income"],   # accrual quality
            "leverage_decreasing": lev_c < lev_p,
            "current_ratio_improving": cr_c > cr_p,
            "no_dilution": cur["shares"] <= prev["shares"],
            "gross_margin_improving": gm_c > gm_p,
            "asset_turnover_improving": at_c > at_p,
        }
    except (KeyError, ZeroDivisionError, TypeError) as e:
        return {"error": f"Piotroski needs full current+prior financials: {e}"}

    score = sum(1 for v in tests.values() if v)
    rating = "strong" if score >= 7 else ("weak" if score <= 2 else "average")
    return {"model": "piotroski_f", "f_score": score, "max": 9,
            "rating": rating, "tests": {k: bool(v) for k, v in tests.items()}}


def sloan_accruals(net_income: float, cfo: float, cfi: float, total_assets: float) -> dict:
    """Sloan ratio = (NI - CFO - CFI) / Total Assets. |ratio| > 10% = low earnings quality."""
    if total_assets <= 0:
        return {"error": "total_assets must be > 0."}
    ratio = (net_income - cfo - cfi) / total_assets
    quality = "high" if abs(ratio) <= 0.10 else ("warning" if abs(ratio) <= 0.25 else "low")
    return {
        "model": "sloan_accruals",
        "sloan_ratio": round(ratio, 4),
        "sloan_ratio_pct": f"{ratio * 100:.2f}%",
        "earnings_quality": quality,
        "note": "High accruals (earnings not backed by cash) tend to mean-revert; |ratio|>10% is a flag.",
    }


def dupont(net_income: float, sales: float, total_assets: float, equity: float,
           ebit: float | None = None, pretax_income: float | None = None) -> dict:
    """DuPont ROE decomposition (3-step always; 5-step if ebit & pretax supplied)."""
    if sales <= 0 or total_assets <= 0 or equity <= 0:
        return {"error": "sales, total_assets, equity must be > 0."}
    net_margin = net_income / sales
    asset_turnover = sales / total_assets
    equity_multiplier = total_assets / equity
    roe = net_margin * asset_turnover * equity_multiplier
    out = {
        "model": "dupont", "roe": round(roe, 4), "roe_pct": f"{roe*100:.2f}%",
        "three_step": {"net_margin": round(net_margin, 4),
                       "asset_turnover": round(asset_turnover, 3),
                       "equity_multiplier": round(equity_multiplier, 3)},
    }
    if ebit is not None and pretax_income is not None and ebit != 0 and pretax_income != 0:
        out["five_step"] = {
            "tax_burden": round(net_income / pretax_income, 4),
            "interest_burden": round(pretax_income / ebit, 4),
            "operating_margin": round(ebit / sales, 4),
            "asset_turnover": round(asset_turnover, 3),
            "leverage": round(equity_multiplier, 3),
        }
    return out


def merton_distance_to_default(
    equity_value: float,
    equity_volatility: float,
    debt_face: float,
    risk_free_rate: float = 0.065,
    horizon_years: float = 1.0,
) -> dict:
    """Merton structural credit model — distance-to-default & implied PD.

    Treats equity as a call option on firm assets. Solves the 2-equation system
    for asset value V and asset vol σV, then DD = (ln(V/F)+(r-0.5σV²)T)/(σV√T),
    PD = N(-DD). `debt_face` is the default point (≈ current liab + ½ long-term debt).
    """
    if equity_value <= 0 or equity_volatility <= 0 or debt_face <= 0:
        return {"error": "equity_value, equity_volatility, debt_face must be > 0."}
    try:
        from scipy.optimize import fsolve
        from scipy.stats import norm
    except Exception as e:  # pragma: no cover
        return {"error": f"scipy required for Merton model: {e}"}

    E, sigE, F, r, T = equity_value, equity_volatility, debt_face, risk_free_rate, horizon_years

    def equations(p):
        V, sigV = p
        if V <= 0 or sigV <= 0:
            return [1e6, 1e6]
        d1 = (math.log(V / F) + (r + 0.5 * sigV**2) * T) / (sigV * math.sqrt(T))
        d2 = d1 - sigV * math.sqrt(T)
        eq1 = V * norm.cdf(d1) - F * math.exp(-r * T) * norm.cdf(d2) - E
        eq2 = norm.cdf(d1) * sigV * V - sigE * E
        return [eq1, eq2]

    try:
        V, sigV = fsolve(equations, [E + F, sigE], full_output=False)
        d2 = (math.log(V / F) + (r - 0.5 * sigV**2) * T) / (sigV * math.sqrt(T))
        dd = d2
        pd = float(norm.cdf(-dd))
    except Exception as e:  # pragma: no cover
        return {"error": f"Merton solve failed: {e}"}

    return {
        "model": "merton_distance_to_default",
        "distance_to_default": round(float(dd), 3),
        "implied_default_probability": round(pd, 5),
        "implied_default_probability_pct": f"{pd*100:.3f}%",
        "implied_asset_value": round(float(V), 2),
        "implied_asset_volatility": round(float(sigV), 4),
        "note": "Higher DD = safer. PD is the model's 1-year default probability.",
        "inputs": {"equity_value": E, "equity_volatility": sigE, "debt_face": F,
                   "risk_free_rate": r, "horizon_years": T},
    }
