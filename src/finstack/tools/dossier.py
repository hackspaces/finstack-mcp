"""
FinStack Dossier — the institutional one-pager.

ONE configurable MCP tool (`dossier`) that builds a desk-grade research one-pager
for a SINGLE NSE stock by concurrently gathering and then SYNTHESIZING:

  - snapshot     : data.nse.get_nse_quote
  - ratios       : data.fundamentals.get_key_ratios
  - technicals   : data.tech_engine.technicals(views="all")  (incl. volume read)
  - risk         : data.quant_engine.compute_risk_metrics
  - forecast     : data.quant_engine.mean_reversion + monte_carlo(horizon=21)
  - models (AUTO-FED from yfinance financials — user supplies NOTHING):
      * reverse_dcf : FCF0 = operating cash flow - capex (from get_cash_flow);
                      shares = market_cap / price; current price -> implied growth.
      * graham_number : eps + book value per share (from get_key_ratios).
      * altman_z (emerging) : working_capital = current_assets - current_liabilities,
                      retained_earnings, ebit, total_liabilities, sales, total_assets
                      (from get_balance_sheet + get_income_statement); MVE = market cap.

The superpower is the AUTO-FEED: the pure-computation valuation/forensic engines
normally need the caller to supply every input by hand. Here we map them straight
off the yfinance statements so a portfolio manager just names the ticker.

Output is decision-grade: a SCORECARD first (six dimensions each scored -2..+2 with
a one-line reason, an overall BULLISH/NEUTRAL/BEARISH stance, and the 3 biggest
supports + 3 biggest risks), THEN the underlying evidence sections.

Every section is wrapped in its own try/except: one failing component never sinks
the call — it is reported with an error note and the rest still render. We never
fabricate a number; a model that can't be fed is marked unavailable with the reason.
"""

from __future__ import annotations

import concurrent.futures

from finstack.utils.respond import dumps

from finstack.data.nse import get_nse_quote
from finstack.data.fundamentals import (
    get_key_ratios,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
)
from finstack.data.tech_engine import technicals
from finstack.data.quant_engine import (
    compute_risk_metrics,
    mean_reversion,
    monte_carlo,
)
from finstack.data.valuation_models import reverse_dcf, graham_number
from finstack.data.forensic import altman_z

# Bound work for a 512MB host. Monte-Carlo sims capped; horizon fixed at 21d.
_MC_SIMS = 4000
_MC_HORIZON = 21


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — tolerant lookups over yfinance statement rows / quote / ratios
# ─────────────────────────────────────────────────────────────────────────────

def _is_err(d) -> bool:
    return not isinstance(d, dict) or bool(d.get("error"))


def _latest(stmt: dict) -> dict:
    """Return the most-recent period dict from a statement payload, or {}.

    get_income_statement/get_balance_sheet/get_cash_flow return
    {..., "data": [<newest>, <older>, ...]}. Columns are taken newest-first.
    """
    if _is_err(stmt):
        return {}
    data = stmt.get("data") or []
    return data[0] if data else {}


def _pick(row: dict, *candidates):
    """First non-None value among candidate snake_case keys (yfinance row names).

    yfinance row labels are lowercased with spaces -> underscores upstream, but
    the exact label varies by ticker/yfinance version, so we try several spellings.
    """
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _num(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-FED models
# ─────────────────────────────────────────────────────────────────────────────

def _build_reverse_dcf(quote: dict, ratios: dict, cashflow: dict) -> dict:
    """Feed reverse_dcf from yfinance: FCF0 = CFO - capex, shares = mcap/price."""
    cf = _latest(cashflow)
    if not cf:
        return {"available": False, "reason": "cash flow statement unavailable"}

    cfo = _num(_pick(cf, "operating_cash_flow", "cash_flow_from_continuing_operating_activities",
                     "total_cash_from_operating_activities"))
    capex = _num(_pick(cf, "capital_expenditure", "capital_expenditures",
                       "net_ppe_purchase_and_sale"))
    fcf_direct = _num(_pick(cf, "free_cash_flow"))

    if cfo is not None and capex is not None:
        # capex is reported negative in yfinance; FCF = CFO + capex (capex<0) == CFO - |capex|.
        fcf0 = cfo + capex if capex < 0 else cfo - capex
    elif fcf_direct is not None:
        fcf0 = fcf_direct
    else:
        return {"available": False, "reason": "could not derive FCF (no CFO/capex or free_cash_flow)"}

    if fcf0 is None or fcf0 <= 0:
        return {"available": False, "reason": f"non-positive trailing FCF ({fcf0}); reverse DCF not meaningful"}

    price = _num(quote.get("price")) if not _is_err(quote) else None
    mcap = _num(quote.get("market_cap")) if not _is_err(quote) else None
    if mcap is None and not _is_err(ratios):
        mcap = _num((ratios.get("valuation") or {}).get("market_cap"))
    if not price or not mcap:
        return {"available": False, "reason": "missing price or market cap to derive share count"}

    shares = mcap / price
    if shares <= 0:
        return {"available": False, "reason": "derived non-positive share count"}

    # 12% discount rate is a defensible Indian-equity cost of equity default; noted in output.
    res = reverse_dcf(price=price, fcf0=fcf0, shares=shares, discount_rate=0.12,
                      years_high=10, growth_terminal=0.04)
    if _is_err(res):
        return {"available": False, "reason": res.get("error", "reverse_dcf failed"), **res}
    res["available"] = True
    res["auto_fed_inputs"] = {
        "fcf0": round(fcf0, 2), "shares_derived": round(shares, 2),
        "price": price, "market_cap": mcap, "discount_rate": 0.12,
        "note": "FCF0 = operating cash flow - capex (yfinance); shares = market_cap/price; r=12% default.",
    }
    return res


def _build_graham(ratios: dict) -> dict:
    """Feed graham_number with trailing EPS + book value per share from key ratios."""
    if _is_err(ratios):
        return {"available": False, "reason": "key ratios unavailable"}
    ps = ratios.get("per_share") or {}
    eps = _num(_pick(ps, "eps_trailing", "eps_forward"))
    bvps = _num(ps.get("book_value"))
    if eps is None or bvps is None:
        return {"available": False, "reason": "missing trailing EPS or book value per share"}
    res = graham_number(eps=eps, book_value_per_share=bvps)
    if _is_err(res):
        return {"available": False, "reason": res.get("error", "graham_number failed"), **res}
    res["available"] = True
    return res


def _build_altman(quote: dict, ratios: dict, balance: dict, income: dict) -> dict:
    """Feed altman_z (emerging-market variant) from balance sheet + income statement."""
    bs = _latest(balance)
    inc = _latest(income)
    if not bs or not inc:
        return {"available": False, "reason": "balance sheet or income statement unavailable"}

    cur_assets = _num(_pick(bs, "current_assets", "total_current_assets"))
    cur_liab = _num(_pick(bs, "current_liabilities", "total_current_liabilities"))
    retained = _num(_pick(bs, "retained_earnings"))
    total_liab = _num(_pick(bs, "total_liabilities_net_minority_interest",
                            "total_liabilities", "total_liab"))
    total_assets = _num(_pick(bs, "total_assets"))

    ebit = _num(_pick(inc, "ebit", "operating_income", "operating_revenue_minus_operating_expense"))
    sales = _num(_pick(inc, "total_revenue", "operating_revenue", "revenue"))

    # MVE = market cap (preferred), else from ratios.
    mve = _num(quote.get("market_cap")) if not _is_err(quote) else None
    if mve is None and not _is_err(ratios):
        mve = _num((ratios.get("valuation") or {}).get("market_cap"))

    missing = [n for n, v in (
        ("current_assets", cur_assets), ("current_liabilities", cur_liab),
        ("retained_earnings", retained), ("total_liabilities", total_liab),
        ("total_assets", total_assets), ("ebit", ebit), ("sales", sales),
        ("market_value_equity", mve),
    ) if v is None]
    if missing:
        return {"available": False, "reason": f"missing fields from statements: {', '.join(missing)}"}

    working_capital = cur_assets - cur_liab
    res = altman_z(
        working_capital=working_capital, retained_earnings=retained, ebit=ebit,
        market_value_equity=mve, total_liabilities=total_liab, sales=sales,
        total_assets=total_assets, model="emerging",
    )
    if _is_err(res):
        return {"available": False, "reason": res.get("error", "altman_z failed"), **res}
    res["available"] = True
    res["auto_fed_inputs"] = {
        "working_capital": round(working_capital, 2), "retained_earnings": round(retained, 2),
        "ebit": round(ebit, 2), "market_value_equity": round(mve, 2),
        "total_liabilities": round(total_liab, 2), "sales": round(sales, 2),
        "total_assets": round(total_assets, 2),
        "note": "Emerging-market Z'' variant (Altman) — better for Indian non-manufacturers.",
    }
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Scorecard synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(score: int) -> int:
    return max(-2, min(2, score))


def _score_valuation(ratios, rev_dcf, graham, quote) -> dict:
    s = 0
    bits = []
    # Reverse DCF: implied growth vs a ~12% plausibility bar.
    if isinstance(rev_dcf, dict) and rev_dcf.get("available") and rev_dcf.get("implied_growth_high") is not None:
        ig = rev_dcf["implied_growth_high"]
        if ig < 0.06:
            s += 2; bits.append(f"market prices only ~{ig*100:.0f}% FCF growth (low bar to beat)")
        elif ig < 0.12:
            s += 1; bits.append(f"market prices ~{ig*100:.0f}% FCF growth (achievable)")
        elif ig < 0.20:
            bits.append(f"market prices ~{ig*100:.0f}% FCF growth (demanding)")
        else:
            s -= 2; bits.append(f"market prices ~{ig*100:.0f}% FCF growth (heroic)")
    # Graham number vs price.
    if isinstance(graham, dict) and graham.get("available") and not _is_err(quote):
        gn = _num(graham.get("graham_number")); px = _num(quote.get("price"))
        if gn and px:
            if px < gn:
                s += 1; bits.append(f"below Graham number ₹{gn:.0f}")
            elif px > gn * 1.5:
                s -= 1; bits.append(f"well above Graham number ₹{gn:.0f}")
    # P/E sanity.
    if not _is_err(ratios):
        pe = _num((ratios.get("valuation") or {}).get("pe_trailing"))
        if pe is not None:
            if pe < 15:
                s += 1; bits.append(f"P/E {pe:.1f} undemanding")
            elif pe > 45:
                s -= 1; bits.append(f"P/E {pe:.1f} rich")
    return {"score": _clamp(s), "reason": "; ".join(bits) or "insufficient valuation inputs"}


def _score_quality(ratios, altman) -> dict:
    s = 0
    bits = []
    if not _is_err(ratios):
        prof = ratios.get("profitability") or {}
        roe = _num(prof.get("roe"))
        if roe is not None:
            if roe >= 0.18:
                s += 2; bits.append(f"ROE {roe*100:.0f}% (excellent)")
            elif roe >= 0.12:
                s += 1; bits.append(f"ROE {roe*100:.0f}% (solid)")
            elif roe < 0.05:
                s -= 1; bits.append(f"ROE {roe*100:.0f}% (weak)")
        fh = ratios.get("financial_health") or {}
        dte = _num(fh.get("debt_to_equity"))
        if dte is not None:
            # yfinance debtToEquity is a percent (e.g. 60 == 0.60x).
            dte_x = dte / 100 if dte > 5 else dte
            if dte_x <= 0.5:
                s += 1; bits.append(f"low leverage (D/E {dte_x:.2f}x)")
            elif dte_x >= 1.5:
                s -= 1; bits.append(f"high leverage (D/E {dte_x:.2f}x)")
    if isinstance(altman, dict) and altman.get("available"):
        zone = altman.get("zone"); z = altman.get("z_score")
        if zone == "safe":
            s += 1; bits.append(f"Altman Z safe ({z})")
        elif zone == "distress":
            s -= 2; bits.append(f"Altman Z distress ({z})")
        elif zone == "grey":
            bits.append(f"Altman Z grey zone ({z})")
    return {"score": _clamp(s), "reason": "; ".join(bits) or "insufficient quality inputs"}


def _score_momentum(risk, mr) -> dict:
    s = 0
    bits = []
    if not _is_err(risk):
        ar = _num(risk.get("annual_return"))
        if ar is not None:
            if ar > 0.25:
                s += 2; bits.append(f"trailing return +{ar*100:.0f}%")
            elif ar > 0.08:
                s += 1; bits.append(f"trailing return +{ar*100:.0f}%")
            elif ar < -0.10:
                s -= 2; bits.append(f"trailing return {ar*100:.0f}%")
            elif ar < 0:
                s -= 1; bits.append(f"trailing return {ar*100:.0f}%")
        alpha = _num(risk.get("alpha_annual"))
        if alpha is not None:
            if alpha > 0.05:
                s += 1; bits.append(f"+{alpha*100:.0f}% alpha vs Nifty")
            elif alpha < -0.05:
                s -= 1; bits.append(f"{alpha*100:.0f}% alpha vs Nifty")
    if isinstance(mr, dict) and not _is_err(mr):
        if mr.get("regime") == "trending":
            bits.append("trending regime (momentum persists)")
    return {"score": _clamp(s), "reason": "; ".join(bits) or "insufficient momentum inputs"}


def _score_technical(tech) -> dict:
    if _is_err(tech):
        return {"score": 0, "reason": "technicals unavailable"}
    summ = tech.get("summary") or {}
    bias = summ.get("bias")
    bull = summ.get("bull_signals") or 0
    bear = summ.get("bear_signals") or 0
    s = 0
    bits = []
    if bias == "BULLISH":
        s += 2; bits.append(f"composite BULLISH ({bull} bull / {bear} bear)")
    elif bias == "BEARISH":
        s -= 2; bits.append(f"composite BEARISH ({bull} bull / {bear} bear)")
    else:
        bits.append(f"composite NEUTRAL ({bull} bull / {bear} bear)")
    sig = tech.get("signals") or {}
    if sig.get("rsi") == "overbought":
        s -= 1; bits.append("RSI overbought")
    elif sig.get("rsi") == "oversold":
        s += 1; bits.append("RSI oversold")
    if summ.get("vs_sma200") == "above":
        bits.append("above 200DMA")
    elif summ.get("vs_sma200") == "below":
        bits.append("below 200DMA")
    return {"score": _clamp(s), "reason": "; ".join(bits)}


def _score_volume(tech) -> dict:
    if _is_err(tech):
        return {"score": 0, "reason": "volume read unavailable"}
    vol = tech.get("volume") or {}
    if not vol:
        return {"score": 0, "reason": "no volume data"}
    s = 0
    bits = []
    diverg = vol.get("price_volume_divergence") or "none"
    if "confirmed_uptrend" in diverg:
        s += 2; bits.append("uptrend confirmed by rising volume")
    elif "bearish" in diverg:
        s -= 2; bits.append("price up on fading volume (distribution risk)")
    elif "bullish" in diverg:
        s += 1; bits.append("selloff on fading volume (supply drying up)")
    cmf = _num(vol.get("cmf_20"))
    if cmf is not None:
        if cmf > 0.05:
            s += 1; bits.append(f"CMF +{cmf:.2f} (accumulation)")
        elif cmf < -0.05:
            s -= 1; bits.append(f"CMF {cmf:.2f} (distribution)")
    if vol.get("obv_trend") == "up":
        bits.append("OBV rising")
    elif vol.get("obv_trend") == "down":
        bits.append("OBV falling")
    return {"score": _clamp(s), "reason": "; ".join(bits) or "neutral flow"}


def _score_risk(risk, mc) -> dict:
    if _is_err(risk):
        return {"score": 0, "reason": "risk metrics unavailable"}
    s = 0
    bits = []
    sharpe = _num(risk.get("sharpe_ratio"))
    if sharpe is not None:
        if sharpe > 1.5:
            s += 2; bits.append(f"Sharpe {sharpe:.2f} (excellent risk-adjusted return)")
        elif sharpe > 0.8:
            s += 1; bits.append(f"Sharpe {sharpe:.2f}")
        elif sharpe < 0:
            s -= 2; bits.append(f"Sharpe {sharpe:.2f} (return not compensating risk)")
        elif sharpe < 0.3:
            s -= 1; bits.append(f"Sharpe {sharpe:.2f} (thin)")
    mdd = _num(risk.get("max_drawdown"))
    if mdd is not None:
        if mdd < -0.40:
            s -= 1; bits.append(f"max drawdown {mdd*100:.0f}% (deep)")
        elif mdd > -0.15:
            s += 1; bits.append(f"shallow max drawdown {mdd*100:.0f}%")
    vol = _num(risk.get("annual_volatility"))
    if vol is not None and vol > 0.45:
        s -= 1; bits.append(f"high annual vol {vol*100:.0f}%")
    return {"score": _clamp(s), "reason": "; ".join(bits) or "neutral risk profile"}


def _synthesize(scorecard: dict) -> dict:
    total = sum(d["score"] for d in scorecard.values())
    if total >= 3:
        stance = "BULLISH"
    elif total <= -3:
        stance = "BEARISH"
    else:
        stance = "NEUTRAL"

    # Supports = most positive dims; risks = most negative dims.
    ordered = sorted(scorecard.items(), key=lambda kv: kv[1]["score"])
    risks = [f"{dim}: {d['reason']}" for dim, d in ordered if d["score"] < 0][:3]
    supports = [f"{dim}: {d['reason']}" for dim, d in reversed(ordered) if d["score"] > 0][:3]
    if not supports:
        supports = ["No dimension scored positive — thesis is defensive/avoid."]
    if not risks:
        risks = ["No dimension scored negative — clean read, watch for mean reversion."]

    return {
        "overall_stance": stance,
        "composite_score": total,
        "score_range": "-12..+12 (six dimensions, each -2..+2)",
        "biggest_supports": supports,
        "biggest_risks": risks,
    }


def register_dossier_tools(mcp):
    """Register the dossier one-pager tool with the MCP server."""

    @mcp.tool()
    def dossier(symbol: str, period: str = "1y") -> str:
        """Institutional one-pager for ONE NSE stock — scorecard FIRST, then evidence.

        Concurrently gathers a live snapshot, key ratios, full technicals + a volume
        read, a risk profile, and two forecasts (mean-reversion + 21-day Monte-Carlo),
        and AUTO-FEEDS three intrinsic models off yfinance financials so you supply
        NOTHING beyond the ticker:

          - reverse_dcf  : the FCF growth the current price already implies
                           (FCF0 = operating cash flow - capex; shares = mcap/price; r=12%)
          - graham_number: Graham's defensive fair-value ceiling (trailing EPS + BVPS)
          - altman_z     : emerging-market distress Z'' (off balance sheet + income stmt)

        It then synthesizes a six-dimension SCORECARD — {valuation, quality, momentum,
        technical, volume, risk}, each scored -2..+2 with a one-line reason — an overall
        BULLISH/NEUTRAL/BEARISH stance, and the 3 biggest supports + 3 biggest risks.

        Every section is isolated: one failed component is reported as an error note and
        the rest still render. Models that can't be auto-fed are marked unavailable with
        the reason — numbers are never fabricated. Monte-Carlo is capped at
        4000 sims over a 21-day horizon to stay responsive on a small host.

        Args:
            symbol: NSE symbol, e.g. "RELIANCE", "TCS", "HDFCBANK" (".NS" auto-appended).
            period: history window for technicals/risk/forecasts (e.g. "1y", "2y").
                    Default "1y".

        Returns:
            Compact JSON string. Top-level keys:
              symbol, scorecard {<dimension>: {score, reason}}, verdict {overall_stance,
              composite_score, biggest_supports, biggest_risks}, then evidence sections:
              snapshot, ratios, technicals, risk, forecast {mean_reversion, monte_carlo},
              models {reverse_dcf, graham_number, altman_z}, and caps.

        Example:
            dossier(symbol="RELIANCE")
            dossier(symbol="HDFCBANK", period="2y")
            dossier(symbol="TCS", period="1y")
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return dumps({"error": "symbol is required, e.g. 'RELIANCE'."})
        per = (period or "1y").strip() or "1y"

        # Each gatherer is self-contained and returns either its payload or an error dict.
        def g_quote():
            return get_nse_quote(sym)

        def g_ratios():
            return get_key_ratios(sym)

        def g_tech():
            return technicals(sym, views="all", period=per)

        def g_risk():
            return compute_risk_metrics(sym, benchmark="^NSEI", period=per)

        def g_mr():
            return mean_reversion(sym)

        def g_mc():
            return monte_carlo(sym, horizon=_MC_HORIZON, sims=_MC_SIMS)

        def g_income():
            return get_income_statement(sym)

        def g_balance():
            return get_balance_sheet(sym)

        def g_cashflow():
            return get_cash_flow(sym)

        tasks = {
            "quote": g_quote, "ratios": g_ratios, "tech": g_tech, "risk": g_risk,
            "mr": g_mr, "mc": g_mc, "income": g_income, "balance": g_balance,
            "cashflow": g_cashflow,
        }

        out: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as ex:
            futs = {name: ex.submit(fn) for name, fn in tasks.items()}
            for name, fut in futs.items():
                try:
                    out[name] = fut.result()
                except Exception as e:  # one failure never sinks the call
                    out[name] = {"error": f"{type(e).__name__}: {e}"}

        quote, ratios, tech = out["quote"], out["ratios"], out["tech"]
        risk, mr, mc = out["risk"], out["mr"], out["mc"]
        income, balance, cashflow = out["income"], out["balance"], out["cashflow"]

        # AUTO-FED models (each self-isolating).
        def _safe(fn, *args):
            try:
                return fn(*args)
            except Exception as e:
                return {"available": False, "reason": f"{type(e).__name__}: {e}"}

        rev_dcf = _safe(_build_reverse_dcf, quote, ratios, cashflow)
        graham = _safe(_build_graham, ratios)
        altman = _safe(_build_altman, quote, ratios, balance, income)

        # Scorecard (each dimension self-isolating so one bad input can't sink it).
        def _safe_dim(fn, *args):
            try:
                return fn(*args)
            except Exception as e:
                return {"score": 0, "reason": f"scoring error: {type(e).__name__}: {e}"}

        scorecard = {
            "valuation": _safe_dim(_score_valuation, ratios, rev_dcf, graham, quote),
            "quality": _safe_dim(_score_quality, ratios, altman),
            "momentum": _safe_dim(_score_momentum, risk, mr),
            "technical": _safe_dim(_score_technical, tech),
            "volume": _safe_dim(_score_volume, tech),
            "risk": _safe_dim(_score_risk, risk, mc),
        }
        verdict = _synthesize(scorecard)

        result = {
            "symbol": sym,
            "as_of": (quote.get("timestamp") if not _is_err(quote) else None),
            # SCORECARD + verdict FIRST — the read.
            "verdict": verdict,
            "scorecard": scorecard,
            # Evidence sections — the proof.
            "snapshot": quote,
            "ratios": ratios,
            "technicals": tech,
            "risk": risk,
            "forecast": {"mean_reversion": mr, "monte_carlo": mc},
            "models": {
                "reverse_dcf": rev_dcf,
                "graham_number": graham,
                "altman_z_emerging": altman,
            },
            "caps": {
                "monte_carlo_sims": _MC_SIMS,
                "monte_carlo_horizon_days": _MC_HORIZON,
                "note": "Sims/horizon capped for a 512MB host; models auto-fed from yfinance "
                        "financials (latest reported period). Unavailable models carry a reason.",
            },
        }
        return dumps(result)
