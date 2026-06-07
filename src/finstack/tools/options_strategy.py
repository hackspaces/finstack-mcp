"""
FinStack Options Strategy Builder — computation-first, datacenter-safe.

A single `options_strategy(...)` tool that *constructs and prices* standard
Indian-market option strategies entirely from a Black-Scholes model. It does
NOT fetch a live option chain: NSE option-chain endpoints are IP-blocked from
datacenters, so instead of scraping unreliable last-traded prices we price every
leg with a self-contained Black-Scholes engine. That makes the output
deterministic and reproducible on a headless host.

What it does:
  - Resolves spot (get_nse_quote if spot=0) and IV (forecast_volatility if iv=0,
    annualized GARCH vol; fallback 0.28).
  - Builds the chosen strategy's legs at sensible strikes (ATM for straddles,
    spot * (1 ± width_pct/100) for spreads/wings).
  - Prices each leg with a local _bs_price() (math.erf normal CDF, no scipy).
  - Computes per-leg greeks via market_intelligence._bs_greeks and nets them.
  - Computes net debit/credit, max profit, max loss, breakeven(s),
    risk:reward, and a full expiry payoff curve (spot*0.8 .. spot*1.2) for
    charting.
  - Emits a clear verdict/summary line.

Reuses (does not reinvent):
  - data.nse.get_nse_quote                          (spot)
  - data.quant_engine.forecast_volatility           (IV estimate)
  - data.market_intelligence._bs_greeks             (per-leg greeks)
  - utils.respond.dumps                             (slimmed JSON out)

Strategies: covered_call, protective_put, bull_call_spread, bear_put_spread,
cash_secured_put, long_straddle, long_strangle, iron_condor. strategy="auto"
maps from `view` (bullish/bearish/neutral/volatile).

Caps (512MB headless host): payoff grid is bounded to 41 points; at most 4 legs
per strategy; only two upstream network calls (quote + vol), run concurrently.
"""

import math
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from finstack.utils.respond import dumps
from finstack.data.nse import get_nse_quote
from finstack.data.quant_engine import forecast_volatility
from finstack.data.market_intelligence import _bs_greeks

logger = logging.getLogger("finstack.tools.options_strategy")

# ── constants / caps ───────────────────────────────────────────────
RISK_FREE_RATE = 0.065          # RBI repo-rate proxy
IV_FALLBACK = 0.28              # used when forecast_volatility is unavailable
PAYOFF_POINTS = 41              # bounded grid (spot*0.8 .. spot*1.2), 512MB-safe
GRID_LO, GRID_HI = 0.80, 1.20

# auto-selection: trader view -> strategy
VIEW_TO_STRATEGY = {
    "bullish": "bull_call_spread",
    "bearish": "bear_put_spread",
    "neutral": "iron_condor",
    "volatile": "long_straddle",
}

VALID_STRATEGIES = {
    "covered_call",
    "protective_put",
    "bull_call_spread",
    "bear_put_spread",
    "cash_secured_put",
    "long_straddle",
    "long_strangle",
    "iron_condor",
}


# ── self-contained Black-Scholes price (no scipy) ──────────────────
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (self-contained)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    """European Black-Scholes option price.

    S spot, K strike, T years to expiry, r risk-free (decimal), sigma IV (decimal),
    kind 'call'|'put'. Degenerates to intrinsic value when T or sigma <= 0.
    """
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        # intrinsic value only
        return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _round_strike(x: float) -> float:
    """Round a strike to a sensible NSE-like increment based on magnitude."""
    if x <= 0:
        return 0.0
    if x < 100:
        step = 2.5
    elif x < 500:
        step = 5.0
    elif x < 2000:
        step = 10.0
    elif x < 10000:
        step = 50.0
    else:
        step = 100.0
    return round(round(x / step) * step, 2)


# ── leg helpers ────────────────────────────────────────────────────
def _leg(action: str, kind: str, strike: float, S: float, T: float, sigma: float) -> dict:
    """Build a priced leg with greeks. action buy/sell, kind call/put."""
    premium = round(_bs_price(S, strike, T, RISK_FREE_RATE, sigma, kind), 2)
    g = _bs_greeks(S, strike, T, RISK_FREE_RATE, sigma, kind)
    sign = 1 if action == "buy" else -1
    return {
        "action": action,
        "type": kind,
        "strike": strike,
        "premium": premium,
        "greeks": g,
        "_sign": sign,  # internal: +1 long, -1 short (stripped before output)
    }


def _leg_payoff(leg: dict, price_at_expiry: float) -> float:
    """Net PnL of a single option leg at expiry for a given underlying price."""
    K = leg["strike"]
    if leg["type"] == "call":
        intrinsic = max(0.0, price_at_expiry - K)
    else:
        intrinsic = max(0.0, K - price_at_expiry)
    sign = leg["_sign"]
    # long: pay premium, receive intrinsic; short: receive premium, pay intrinsic
    return sign * (intrinsic - leg["premium"])


def _net_greeks(legs: list) -> dict:
    out = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for leg in legs:
        s = leg["_sign"]
        for k in out:
            out[k] += s * leg["greeks"].get(k, 0.0)
    return {k: round(v, 4) for k, v in out.items()}


# ── strategy construction ──────────────────────────────────────────
def _build_legs(strategy: str, S: float, T: float, sigma: float, width_pct: float) -> dict:
    """Return {legs:[...], stock_leg:bool, notes:str} for the chosen strategy."""
    w = max(0.5, width_pct) / 100.0
    atm = _round_strike(S)
    up = _round_strike(S * (1 + w))
    dn = _round_strike(S * (1 - w))
    up2 = _round_strike(S * (1 + 2 * w))
    dn2 = _round_strike(S * (1 - 2 * w))

    if strategy == "covered_call":
        return {
            "legs": [_leg("sell", "call", up, S, T, sigma)],
            "stock_leg": True,
            "notes": "Long 1 lot of stock at spot + short 1 OTM call. Income on a held position.",
        }
    if strategy == "protective_put":
        return {
            "legs": [_leg("buy", "put", dn, S, T, sigma)],
            "stock_leg": True,
            "notes": "Long 1 lot of stock at spot + long 1 OTM put. Insurance against a fall.",
        }
    if strategy == "bull_call_spread":
        return {
            "legs": [
                _leg("buy", "call", atm, S, T, sigma),
                _leg("sell", "call", up, S, T, sigma),
            ],
            "stock_leg": False,
            "notes": "Buy ATM call, sell OTM call. Defined-risk bullish.",
        }
    if strategy == "bear_put_spread":
        return {
            "legs": [
                _leg("buy", "put", atm, S, T, sigma),
                _leg("sell", "put", dn, S, T, sigma),
            ],
            "stock_leg": False,
            "notes": "Buy ATM put, sell OTM put. Defined-risk bearish.",
        }
    if strategy == "cash_secured_put":
        return {
            "legs": [_leg("sell", "put", dn, S, T, sigma)],
            "stock_leg": False,
            "notes": "Sell 1 OTM put backed by cash. Income / acquire stock lower.",
        }
    if strategy == "long_straddle":
        return {
            "legs": [
                _leg("buy", "call", atm, S, T, sigma),
                _leg("buy", "put", atm, S, T, sigma),
            ],
            "stock_leg": False,
            "notes": "Buy ATM call + ATM put. Long volatility / big-move bet.",
        }
    if strategy == "long_strangle":
        return {
            "legs": [
                _leg("buy", "call", up, S, T, sigma),
                _leg("buy", "put", dn, S, T, sigma),
            ],
            "stock_leg": False,
            "notes": "Buy OTM call + OTM put. Cheaper long-volatility bet, wider breakevens.",
        }
    if strategy == "iron_condor":
        return {
            "legs": [
                _leg("sell", "put", dn, S, T, sigma),
                _leg("buy", "put", dn2, S, T, sigma),
                _leg("sell", "call", up, S, T, sigma),
                _leg("buy", "call", up2, S, T, sigma),
            ],
            "stock_leg": False,
            "notes": "Short OTM put + call spreads. Range-bound, net-credit, defined-risk.",
        }
    raise ValueError(f"Unsupported strategy: {strategy}")


def _stock_pnl(price_at_expiry: float, S: float) -> float:
    """PnL on 1 unit of long stock bought at spot S."""
    return price_at_expiry - S


def _analyze(strategy: str, legs: list, stock_leg: bool, S: float) -> dict:
    """Compute net debit/credit, payoff curve, max P/L, breakevens, R:R."""
    # Net cash flow at entry: buying legs cost premium (debit, negative cash),
    # selling legs collect premium (credit, positive cash).
    net_cash = sum((1 if leg["action"] == "sell" else -1) * leg["premium"] for leg in legs)
    net_label = "credit" if net_cash > 0 else "debit"

    lo = S * GRID_LO
    hi = S * GRID_HI
    step = (hi - lo) / (PAYOFF_POINTS - 1)

    curve = []
    pnls = []
    breakevens = []
    prev_p = None
    prev_pnl = None
    for i in range(PAYOFF_POINTS):
        px = lo + i * step
        pnl = sum(_leg_payoff(leg, px) for leg in legs)
        if stock_leg:
            pnl += _stock_pnl(px, S)
        pnl = round(pnl, 2)
        curve.append({"price": round(px, 2), "pnl": pnl})
        pnls.append(pnl)
        # detect breakeven by sign change (linear interpolation)
        if prev_pnl is not None and (prev_pnl == 0 or (prev_pnl < 0) != (pnl < 0)):
            if pnl != prev_pnl:
                be = prev_p + (0 - prev_pnl) * (px - prev_p) / (pnl - prev_pnl)
                breakevens.append(round(be, 2))
        prev_p, prev_pnl = px, pnl

    max_profit = round(max(pnls), 2)
    max_loss = round(min(pnls), 2)

    # Risk:reward as |max_profit| : |max_loss| over the modeled grid.
    rr = None
    if max_loss < 0 and max_profit > 0:
        rr = round(abs(max_profit) / abs(max_loss), 3)

    # Honest caveat: for naked-short / unhedged-tail strategies the grid bounds
    # the loss/gain; flag when an extreme sits at a grid edge (theoretically larger).
    edge_flags = []
    if pnls[0] == max_loss or pnls[-1] == max_loss:
        edge_flags.append("max_loss sits at a grid edge — true loss may be larger outside ±20% band")
    if pnls[0] == max_profit or pnls[-1] == max_profit:
        edge_flags.append("max_profit sits at a grid edge — true profit may be larger outside ±20% band")

    return {
        "net": round(net_cash, 2),
        "net_type": net_label,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakevens": sorted(set(breakevens)),
        "risk_reward": rr,
        "payoff_curve": curve,
        "grid_caveats": edge_flags,
    }


def _verdict(strategy: str, view: str, a: dict, S: float, sigma: float) -> str:
    rr = a["risk_reward"]
    bes = a["breakevens"]
    be_txt = ", ".join(f"₹{b}" for b in bes) if bes else "n/a"
    rr_txt = f"{rr}:1 reward:risk" if rr else "undefined R:R (grid-bounded)"
    return (
        f"{strategy.replace('_', ' ').title()} on a {view} view at spot ₹{round(S, 2)} "
        f"(IV {round(sigma * 100, 1)}%): net {a['net_type']} ₹{abs(a['net'])}, "
        f"max profit ₹{a['max_profit']}, max loss ₹{a['max_loss']}, "
        f"breakeven(s) {be_txt}, {rr_txt}. Prices are Black-Scholes model marks, "
        f"not live chain quotes."
    )


# ── MCP registration ───────────────────────────────────────────────
def register_options_strategy_tools(mcp):
    """Register the options_strategy tool with the MCP server."""

    @mcp.tool()
    def options_strategy(
        symbol: str = "",
        view: str = "bullish",
        strategy: str = "auto",
        spot: float = 0,
        iv: float = 0,
        expiry_days: int = 30,
        width_pct: float = 5,
        lots_capital: float = 0,
    ) -> str:
        """Build and price a standard options strategy from a Black-Scholes model.

        COMPUTATION-FIRST and datacenter-safe: this does NOT fetch a live NSE
        option chain (those endpoints are IP-blocked from servers). Every leg is
        priced with a self-contained Black-Scholes engine, so output is
        deterministic. Greeks come from the shared Black-Scholes greeks engine.

        Spot defaults to the live NSE quote (get_nse_quote) when spot=0. IV
        defaults to the GARCH annualized volatility forecast (forecast_volatility)
        when iv=0, falling back to 0.28 if unavailable. r = 0.065, T = expiry_days/365.

        Strategies: covered_call, protective_put, bull_call_spread, bear_put_spread,
        cash_secured_put, long_straddle, long_strangle, iron_condor.
        strategy="auto" picks from `view`:
        bullish->bull_call_spread, bearish->bear_put_spread,
        neutral->iron_condor, volatile->long_straddle.

        Args:
            symbol: NSE symbol, e.g. "RELIANCE" (needed when spot=0).
            view: "bullish" | "bearish" | "neutral" | "volatile" (drives strategy="auto").
            strategy: explicit strategy name or "auto".
            spot: underlying price; 0 = fetch live via get_nse_quote(symbol).
            iv: annualized implied volatility as a decimal (e.g. 0.30); 0 = estimate.
            expiry_days: days to expiry (T = expiry_days/365). Default 30.
            width_pct: strike offset for spreads/wings, in % of spot. Default 5.
            lots_capital: optional capital (₹) to scale the strategy; if > 0, returns
                a rough per-share-to-capital multiplier note (informational only).

        Returns:
            JSON string with: strategy, view, spot, iv, expiry_days, legs
            [{action, type, strike, premium, greeks}], net (+credit/-debit),
            net_type, max_profit, max_loss, breakevens, risk_reward, net_greeks,
            payoff_curve [{price, pnl}], grid_caveats, and a verdict line.

        Example (Indian market):
            options_strategy(symbol="RELIANCE", view="bullish", strategy="auto",
                             expiry_days=30, width_pct=5)
            -> bull call spread on RELIANCE, ATM long call / 5%-OTM short call,
               priced via Black-Scholes with net debit, max profit/loss and the
               expiry payoff curve for charting.
        """
        result = {
            "tool": "options_strategy",
            "symbol": (symbol or "").upper(),
            "view": view,
            "data_basis": "Black-Scholes model marks (no live chain — NSE chain is IP-blocked from datacenters)",
            "risk_free_rate": RISK_FREE_RATE,
            "timestamp": datetime.now().isoformat(),
        }

        # ── resolve strategy ──
        strat = (strategy or "auto").strip().lower()
        view_l = (view or "bullish").strip().lower()
        if strat == "auto":
            strat = VIEW_TO_STRATEGY.get(view_l)
            if not strat:
                return dumps({
                    **result,
                    "error": f"Unknown view '{view}' for strategy=auto.",
                    "valid_views": sorted(VIEW_TO_STRATEGY.keys()),
                })
            result["strategy_auto_selected"] = True
        if strat not in VALID_STRATEGIES:
            return dumps({
                **result,
                "error": f"Unknown strategy '{strategy}'.",
                "valid_strategies": sorted(VALID_STRATEGIES),
            })
        result["strategy"] = strat

        # ── resolve spot + iv concurrently (two independent network calls) ──
        S = float(spot) if spot and spot > 0 else 0.0
        sigma = float(iv) if iv and iv > 0 else 0.0
        spot_meta = {"source": "user"} if S > 0 else {}
        iv_meta = {"source": "user"} if sigma > 0 else {}

        need_spot = S <= 0
        need_iv = sigma <= 0

        if (need_spot or need_iv):
            if not symbol:
                return dumps({
                    **result,
                    "error": "symbol is required when spot=0 or iv=0 (need it to fetch live data).",
                })
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_spot = ex.submit(get_nse_quote, symbol) if need_spot else None
                fut_iv = ex.submit(forecast_volatility, symbol) if need_iv else None

                if fut_spot is not None:
                    try:
                        q = fut_spot.result()
                        if isinstance(q, dict) and not q.get("error") and q.get("price"):
                            S = float(q["price"])
                            spot_meta = {"source": "get_nse_quote", "name": q.get("name")}
                        else:
                            return dumps({
                                **result,
                                "error": f"Could not resolve spot for '{symbol}'.",
                                "quote_response": q if isinstance(q, dict) else str(q),
                            })
                    except Exception as e:
                        return dumps({**result, "error": f"Spot fetch failed: {type(e).__name__}: {e}"})

                if fut_iv is not None:
                    try:
                        fv = fut_iv.result()
                        if isinstance(fv, dict) and not fv.get("error") and fv.get("current_annual_volatility"):
                            sigma = float(fv["current_annual_volatility"])
                            iv_meta = {"source": "forecast_volatility(GARCH)", "model": fv.get("model")}
                        else:
                            sigma = IV_FALLBACK
                            iv_meta = {"source": "fallback", "reason": (fv.get("error") if isinstance(fv, dict) else "no vol"), "value": IV_FALLBACK}
                    except Exception as e:
                        sigma = IV_FALLBACK
                        iv_meta = {"source": "fallback", "reason": f"{type(e).__name__}: {e}", "value": IV_FALLBACK}

        if S <= 0:
            return dumps({**result, "error": "Resolved spot is non-positive; cannot price."})
        if sigma <= 0:
            sigma = IV_FALLBACK
            iv_meta = {"source": "fallback", "value": IV_FALLBACK}

        T = max(int(expiry_days), 1) / 365.0
        result.update({
            "spot": round(S, 2),
            "spot_meta": spot_meta,
            "iv": round(sigma, 4),
            "iv_pct": round(sigma * 100, 2),
            "iv_meta": iv_meta,
            "expiry_days": int(expiry_days),
            "T_years": round(T, 4),
            "width_pct": width_pct,
        })

        # ── build + price legs (fail-loud per section) ──
        try:
            built = _build_legs(strat, S, T, sigma, width_pct)
        except Exception as e:
            return dumps({**result, "error": f"Leg construction failed: {type(e).__name__}: {e}"})

        legs = built["legs"]
        try:
            analysis = _analyze(strat, legs, built["stock_leg"], S)
        except Exception as e:
            return dumps({**result, "error": f"Payoff analysis failed: {type(e).__name__}: {e}"})

        try:
            net_g = _net_greeks(legs)
            if built["stock_leg"]:
                # long stock contributes +1 delta per share
                net_g["delta"] = round(net_g["delta"] + 1.0, 4)
        except Exception as e:
            net_g = {"error": f"{type(e).__name__}: {e}"}

        # strip internal _sign before emitting
        public_legs = [
            {"action": l["action"], "type": l["type"], "strike": l["strike"],
             "premium": l["premium"], "greeks": l["greeks"]}
            for l in legs
        ]

        result.update({
            "includes_stock_leg": built["stock_leg"],
            "structure_note": built["notes"],
            "legs": public_legs,
            "net": analysis["net"],
            "net_type": analysis["net_type"],
            "net_debit_or_credit": ("+%.2f credit" % analysis["net"]) if analysis["net"] > 0
                                   else ("-%.2f debit" % abs(analysis["net"])),
            "max_profit": analysis["max_profit"],
            "max_loss": analysis["max_loss"],
            "breakevens": analysis["breakevens"],
            "risk_reward": analysis["risk_reward"],
            "net_greeks": net_g,
            "payoff_curve": analysis["payoff_curve"],
            "payoff_grid": {"points": PAYOFF_POINTS, "range": f"spot*{GRID_LO}..spot*{GRID_HI}"},
            "grid_caveats": analysis["grid_caveats"],
        })

        if lots_capital and lots_capital > 0 and S > 0:
            # informational: how many shares the capital notionally covers at spot.
            result["capital_note"] = {
                "lots_capital": lots_capital,
                "shares_at_spot": round(float(lots_capital) / S, 2),
                "note": "Per-share P/L above × shares ≈ position P/L. Lot size not modeled; "
                        "multiply by the contract lot size for actual exposure.",
            }

        result["verdict"] = _verdict(strat, view_l, analysis, S, sigma)
        result["caveats"] = [
            "Black-Scholes European pricing; Indian equity options are American (early exercise not modeled).",
            "Premiums are theoretical marks, not live bid/ask — actual fills will differ.",
            "Lot sizes, brokerage, STT and slippage are NOT included.",
            "Payoff is at expiry only; no intraday/T+0 path is modeled.",
        ]
        return dumps(result)
