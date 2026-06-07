"""
FinStack `risk_report` — PM-grade portfolio risk X-ray.

Sector exposure, concentration (HHI/top-5), correlation/diversification,
portfolio annualized vol, 1-day 95% VaR & CVaR, beta vs Nifty, max drawdown,
and risk flags — from one batched price fetch.
"""

import math

import numpy as np
import pandas as pd

from finstack.utils.respond import dumps
from finstack.data.sector_engine import _batch_close, ALL_BASKETS
from finstack.data.universe import UNIVERSE

# symbol -> NSE sector (from the official nse_* baskets)
_SECTOR_OF = {}
for _name, _b in ALL_BASKETS.items():
    if _name.startswith("nse_"):
        for _s in _b["symbols"]:
            _SECTOR_OF.setdefault(_s, _b.get("industry", _name))


def register_risk_report_tools(mcp):
    """Register the portfolio risk-report tool."""

    @mcp.tool()
    def risk_report(holdings: list, benchmark: str = "^NSEI") -> str:
        """Portfolio risk X-ray for a set of holdings (PM-grade dashboard).

        Args:
            holdings: list of dicts — {"symbol","qty","avg_price"} (weights by market
                      value) OR {"symbol","weight"} OR just {"symbol"} (equal weight).
            benchmark: beta benchmark (default Nifty 50 "^NSEI").

        Returns:
            weights, sector_exposure, concentration (HHI/top5/largest), correlation
            + diversification, portfolio vol/VaR/CVaR/beta/max_drawdown, flags, rating.

        Example:
            risk_report(holdings=[{"symbol":"RELIANCE","qty":10,"avg_price":1200},
                                  {"symbol":"TCS","qty":5},{"symbol":"HDFCBANK","weight":0.3}])
        """
        if not holdings or not isinstance(holdings, list):
            return dumps({"error": "holdings must be a non-empty list of {symbol,...}"})
        syms = [str(h.get("symbol", "")).strip().upper() for h in holdings if h.get("symbol")]
        syms = list(dict.fromkeys([s for s in syms if s]))
        if not syms:
            return dumps({"error": "no valid symbols in holdings"})

        try:
            close = _batch_close(syms + [benchmark], "1y")
        except Exception as e:
            return dumps({"error": f"price fetch failed: {e}"})
        resolved = [s for s in syms if s in close.columns]
        if len(resolved) < 1:
            return dumps({"error": "no price data resolved for holdings"})

        # ---- weights ----
        by = {str(h["symbol"]).strip().upper(): h for h in holdings if h.get("symbol")}
        mv, raw_w = {}, {}
        for s in resolved:
            h = by.get(s, {})
            px = float(close[s].dropna().iloc[-1]) if close[s].notna().any() else None
            if h.get("qty") and px:
                mv[s] = float(h["qty"]) * px
            elif h.get("weight") is not None:
                raw_w[s] = float(h["weight"])
        if mv:
            tot = sum(mv.values()) or 1.0
            weights = {s: mv.get(s, 0) / tot for s in resolved}
        elif raw_w:
            tot = sum(raw_w.values()) or 1.0
            weights = {s: raw_w.get(s, 0) / tot for s in resolved}
        else:
            weights = {s: 1.0 / len(resolved) for s in resolved}

        # ---- sector exposure ----
        sect = {}
        for s, w in weights.items():
            sct = _SECTOR_OF.get(s, "Unclassified")
            sect[sct] = sect.get(sct, 0) + w
        sector_exposure = {k: round(v * 100, 1) for k, v in sorted(sect.items(), key=lambda x: -x[1])}

        # ---- concentration ----
        wv = sorted(weights.values(), reverse=True)
        hhi = sum(w * w for w in weights.values())
        top5 = sum(wv[:5])
        largest = max(weights.items(), key=lambda x: x[1])

        # ---- returns / risk ----
        rets = close[resolved].pct_change().dropna(how="all")
        w_vec = np.array([weights[s] for s in resolved])
        w_vec = w_vec / w_vec.sum()
        port = (rets[resolved].fillna(0) * w_vec).sum(axis=1)
        ann_vol = float(port.std() * math.sqrt(252) * 100)
        var95 = float(np.percentile(port.dropna(), 5) * 100)
        cvar95 = float(port[port <= np.percentile(port, 5)].mean() * 100)
        # beta vs benchmark
        beta = None
        if benchmark in close.columns:
            br = close[benchmark].pct_change().reindex(port.index).dropna()
            j = port.reindex(br.index).dropna()
            br = br.reindex(j.index)
            if len(j) > 30 and br.var() > 0:
                beta = float(np.cov(j, br)[0, 1] / br.var())
        # max drawdown of weighted equity curve
        eq = (1 + port).cumprod()
        mdd = float((eq / eq.cummax() - 1).min() * 100)
        # avg pairwise correlation
        corr = rets[resolved].corr()
        iu = np.triu_indices_from(corr.values, k=1)
        avg_corr = float(corr.values[iu].mean()) if len(iu[0]) else None

        # ---- flags + rating ----
        flags = []
        if largest[1] > 0.25:
            flags.append(f"Concentrated: {largest[0]} is {largest[1]*100:.0f}% of book (>25%)")
        for k, v in sect.items():
            if v > 0.40:
                flags.append(f"Sector concentration: {k} {v*100:.0f}% (>40%)")
        if avg_corr is not None and avg_corr > 0.6:
            flags.append(f"Low diversification: avg pairwise correlation {avg_corr:.2f} (>0.6)")
        score = sum([ann_vol > 30, abs(var95) > 3, largest[1] > 0.25, (avg_corr or 0) > 0.6,
                     top5 > 0.7, (beta or 1) > 1.2])
        rating = "HIGH" if score >= 3 else ("ELEVATED" if score == 2 else "MODERATE" if score == 1 else "CONTAINED")

        return dumps({
            "risk_rating": rating, "holdings_resolved": len(resolved), "holdings_requested": len(syms),
            "flags": flags or ["none material"],
            "portfolio_risk": {
                "annualized_vol_pct": round(ann_vol, 2), "var_95_1d_pct": round(var95, 2),
                "cvar_95_1d_pct": round(cvar95, 2), "beta_vs_benchmark": round(beta, 2) if beta is not None else None,
                "max_drawdown_pct": round(mdd, 2),
            },
            "concentration": {"herfindahl_index": round(hhi, 3), "top5_weight_pct": round(top5 * 100, 1),
                              "largest_position": {"symbol": largest[0], "weight_pct": round(largest[1] * 100, 1),
                                                   "name": UNIVERSE.get(largest[0], "")}},
            "diversification": {"avg_pairwise_correlation": round(avg_corr, 3) if avg_corr is not None else None,
                                "read": ("well diversified" if (avg_corr or 0) < 0.4 else
                                         "moderately diversified" if (avg_corr or 0) < 0.6 else "highly correlated")},
            "sector_exposure_pct": sector_exposure,
            "weights_pct": {s: round(w * 100, 1) for s, w in sorted(weights.items(), key=lambda x: -x[1])},
            "note": "VaR/CVaR are 1-day 95% historical; vol/beta/drawdown from 1y daily.",
        })
