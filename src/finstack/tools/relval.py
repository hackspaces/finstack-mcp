"""
FinStack `relative_value` — stat-arb pair finder.

Within a basket (or custom list), test every pair for cointegration, rank the
tradeable ones by p-value then |z|, with hedge ratio, half-life and signal.
Prefetches all prices ONCE (one batched call) and computes pairs in-memory to
avoid a rate-limit storm.
"""

import itertools
import math

import numpy as np

from finstack.utils.respond import dumps
from finstack.data.sector_engine import _batch_close, ALL_BASKETS


def _half_life(spread: np.ndarray):
    """Half-life of mean reversion via AR(1) on the spread."""
    s = spread[~np.isnan(spread)]
    if len(s) < 30:
        return None
    lag = s[:-1]
    delta = np.diff(s)
    b = np.polyfit(lag, delta, 1)[0]
    return round(float(-math.log(2) / b), 1) if b < 0 else None


def register_relval_tools(mcp):
    """Register the relative-value (stat-arb pair finder) tool."""

    @mcp.tool()
    def relative_value(basket: str = "", symbols: str = "",
                       max_pairs: int = 10, period: str = "2y") -> str:
        """Find cointegrated, tradeable stat-arb pairs within a basket or symbol list.

        Args:
            basket: a basket name (see sector tool's `list`); capped to first ~12 names.
            symbols: comma-separated custom tickers (alternative to basket; capped ~12).
            max_pairs: max ranked pairs to return (default 10).
            period: history window (default "2y").

        Returns:
            Ranked cointegrated pairs (p<0.05) with hedge ratio, current spread z-score,
            half-life, and a signal (LONG/SHORT_SPREAD when |z|>2).

        Example:
            relative_value(basket="private_banks")
            relative_value(symbols="HDFCBANK,ICICIBANK,AXISBANK,KOTAKBANK,SBIN")
        """
        if symbols.strip():
            syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        elif basket in ALL_BASKETS:
            syms = list(ALL_BASKETS[basket]["symbols"])[:12]
        else:
            return dumps({"error": "pass a valid basket= (see sector list) or symbols=",
                          "got_basket": basket})
        syms = list(dict.fromkeys(syms))[:12]
        if len(syms) < 2:
            return dumps({"error": "need at least 2 symbols"})

        try:
            from statsmodels.tsa.stattools import coint
            import statsmodels.api as sm
        except Exception as e:
            return dumps({"error": f"statsmodels required: {e}"})

        try:
            close = _batch_close(syms, period)
        except Exception as e:
            return dumps({"error": f"price fetch failed: {e}"})
        resolved = [s for s in syms if s in close.columns]
        if len(resolved) < 2:
            return dumps({"error": "fewer than 2 symbols resolved", "resolved": resolved})

        results = []
        for a, b in itertools.combinations(resolved, 2):
            try:
                pair = close[[a, b]].dropna()
                if len(pair) < 60:
                    continue
                y, x = pair[a].values, pair[b].values
                _t, p, _c = coint(y, x)
                if p >= 0.05:
                    continue
                beta = float(sm.OLS(y, sm.add_constant(x)).fit().params[1])
                spread = y - beta * x
                mu, sd = float(np.mean(spread)), float(np.std(spread, ddof=1))
                if sd <= 0:
                    continue
                z = round((float(spread[-1]) - mu) / sd, 2)
                sig = "SHORT_SPREAD" if z > 2 else ("LONG_SPREAD" if z < -2 else "NEUTRAL")
                results.append({
                    "pair": f"{a}/{b}", "y": a, "x": b, "coint_p": round(float(p), 5),
                    "hedge_ratio": round(beta, 4), "zscore": z,
                    "half_life_days": _half_life(spread), "signal": sig,
                })
            except Exception:
                continue

        results.sort(key=lambda r: (r["coint_p"], -abs(r["zscore"])))
        actionable = [r for r in results if r["signal"] != "NEUTRAL"]
        return dumps({
            "universe": basket or "custom", "symbols_resolved": len(resolved),
            "pairs_tested": len(resolved) * (len(resolved) - 1) // 2,
            "cointegrated_found": len(results),
            "actionable_now": actionable[:max_pairs],
            "all_cointegrated": results[:max_pairs],
            "note": "Cointegrated = p<0.05. |z|>2 = entry, z->0 = exit. Trade y vs hedge_ratio*x. "
                    "Capped to 12 symbols; verify regime stability before sizing.",
        })
