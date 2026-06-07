"""
FinStack Sector / Thematic Basket Engine

Computes on-the-fly "indices" for ~120 curated + NSE-official sector baskets
(and ANY custom/combinatorial ticker list) from yfinance prices — equal-weight
returns across timeframes, breadth, relative strength vs Nifty, and constituent
leaders/laggards. Uses ONE batched yf.download per call so large baskets stay
fast. Full ~2.1k NSE universe is searchable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from finstack.data.baskets import BASKETS
from finstack.data.baskets_auto import NSE_SECTOR_BASKETS
from finstack.data.universe import UNIVERSE

warnings.filterwarnings("ignore")

# merged basket registry: curated niche + NSE official sectors
ALL_BASKETS: dict[str, dict] = {**NSE_SECTOR_BASKETS, **BASKETS}

_WINDOWS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 250}
_MAX_FETCH = 60          # cap constituents fetched per basket (sampled if larger)
_BENCH = "^NSEI"


def list_baskets() -> dict:
    out = []
    for name, b in sorted(ALL_BASKETS.items()):
        out.append({"name": name, "category": b.get("category", ""),
                    "constituents": len(b["symbols"])})
    return {"count": len(out), "universe_size": len(UNIVERSE), "baskets": out}


def _resolve(basket: str | None, symbols: str | None, combine: str | None) -> tuple[list[str], str, dict]:
    """Return (symbol_list, label, meta) from a basket name, custom symbols, or combined baskets."""
    note = {}
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        return syms, "custom", note
    if combine:
        names = [n.strip() for n in combine.split(",") if n.strip()]
        merged: list[str] = []
        missing = []
        for n in names:
            if n in ALL_BASKETS:
                merged += ALL_BASKETS[n]["symbols"]
            else:
                missing.append(n)
        if missing:
            note["unknown_baskets"] = missing
        return sorted(set(merged)), "+".join(names), note
    if basket and basket in ALL_BASKETS:
        return list(ALL_BASKETS[basket]["symbols"]), basket, note
    return [], basket or "", {"error": f"unknown basket '{basket}'"}


def _batch_close(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    import yfinance as yf
    ns = [f"{s}.NS" if not (s.startswith("^") or s.endswith((".NS", ".BO"))) else s for s in symbols]
    df = yf.download(ns, period=period, progress=False, threads=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    # map columns back to bare symbols
    rename = {f"{s}.NS" if not (s.startswith("^") or s.endswith((".NS", ".BO"))) else s: s for s in symbols}
    close = close.rename(columns=rename)
    keep = [s for s in symbols if s in close.columns and close[s].notna().any()]
    return close[keep].dropna(how="all")


def _win_return(series: pd.Series, ndays: int) -> float | None:
    s = series.dropna()
    if len(s) < ndays + 1:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - ndays] - 1.0) * 100.0


def basket_performance(basket: str | None = None, symbols: str | None = None,
                       combine: str | None = None) -> dict:
    syms, label, meta = _resolve(basket, symbols, combine)
    if not syms:
        return {"error": meta.get("error", "no symbols"), "hint": "pass basket=, symbols=, or combine="}
    sampled = False
    if len(syms) > _MAX_FETCH:
        syms = syms[:_MAX_FETCH]
        sampled = True

    close = _batch_close(syms + [_BENCH], "1y")
    resolved = [s for s in syms if s in close.columns]
    if not resolved:
        return {"error": "no price data resolved for this basket", "requested": len(syms)}

    # equal-weight basket return per window = mean of per-stock window returns
    perf = {}
    for w, nd in _WINDOWS.items():
        rs = [_win_return(close[s], nd) for s in resolved]
        rs = [r for r in rs if r is not None]
        perf[w] = round(float(np.mean(rs)), 2) if rs else None

    # relative strength vs Nifty (3m)
    rs_vs_nifty = None
    if _BENCH in close.columns and perf.get("3m") is not None:
        bench_3m = _win_return(close[_BENCH], _WINDOWS["3m"])
        if bench_3m is not None:
            rs_vs_nifty = round(perf["3m"] - bench_3m, 2)

    # breadth (% positive over 1m) + constituent leaders/laggards (1m)
    contrib = []
    for s in resolved:
        r1m = _win_return(close[s], _WINDOWS["1m"])
        if r1m is not None:
            contrib.append({"symbol": s, "ret_1m": round(r1m, 2), "name": UNIVERSE.get(s, "")})
    contrib.sort(key=lambda x: x["ret_1m"], reverse=True)
    up = sum(1 for c in contrib if c["ret_1m"] > 0)
    breadth = round(100 * up / len(contrib), 1) if contrib else None

    out = {
        "basket": label, "category": ALL_BASKETS.get(label, {}).get("category", "custom"),
        "constituents_resolved": len(resolved), "constituents_requested": len(syms),
        "equal_weight_return_pct": perf,
        "breadth_pct_up_1m": breadth,
        "relative_strength_vs_nifty_3m": rs_vs_nifty,
        "leaders_1m": contrib[:3], "laggards_1m": contrib[-3:][::-1],
    }
    if sampled:
        out["note"] = f"basket capped to first {_MAX_FETCH} constituents for speed"
    if meta.get("unknown_baskets"):
        out["unknown_baskets"] = meta["unknown_baskets"]
    return out


def rotation(baskets: list[str] | None = None, lookback: str = "3m", cap: int = 15) -> dict:
    """Rank baskets by equal-weight momentum over `lookback` (sector rotation)."""
    names = baskets or list(NSE_SECTOR_BASKETS.keys())  # default: the 22 NSE sectors
    nd = _WINDOWS.get(lookback, 63)
    # gather a capped sample of each basket, fetch ALL in one batched call
    sample = {}
    allsyms = set()
    for n in names:
        if n in ALL_BASKETS:
            syms = ALL_BASKETS[n]["symbols"][:cap]
            sample[n] = syms
            allsyms.update(syms)
    if not allsyms:
        return {"error": "no valid baskets", "hint": "see action=list"}
    close = _batch_close(sorted(allsyms) + [_BENCH], "1y")
    bench = _win_return(close[_BENCH], nd) if _BENCH in close.columns else None
    ranked = []
    for n, syms in sample.items():
        rs = [_win_return(close[s], nd) for s in syms if s in close.columns]
        rs = [r for r in rs if r is not None]
        if rs:
            ret = round(float(np.mean(rs)), 2)
            ranked.append({"basket": n, f"return_{lookback}": ret,
                           "rs_vs_nifty": round(ret - bench, 2) if bench is not None else None,
                           "n": len(rs)})
    ranked.sort(key=lambda x: x[f"return_{lookback}"], reverse=True)
    return {
        "lookback": lookback, "nifty_return": round(bench, 2) if bench is not None else None,
        "baskets_ranked": len(ranked),
        "leaders": ranked[:5], "laggards": ranked[-5:][::-1], "all": ranked,
        "note": f"each basket sampled to first {cap} constituents",
    }


def constituents(basket: str) -> dict:
    if basket not in ALL_BASKETS:
        return {"error": f"unknown basket '{basket}'", "hint": "see action=list"}
    b = ALL_BASKETS[basket]
    return {"basket": basket, "category": b.get("category", ""), "industry": b.get("industry"),
            "count": len(b["symbols"]),
            "constituents": [{"symbol": s, "name": UNIVERSE.get(s, "")} for s in b["symbols"]]}


def search_universe(query: str, limit: int = 30) -> dict:
    q = query.strip().lower()
    hits = [{"symbol": s, "name": n} for s, n in UNIVERSE.items()
            if q in s.lower() or q in n.lower()]
    return {"query": query, "universe_size": len(UNIVERSE), "matches": len(hits),
            "results": hits[:limit]}
