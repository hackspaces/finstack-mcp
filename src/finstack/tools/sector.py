"""
FinStack Sector / Thematic tool — one configurable, Claude-first tool over
~120 baskets (NSE official sectors + curated niche themes) and the full ~2.1k
NSE universe. Supports ad-hoc/combinatorial baskets (any ticker list).
"""

import json
from finstack.utils.respond import dumps as _dumps

from finstack.data import sector_engine as se


def register_sector_tools(mcp):
    """Register the sector/thematic basket tool."""

    @mcp.tool()
    def sector(action: str = "list", basket: str = "", baskets: str = "",
               symbols: str = "", combine: str = "", lookback: str = "3m",
               query: str = "", limit: int = 30) -> str:
        """Sector & thematic-basket analytics over ~120 indices + the full NSE universe.

        Baskets include NSE official sectors (nse_*) AND curated niche themes
        (pipes_plumbing, hvac_cooling, fluorochemicals_refrigerants, defense,
        railways, cdmo_cram, gold_financiers, ...). You can also analyze ANY
        custom or combinatorial basket by passing a ticker list.

        Args:
            action: one of —
                - list         : all basket names + categories + sizes (+ universe size)
                - performance  : equal-weight returns (1d..1y), breadth, RS vs Nifty,
                                 constituent leaders/laggards. Use basket= OR symbols=
                                 (comma list, ad-hoc) OR combine= (comma basket names).
                - rotation     : rank baskets by momentum over `lookback` (sector
                                 rotation leaders/laggards). Optional baskets= (comma
                                 list); defaults to the 22 NSE sectors.
                - constituents : list a basket's members (basket=).
                - compare      : side-by-side performance of several baskets (baskets=).
                - search       : find tickers in the ~2.1k NSE universe (query=).
            basket: a basket name (performance / constituents).
            baskets: comma-separated basket names (rotation / compare).
            symbols: comma-separated tickers for an ad-hoc basket (performance).
            combine: comma-separated basket names to merge into one (performance).
            lookback: rotation window — 1w/1m/3m/6m/1y.
            query: substring to search the universe (search).
            limit: max search results.

        Examples:
            sector(action="list")
            sector(action="performance", basket="fluorochemicals_refrigerants")
            sector(action="performance", symbols="VOLTAS,BLUESTARCO,AMBER,SYMPHONY")
            sector(action="rotation", lookback="3m")
            sector(action="compare", baskets="defense,railways,pipes_plumbing")
            sector(action="search", query="pipe")
        """
        a = action.strip().lower()
        try:
            if a == "list":
                return _dumps(se.list_baskets(), indent=2, default=str)
            if a == "performance":
                return _dumps(se.basket_performance(basket or None, symbols or None, combine or None),
                                  indent=2, default=str)
            if a == "rotation":
                bl = [b.strip() for b in baskets.split(",") if b.strip()] or None
                return _dumps(se.rotation(bl, lookback), indent=2, default=str)
            if a == "constituents":
                return _dumps(se.constituents(basket), indent=2, default=str)
            if a == "compare":
                names = [b.strip() for b in baskets.split(",") if b.strip()]
                if not names:
                    return _dumps({"error": "compare needs baskets= (comma list)"}, indent=2)
                return _dumps({"compare": [se.basket_performance(basket=n) for n in names]},
                                  indent=2, default=str)
            if a == "search":
                return _dumps(se.search_universe(query, limit), indent=2, default=str)
            return _dumps({"error": f"unknown action '{action}'",
                               "valid_actions": ["list", "performance", "rotation",
                                                 "constituents", "compare", "search"]}, indent=2)
        except Exception as e:
            return _dumps({"error": f"{type(e).__name__}: {e}", "action": a}, indent=2)
