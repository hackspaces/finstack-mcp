"""
FinStack AnalyzeX Tools — one configurable per-symbol analysis tool plus a
portfolio X-ray, replacing a whole bucket of narrow single-symbol tools.

`analyze(symbol, lenses)` runs ONE or MANY analysis "lenses" over a single
ticker in one call. Each lens maps to an existing data-layer function (the same
ones the dedicated tools wrap). Lenses are isolated: one failing lens never
fails the others, and an unknown lens is reported in the result rather than
silently dropped.

`portfolio(holdings)` is the portfolio X-ray (wraps analyze_portfolio).
"""

import json
from finstack.utils.respond import dumps as _dumps

# Per-symbol data-layer functions, copied verbatim from the dedicated wrappers.
from finstack.data.agents import get_stock_brief, get_stock_debate
from finstack.data.research import (
    get_stock_signal_score,
    get_stock_timeline,
    get_sector_peer_context,
)
from finstack.data.divergence import get_fii_retail_divergence
from finstack.data.pump_detector import detect_pump
from finstack.data.sentiment import get_social_sentiment
from finstack.data.earnings import predict_earnings
from finstack.data.smart_money import detect_unusual_activity
from finstack.data.probability import get_nifty_outlook, get_fno_trade_setup
from finstack.data.circuit import predict_circuit
from finstack.data.portfolio import analyze_portfolio


# lens name -> callable(symbol) -> dict. Each lambda calls the real function the
# way its original tool wrapper did (matching default args).
ANALYZE_LENSES = {
    "brief": lambda symbol: get_stock_brief(symbol=symbol),
    "debate": lambda symbol: get_stock_debate(symbol=symbol),
    "score": lambda symbol: get_stock_signal_score(symbol),
    "timeline": lambda symbol: get_stock_timeline(symbol, max_events=12),
    "divergence": lambda symbol: get_fii_retail_divergence(symbol),
    "pump": lambda symbol: detect_pump(symbol),
    "sentiment": lambda symbol: get_social_sentiment(symbol=symbol, limit=100),
    "earnings": lambda symbol: predict_earnings(symbol),
    "smart_money": lambda symbol: detect_unusual_activity(symbol=symbol),
    "outlook": lambda symbol: get_nifty_outlook(),          # index-wide, ignores symbol
    "fno_setup": lambda symbol: get_fno_trade_setup(symbol),
    "peer": lambda symbol: get_sector_peer_context(symbol),
    "predict_circuit": lambda symbol: predict_circuit(symbol),
}


def register_analyzex_tools(mcp):
    """Register the configurable analyze / portfolio tools with the MCP server."""

    @mcp.tool()
    def analyze(symbol: str, lenses: str = "brief") -> str:
        """Run one or MANY analysis lenses over a single NSE stock in one call.

        Instead of calling a dozen narrow single-symbol tools, pass a
        comma-separated list of lenses and get them all back in one JSON object.
        Each lens is run independently — a failing lens returns
        {"error": "..."} under its key without breaking the others.

        Args:
            symbol: NSE stock symbol (e.g. RELIANCE, TCS, HDFCBANK).
            lenses: comma-separated lens names (default "brief"). One or more of:
                - brief            6-agent BUY/HOLD/SELL consensus (parallel)
                - debate           3-round sequential agent debate
                - score            0-100 composite signal score (BUY/HOLD/SELL)
                - timeline         recent events (news, results, insider, deals)
                - divergence       FII vs retail divergence signal
                - pump             pump-and-dump operator risk
                - sentiment        social sentiment (Reddit/Twitter/StockTwits)
                - earnings         earnings beat/miss preview
                - smart_money      smart-money / unusual activity detector
                - outlook          Nifty next-session up-probability (index-wide,
                                   ignores `symbol`)
                - fno_setup        NIFTY/BANKNIFTY options setup (pass NIFTY or
                                   BANKNIFTY as `symbol`)
                - peer             sector / peer valuation context
                - predict_circuit  lower-circuit risk

        Returns:
            JSON string: {symbol, lenses, count, results: {lens: <result>}, ...}.
            A lens that errors gets {"error": "..."} under its key. Unknown lens
            names are reported under `unknown_lenses` and `valid_lenses`.

        Examples:
            analyze(symbol="RELIANCE", lenses="brief,score,sentiment")
            analyze(symbol="ADANIENT", lenses="pump,predict_circuit,divergence")
            analyze(symbol="NIFTY", lenses="outlook,fno_setup")
        """
        requested = [s.strip().lower() for s in lenses.split(",") if s.strip()]
        if not requested:
            return _dumps({
                "error": "No lenses provided.",
                "valid_lenses": sorted(ANALYZE_LENSES.keys()),
            }, indent=2)

        # de-duplicate while preserving order
        seen: set[str] = set()
        ordered_lenses = [l for l in requested if not (l in seen or seen.add(l))]

        valid = [l for l in ordered_lenses if l in ANALYZE_LENSES]
        unknown = [l for l in ordered_lenses if l not in ANALYZE_LENSES]

        if not valid:
            return _dumps({
                "error": f"No valid lenses in {lenses!r}.",
                "unknown_lenses": unknown,
                "valid_lenses": sorted(ANALYZE_LENSES.keys()),
            }, indent=2)

        results: dict[str, object] = {}
        for lens in valid:
            try:
                results[lens] = ANALYZE_LENSES[lens](symbol)
            except Exception as e:  # isolate per-lens failures
                results[lens] = {"error": f"{type(e).__name__}: {e}"}

        out: dict[str, object] = {
            "symbol": symbol,
            "lenses": valid,
            "count": len(valid),
            "results": results,
        }
        if unknown:
            out["unknown_lenses"] = unknown
            out["valid_lenses"] = sorted(ANALYZE_LENSES.keys())
        return _dumps(out, indent=2, default=str)

    @mcp.tool()
    def portfolio(holdings: list) -> str:
        """Portfolio X-ray: deep risk + return analysis for your holdings.

        Input format — list of holdings:
          [
            {"symbol": "RELIANCE", "qty": 10, "avg_price": 2400, "buy_date": "2024-01-15"},
            {"symbol": "TCS",      "qty": 5,  "avg_price": 3800}
          ]

        Returns:
          - total invested, current value, P&L, P&L %
          - XIRR (if buy_date provided)
          - per-holding breakdown with sector
          - sector concentration % (flags if > 40% in one sector)
          - risk flags: pledged promoters, single stock > 30%, FII reducing
          - diversification score (0-100)

        Args:
            holdings: list of {symbol, qty, avg_price, buy_date (optional)}
        """
        if not holdings:
            return _dumps({"error": "No holdings provided."}, indent=2)
        return _dumps(analyze_portfolio(holdings), indent=2, default=str)
