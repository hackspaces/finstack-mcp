"""
FinStack Batch Tools

A single tool that runs any supported per-symbol analysis over many tickers in
ONE call, executed concurrently. Lets an assistant pass a whole watchlist
instead of calling a single-symbol tool N times.

Each operation maps to an existing per-symbol function (each takes `symbol` as
its first argument). Failures are isolated per symbol so one bad ticker never
fails the whole batch.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from finstack.data.nse import get_nse_quote
from finstack.data.fundamentals import get_key_ratios
from finstack.data.research import get_stock_signal_score, get_stock_timeline
from finstack.data.pump_detector import detect_pump
from finstack.data.promoter_watch import get_pledge_alert
from finstack.data.insider_pattern import get_insider_signal
from finstack.data.sentiment import get_social_sentiment
from finstack.data.market_intelligence import get_promoter_shareholding
from finstack.data.agents import get_stock_brief
from finstack.data.earnings import predict_earnings
from finstack.data.quant_engine import compute_risk_metrics

# operation name -> callable(symbol) -> dict
BATCH_OPS = {
    "quote": get_nse_quote,
    "key_ratios": get_key_ratios,
    "signal_score": get_stock_signal_score,
    "timeline": get_stock_timeline,
    "pump": detect_pump,
    "pledge": get_pledge_alert,
    "insider": get_insider_signal,
    "sentiment": get_social_sentiment,
    "promoter": get_promoter_shareholding,
    "brief": get_stock_brief,
    "earnings": predict_earnings,
    "risk_metrics": compute_risk_metrics,
}

MAX_SYMBOLS = 25


def register_batch_tools(mcp):
    """Register batch (multi-ticker) tools with the MCP server."""

    @mcp.tool()
    def batch_analyze(symbols: str, analysis: str = "quote") -> str:
        """Run one analysis across MANY tickers in a single concurrent call.

        Pass a whole watchlist instead of calling a single-symbol tool repeatedly.
        Each ticker is fetched in parallel and failures are isolated per symbol.

        Args:
            symbols: comma-separated NSE symbols, e.g. "RELIANCE,TCS,HDFCBANK,INFY".
                     Up to 25 per call; extras are ignored (reported in `dropped`).
            analysis: which per-symbol analysis to run for every ticker. One of:
                - quote         live NSE quote (price, 52w, PE, market cap)
                - key_ratios    valuation & profitability ratios
                - signal_score  0-100 composite BUY/HOLD/SELL score
                - timeline      recent events (news, results, insider, deals)
                - pump          pump-and-dump risk
                - pledge        promoter pledge early warning
                - insider       SEBI SAST insider buy/sell pattern
                - sentiment     social sentiment (Reddit/Twitter)
                - promoter      shareholding pattern
                - brief         6-agent BUY/HOLD/SELL consensus
                - earnings      earnings beat/miss preview
                - risk_metrics  Sharpe/Sortino/VaR/beta vs Nifty

        Returns:
            JSON string: {analysis, count, results: {symbol: <result>}, ...}.
            A ticker that errors gets {"error": "..."} under its key.

        Examples:
            batch_analyze(symbols="RELIANCE,TCS,HDFCBANK", analysis="quote")
            batch_analyze(symbols="ADANIENT,JPPOWER,RPOWER", analysis="pump")
            batch_analyze(symbols="RELIANCE,TCS,INFY,ITC", analysis="risk_metrics")
        """
        op = analysis.strip().lower()
        if op not in BATCH_OPS:
            return json.dumps({
                "error": f"Unknown analysis '{analysis}'.",
                "valid_analyses": sorted(BATCH_OPS.keys()),
            }, indent=2)

        raw = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        # de-duplicate while preserving order
        seen: set[str] = set()
        tickers = [s for s in raw if not (s in seen or seen.add(s))]
        dropped = tickers[MAX_SYMBOLS:]
        tickers = tickers[:MAX_SYMBOLS]

        if not tickers:
            return json.dumps({"error": "No symbols provided."}, indent=2)

        fn = BATCH_OPS[op]
        results: dict[str, object] = {}

        def _run(sym: str):
            try:
                return sym, fn(sym)
            except Exception as e:  # isolate per-symbol failures
                return sym, {"error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(max_workers=min(8, len(tickers))) as pool:
            futures = [pool.submit(_run, s) for s in tickers]
            for fut in as_completed(futures):
                sym, res = fut.result()
                results[sym] = res

        # keep output ordered the way the caller listed them
        ordered = {s: results[s] for s in tickers}
        out = {
            "analysis": op,
            "count": len(ordered),
            "results": ordered,
        }
        if dropped:
            out["dropped"] = dropped
            out["note"] = f"Capped at {MAX_SYMBOLS} symbols per call; {len(dropped)} ignored."
        return json.dumps(out, indent=2, default=str)
