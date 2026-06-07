"""Main entry point for the FinStack MCP server."""

import logging
import sys

from mcp.server.fastmcp import FastMCP

from finstack.config import config
# Compressed, Claude-first tool surface: a few configurable umbrella tools that
# reuse the same data layer the old ~90 narrow tools used. Legacy tool modules
# remain on disk (their data fns are reused) but are no longer registered.
from finstack.tools.quote import register_quote_tools
from finstack.tools.history import register_history_tools
from finstack.tools.funda import register_funda_tools
from finstack.tools.pulse import register_pulse_tools
from finstack.tools.corporate import register_corporate_tools
from finstack.tools.optionsx import register_optionsx_tools
from finstack.tools.analyzex import register_analyzex_tools
from finstack.tools.quant import register_quant_tools
from finstack.tools.pro import register_pro_tools
from finstack.tools.batch import register_batch_tools
from finstack.tools.tax import register_tax_tools
from finstack.tools.charts import register_charts_tools

config.setup_logging()
logger = logging.getLogger("finstack")

TOOL_CATALOG = [
    {"name": "nse_quote", "description": "Real-time NSE stock quote", "tier": "free"},
    {"name": "bse_quote", "description": "Real-time BSE stock quote", "tier": "free"},
    {"name": "nse_market_status", "description": "Market open/closed status", "tier": "free"},
    {"name": "nifty_index", "description": "Index values (Nifty, Sensex, Bank Nifty)", "tier": "free"},
    {"name": "nse_historical", "description": "Historical OHLCV data", "tier": "free"},
    {"name": "nse_top_movers", "description": "Top gainers, losers, most active", "tier": "free"},
    {"name": "mutual_fund_nav", "description": "Live NAV for any Indian mutual fund", "tier": "free"},
    {"name": "nse_circuit_breakers", "description": "Stocks hitting upper/lower circuit limits", "tier": "free"},
    {"name": "sensex_components", "description": "All stocks in Nifty 50 or Sensex with live prices", "tier": "free"},
    {"name": "nse_52week_scanner", "description": "Stocks near 52-week high or low", "tier": "free"},
    {"name": "stock_quote", "description": "Global stock quote (US, EU, Asia)", "tier": "free"},
    {"name": "stock_historical", "description": "Global historical OHLCV data", "tier": "free"},
    {"name": "crypto_price", "description": "Live crypto prices (BTC, ETH, SOL)", "tier": "free"},
    {"name": "crypto_historical", "description": "Historical crypto data", "tier": "free"},
    {"name": "forex_rate", "description": "Live forex rates (USD/INR, EUR/INR)", "tier": "free"},
    {"name": "market_news", "description": "Market news by ticker or general", "tier": "free"},
    {"name": "sec_filing", "description": "SEC filings (10-K, 10-Q, 8-K)", "tier": "free"},
    {"name": "sec_filing_search", "description": "Search SEC EDGAR for companies", "tier": "free"},
    {"name": "income_statement", "description": "Income statement / P&L", "tier": "free"},
    {"name": "balance_sheet", "description": "Balance sheet data", "tier": "free"},
    {"name": "cash_flow", "description": "Cash flow statement", "tier": "free"},
    {"name": "key_ratios", "description": "P/E, ROE, margins, debt/equity, growth", "tier": "free"},
    {"name": "company_profile", "description": "Company overview and description", "tier": "free"},
    {"name": "dividend_history", "description": "Historical dividend payments", "tier": "free"},
    {"name": "technical_indicators", "description": "RSI, MACD, SMA, Bollinger, ATR, Stochastic, ADX", "tier": "free"},
    {"name": "compare_stocks_tool", "description": "Side-by-side stock comparison (2-5 stocks)", "tier": "free"},
    {"name": "sector_performance", "description": "Nifty sectoral index performance", "tier": "free"},
    {"name": "nse_fii_dii_data", "description": "FII/DII institutional activity", "tier": "free"},
    {"name": "nse_bulk_deals", "description": "Bulk & block deals on NSE", "tier": "free"},
    {"name": "nse_corporate_actions", "description": "Dividends, splits, bonuses", "tier": "free"},
    {"name": "nse_quarterly_results", "description": "Latest quarterly financials with QoQ growth", "tier": "free"},
    {"name": "earnings_calendar", "description": "Upcoming earnings dates", "tier": "free"},
    {"name": "ipo_calendar", "description": "Upcoming & recent IPOs", "tier": "free"},
    {"name": "stock_screener", "description": "Screen stocks by P/E, ROE, market cap, sector", "tier": "pro"},
    {"name": "support_resistance", "description": "Pivot points & key price levels", "tier": "pro"},
    {"name": "nse_options_chain", "description": "Options chain with PCR analysis", "tier": "pro"},
    {"name": "portfolio_analysis", "description": "Portfolio P&L, weights, risk analysis", "tier": "pro"},
    {"name": "backtest_strategy", "description": "SMA crossover strategy backtesting", "tier": "pro"},
    {"name": "calculate_tax_liability", "description": "LTCG/STCG tax calculator for Indian equity and mutual fund trades", "tier": "free"},
    # ── Market Intelligence (Tools 41–48) — features paid platforms charge for ──
    {"name": "options_oi_analytics", "description": "Max Pain, PCR trend, IV summary, top OI strikes [Sensibull Pro ₹1,300/mo → FREE]", "tier": "free"},
    {"name": "options_greeks", "description": "Black-Scholes Greeks: Delta, Gamma, Theta, Vega, Rho [Sensibull Pro → FREE]", "tier": "free"},
    {"name": "nse_insider_trading", "description": "NSE SAST insider trading disclosures [Trendlyne ₹4,950/yr → FREE]", "tier": "free"},
    {"name": "promoter_shareholding", "description": "Promoter/FII/DII/public shareholding pattern [Screener Pro ₹4,999/yr → FREE]", "tier": "free"},
    {"name": "rbi_policy_rates", "description": "RBI repo, CRR, SLR, MSF, bank rate [Bloomberg $31,980/yr → FREE]", "tier": "free"},
    {"name": "india_macro_indicators", "description": "CPI inflation, GDP growth, current account [Bloomberg $31,980/yr → FREE]", "tier": "free"},
    {"name": "amfi_fund_flows", "description": "MF industry AUM, SIP flows, category breakdown [Morningstar $17,500/yr → FREE]", "tier": "free"},
    {"name": "india_gsec_yields", "description": "G-Sec yield curve: T-bill to 30-yr bond [Bloomberg $31,980/yr → FREE]", "tier": "free"},
    {"name": "india_vix", "description": "India VIX fear index + signal + history [Trendlyne paid → FREE]", "tier": "free"},
    {"name": "gift_nifty", "description": "GIFT Nifty pre-market + overnight global indices [Bloomberg paid → FREE]", "tier": "free"},
    {"name": "promoter_pledge", "description": "Promoter pledge % risk signal [Screener Pro ₹4,999/yr → FREE]", "tier": "free"},
    {"name": "dividend_history_deep", "description": "10-year dividend history + trailing yield [Bloomberg/FactSet paid → FREE]", "tier": "free"},
    {"name": "nifty_pcr_trend", "description": "Nifty PCR across all expiries + overall sentiment [Sensibull ₹1,300/mo → FREE]", "tier": "free"},
    # ── Broker + Credit + ESG (Tools 54–58) ──
    {"name": "live_quote", "description": "Real-time NSE quote via Angel One SmartAPI — zero delay [Zerodha ₹500/mo → FREE]", "tier": "free"},
    {"name": "market_depth", "description": "Level 2 order book top 5 bid/ask via Angel One [Zerodha ₹500/mo → FREE]", "tier": "free"},
    {"name": "broker_setup_status", "description": "Check Angel One SmartAPI integration status + setup guide", "tier": "free"},
    {"name": "credit_ratings", "description": "NSE/BSE credit ratings from SEBI filings [Bloomberg $24k/yr → FREE]", "tier": "free"},
    {"name": "brsr_esg", "description": "BRSR sustainability data from SEBI filings [Bloomberg ESG $24k/yr → FREE]", "tier": "free"},
    # ── Broker: Fyers API v3 ──
    {"name": "fyers_live_quote", "description": "Real-time NSE quote via Fyers API v3 (zero delay)", "tier": "free"},
    {"name": "fyers_candles", "description": "Historical OHLCV candles from Fyers API v3", "tier": "free"},
    {"name": "fyers_status", "description": "Fyers API configuration status + setup guide", "tier": "free"},
    # ── Broker: ICICI Breeze ──
    {"name": "icici_live_quote", "description": "Real-time NSE quote via ICICI Breeze (zero delay)", "tier": "free"},
    {"name": "icici_candles", "description": "Historical OHLCV candles from ICICI Breeze", "tier": "free"},
    {"name": "icici_status", "description": "ICICI Breeze configuration status + daily session guide", "tier": "free"},
    # ── Phase 3: Multi-agent + Intelligence ──
    {"name": "get_social_sentiment", "description": "Social sentiment for any NSE stock from Reddit + Twitter (BUY/HOLD/SELL)", "tier": "free"},
    {"name": "get_stock_brief", "description": "Multi-agent AI debate: 6 personas analyse a stock → consensus signal", "tier": "free"},
    {"name": "get_stock_debate", "description": "3-round sequential debate: agents read each other and rebut → emergent consensus with reasoning chain", "tier": "free"},
    {"name": "detect_unusual_activity", "description": "Smart money detector: OI buildup, block deals, promoter buying, volume spike", "tier": "free"},
    {"name": "get_nifty_outlook", "description": "Nifty direction probability % (RSI + FII + PCR + VIX + G-Sec + GIFT Nifty)", "tier": "free"},
    {"name": "get_fno_trade_setup", "description": "NIFTY/BANKNIFTY options setup: BUY_CE, BUY_PE, or NO_TRADE with ATM strike and approval-ready reasoning", "tier": "free"},
    # ── Phase 3: Intelligence tools ──
    {"name": "predict_earnings", "description": "AI earnings preview: beat/miss probability before quarterly results", "tier": "free"},
    {"name": "analyze_portfolio", "description": "Portfolio X-ray: P&L, XIRR, sector concentration, risk flags, diversification score", "tier": "free"},
    {"name": "get_mf_overlap", "description": "Mutual fund overlap: % common holdings between two funds (AMFI data)", "tier": "free"},
    {"name": "get_fii_retail_divergence", "description": "FII vs retail divergence signal — highest-conviction Indian market signal", "tier": "free"},
    {"name": "get_morning_brief", "description": "8:15 AM pre-market brief: GIFT Nifty + FII + top setups + direction probability", "tier": "free"},
    {"name": "get_morning_fno_brief", "description": "8:15 AM F&O brief: NIFTY/BANKNIFTY setup, GIFT Nifty, VIX, and approval-ready trade summary", "tier": "free"},
    {"name": "get_pledge_alert", "description": "Promoter pledge early warning: pledge % + QoQ change + risk level", "tier": "free"},
    {"name": "scan_pledge_risks", "description": "Scan multiple stocks for promoter pledge risk simultaneously", "tier": "free"},
    {"name": "detect_pump", "description": "Pump-and-dump detector: volume spike + circuit days + price surge", "tier": "free"},
    # ── Phase 4: Never-before-built Indian market tools ──
    {"name": "scan_watchlist", "description": "Batch-rank a watchlist by signal score for daily triage and automation", "tier": "free"},
    {"name": "get_stock_timeline", "description": "Unified stock timeline: news, results, insider, bulk deals, sentiment, pledge", "tier": "free"},
    {"name": "get_stock_signal_score", "description": "Automation-friendly stock ranking score with supports, risks, and factor breakdown", "tier": "free"},
    {"name": "get_sector_peer_context", "description": "Sector strength and peer comparison context for a stock", "tier": "free"},
    {"name": "evaluate_signal_quality", "description": "Evaluation/proof layer for the signal engine's price-action core", "tier": "free"},
    {"name": "predict_circuit", "description": "Lower circuit risk predictor: pledge + FII selling + 52W low proximity", "tier": "free"},
    {"name": "get_sebi_alerts", "description": "SEBI enforcement order tracker — early warning before regulatory crash", "tier": "free"},
    {"name": "correlate_gst_to_stocks", "description": "GST collection trend → sector stock predictor (1-3mo leading indicator)", "tier": "free"},
    {"name": "get_agm_brief", "description": "AGM/EGM unusual resolution detector: debt raise, salary hike, pledge approval", "tier": "free"},
    {"name": "get_insider_signal", "description": "SEBI SAST insider trading pattern: who is buying/selling their own stock", "tier": "free"},
    {"name": "get_telegram_tracker", "description": "Dalal Street Telegram tip channel accuracy + pump-and-dump scoring", "tier": "free"},
    {"name": "analyze_budget_live", "description": "Real-time budget speech analyzer: paste FM text → instant sector/stock signals", "tier": "free"},
    {"name": "get_budget_impact", "description": "Historical Union Budget impact by year: winners, losers, key announcements", "tier": "free"},
    # ── Signal outcome tracking (data moat) ──
    {"name": "get_signal_accuracy", "description": "Accuracy stats for FinStack signals — backed by real 7d/30d outcome data [unique to finstack-mcp]", "tier": "free"},
    {"name": "get_signal_history", "description": "View recent BUY/HOLD/SELL signals with actual 7-day returns and outcome labels", "tier": "free"},
    {"name": "check_signal_outcomes", "description": "Trigger outcome check for pending signals (runs automatically, call manually to force)", "tier": "free"},
]

TOTAL_TOOLS = len(TOOL_CATALOG) + 1

mcp = FastMCP("FinStack")

register_quote_tools(mcp)        # market_quote
register_history_tools(mcp)      # history
register_funda_tools(mcp)        # fundamentals
register_pulse_tools(mcp)        # market_pulse, screen, funds
register_corporate_tools(mcp)    # corporate_intel
register_optionsx_tools(mcp)     # options
register_analyzex_tools(mcp)     # analyze, portfolio
register_quant_tools(mcp)        # quant
register_pro_tools(mcp)          # macro, valuation, forensic_diagnostics
register_batch_tools(mcp)        # batch_analyze
register_charts_tools(mcp)       # chart_data
register_tax_tools(mcp)          # calculate_tax_liability


@mcp.tool()
def finstack_info() -> str:
    """Return basic server metadata and useful links."""
    import json

    from finstack import __version__

    # Compressed, Claude-first surface: a few configurable umbrella tools.
    tools = {
        "market_quote": "Live quotes/index/compare/support-resistance. Args: symbols (comma list), view, market.",
        "history": "Historical OHLCV. Args: symbols, period, interval, market.",
        "fundamentals": "Income/balance/cashflow/ratios/profile/dividends/technicals. Args: symbol, statements.",
        "market_pulse": "Market breadth: status/movers/circuit/fii_dii/bulk/sector/vix/gift/pcr/... Args: views.",
        "screen": "Stock screener by P/E, ROE, market cap, sector, etc.",
        "funds": "Mutual funds. Args: action=nav|overlap|flows.",
        "corporate_intel": "Insider/promoter/pledge/credit/esg/sebi/agm. Args: symbol, kinds.",
        "options": "Option chain/OI/greeks/PCR/max-pain. Args: symbol, views.",
        "analyze": "Multi-agent brief/score/timeline/divergence/pump/sentiment/earnings/etc. Args: symbol, lenses.",
        "portfolio": "Portfolio X-ray (P&L, risk, concentration). Args: holdings.",
        "quant": "Risk metrics/optimize/GARCH vol/correlation/pairs/backtest. Args: symbols, analysis.",
        "valuation": "DCF, reverse-DCF, Graham, owner-earnings, EPV (you supply inputs).",
        "forensic_diagnostics": "Beneish-M, Altman-Z, Piotroski-F, Sloan, DuPont, Merton-DD (you supply inputs).",
        "macro": "Live key-free macro (World Bank/DBnomics) with as_of/source/is_stale stamps.",
        "batch_analyze": "Run any per-symbol analysis across many tickers concurrently.",
        "calculate_tax_liability": "Indian LTCG/STCG tax on an equity/MF trade.",
        "chart_data": "Plot-ready data for interactive charts (price/candlestick/comparison/drawdown/heatmap/frontier/...).",
    }
    return json.dumps(
        {
            "name": "FinStack MCP",
            "version": __version__,
            "description": "Configurable, Claude-first financial analytics (compressed tool surface)",
            "tier": config.mode.value,
            "tools_available": len(tools),
            "tools": tools,
            "links": {
                "github": "https://github.com/finstacklabs/finstack-mcp",
                "website": "https://finstacklabs.github.io/",
                "pricing": "https://finstacklabs.github.io/#pricing",
                "docs": "https://github.com/finstacklabs/finstack-mcp#readme",
            },
            "data_sources": [
                "yfinance (NSE, BSE, US, Crypto - free, no API key)",
                "SEC EDGAR (US filings - free, no API key)",
                "CoinGecko (Crypto - free tier, 30 calls/min)",
            ],
        },
        indent=2,
    )


def main() -> None:
    """Start the MCP server using stdio or streamable HTTP transport."""
    transport = "stdio"

    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    import os

    transport = os.getenv("FINSTACK_TRANSPORT", transport)

    logger.info("Starting FinStack MCP server v%s", __import__("finstack").__version__)
    logger.info("Transport: %s", transport)
    logger.info("Mode: %s", config.mode.value)

    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    if transport in ("http", "streamable-http"):
        # FastMCP.run() no longer accepts host/port kwargs (mcp >= ~1.8);
        # configure them on the settings object before starting.
        mcp.settings.host = config.host
        mcp.settings.port = config.port

        # The MCP SDK enables DNS-rebinding protection by default, which only
        # allows localhost Host headers. Behind a public domain (Render, Railway,
        # Fly, a reverse proxy, ...) every request would otherwise be rejected
        # with "421 Invalid Host header". Let the deployer declare the host(s):
        #   FINSTACK_ALLOWED_HOSTS=example.onrender.com,api.example.com
        #   FINSTACK_ALLOWED_HOSTS=*   -> disable the check entirely
        allowed = os.getenv("FINSTACK_ALLOWED_HOSTS", "").strip()
        if allowed:
            from mcp.server.transport_security import TransportSecuritySettings

            if allowed == "*":
                mcp.settings.transport_security = TransportSecuritySettings(
                    enable_dns_rebinding_protection=False
                )
            else:
                hosts = [h.strip() for h in allowed.split(",") if h.strip()]
                origins = [f"https://{h}" for h in hosts] + [f"http://{h}" for h in hosts]
                mcp.settings.transport_security = TransportSecuritySettings(
                    enable_dns_rebinding_protection=True,
                    allowed_hosts=hosts,
                    allowed_origins=origins,
                )

        mcp.run(transport="streamable-http")
        return

    logger.error("Unknown transport: %s", transport)
    sys.exit(1)


def health_check() -> dict:
    """Return a simple health payload for uptime checks."""
    from finstack import __version__

    return {
        "status": "ok",
        "version": __version__,
        "mode": config.mode.value,
        "tools": TOTAL_TOOLS,
    }


if __name__ == "__main__":
    main()
