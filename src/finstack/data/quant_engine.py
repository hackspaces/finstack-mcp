"""
FinStack Quant Engine

Pure computation functions for quantitative analytics on Indian (NSE) equities.
No MCP dependency — every function fetches prices via yfinance and returns a plain
dict. Built on numpy, pandas, scipy, statsmodels, and arch.

Design principles:
  - NSE symbols: a plain symbol like "RELIANCE" is mapped to "RELIANCE.NS".
    Index symbols (starting with "^") are left untouched.
  - FAIL LOUD: when price data is missing or insufficient, return {"error": ...}.
    Never fabricate numbers.
  - Every function is wrapped in try/except returning {"error": ...}.
  - numpy scalar types are converted to native python floats before returning.

Functions:
  - compute_risk_metrics  - annualized return/vol, Sharpe, Sortino, max drawdown,
                            VaR/CVaR, beta & alpha vs a benchmark
  - optimize_portfolio    - long-only mean-variance optimization (scipy.optimize)
  - forecast_volatility   - GARCH(1,1) volatility forecast (arch)
  - correlation_matrix    - correlation matrix + average pairwise correlation
  - pairs_cointegration   - cointegration test, hedge ratio, spread z-score, signal
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_nse(symbol: str) -> str:
    """Map a plain symbol to its NSE yfinance ticker (append .NS).

    Index symbols (starting with '^') and symbols that already carry a suffix
    (e.g. '.NS', '.BO') are returned unchanged.
    """
    symbol = symbol.strip().upper()
    if symbol.startswith("^"):
        return symbol
    if "." in symbol:
        return symbol
    return f"{symbol}.NS"


def _fetch_close(symbol: str, period: str) -> pd.Series:
    """Fetch the adjusted close price series for a symbol via yfinance.

    Raises ValueError if no data comes back (FAIL LOUD).
    """
    import yfinance as yf

    ticker = _to_nse(symbol)
    hist = yf.Ticker(ticker).history(period=period)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(f"No price data for '{symbol}' ({ticker}) over period '{period}'.")
    close = hist["Close"].dropna()
    if close.empty:
        raise ValueError(f"No valid close prices for '{symbol}' ({ticker}).")
    return close


def _daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns from a close-price series."""
    return close.pct_change().dropna()


def _f(value) -> float:
    """Convert a numpy/pandas scalar to a native python float."""
    return float(value)


def compute_risk_metrics(
    symbol: str,
    benchmark: str = "^NSEI",
    period: str = "1y",
    rf: float = 0.065,
) -> dict:
    """Compute a full risk profile for a symbol vs a benchmark.

    Returns annualized return & volatility, Sharpe and Sortino ratios, max
    drawdown, historical VaR(95%) and CVaR(95%), and beta & alpha vs the
    benchmark (OLS regression of asset returns on benchmark returns).

    Args:
        symbol: NSE symbol (e.g. "RELIANCE"); ".NS" is appended automatically.
        benchmark: benchmark ticker, default Nifty 50 ("^NSEI").
        period: yfinance period string (e.g. "1y", "2y").
        rf: annual risk-free rate (default 0.065 = 6.5%).

    Returns:
        dict of metrics, or {"error": ...} on failure.
    """
    try:
        close = _fetch_close(symbol, period)
        rets = _daily_returns(close)
        if len(rets) < 30:
            return {"error": f"Insufficient data for '{symbol}': only {len(rets)} return observations (need >= 30)."}

        ann_return = _f(rets.mean() * TRADING_DAYS)
        ann_vol = _f(rets.std(ddof=1) * np.sqrt(TRADING_DAYS))

        sharpe = _f((ann_return - rf) / ann_vol) if ann_vol > 0 else None

        downside = rets[rets < 0]
        downside_dev = _f(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
        sortino = _f((ann_return - rf) / downside_dev) if downside_dev > 0 else None

        # Max drawdown from the cumulative wealth curve.
        cum = (1.0 + rets).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_drawdown = _f(drawdown.min())

        # Historical VaR / CVaR at 95% (1-day, expressed as positive loss).
        var_95 = _f(-np.percentile(rets, 5))
        tail = rets[rets <= -var_95]
        cvar_95 = _f(-tail.mean()) if len(tail) > 0 else var_95

        # Beta & alpha vs benchmark via OLS regression of aligned daily returns.
        beta = None
        alpha = None
        try:
            from scipy import stats

            bench_close = _fetch_close(benchmark, period)
            bench_rets = _daily_returns(bench_close)
            aligned = pd.concat([rets, bench_rets], axis=1, join="inner").dropna()
            aligned.columns = ["asset", "bench"]
            if len(aligned) >= 30:
                slope, intercept, r_value, _p, _se = stats.linregress(
                    aligned["bench"].values, aligned["asset"].values
                )
                beta = _f(slope)
                # Annualized Jensen's alpha.
                alpha = _f(intercept * TRADING_DAYS)
        except Exception as be:  # benchmark is best-effort; report it but keep metrics
            beta = None
            alpha = None
            beta_error = str(be)
        else:
            beta_error = None

        result = {
            "symbol": symbol.upper(),
            "benchmark": benchmark,
            "period": period,
            "observations": int(len(rets)),
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "var_95_daily": var_95,
            "cvar_95_daily": cvar_95,
            "beta": beta,
            "alpha_annual": alpha,
            "risk_free_rate": rf,
        }
        if beta is None and beta_error:
            result["beta_note"] = f"Could not compute beta/alpha vs {benchmark}: {beta_error}"
        return result
    except Exception as e:
        return {"error": str(e)}


def optimize_portfolio(
    symbols: list,
    objective: str = "max_sharpe",
    period: str = "1y",
    rf: float = 0.065,
) -> dict:
    """Long-only mean-variance portfolio optimization.

    Uses scipy.optimize.minimize with weights summing to 1 and bounds 0..1
    (long-only, no leverage). Supports two objectives:
      - "max_sharpe": maximize the Sharpe ratio
      - "min_vol":    minimize annualized volatility

    Deliberately avoids cvxpy / PyPortfolioOpt to keep the dependency footprint
    small for a 512MB host.

    Args:
        symbols: list of NSE symbols (e.g. ["RELIANCE", "TCS", "HDFCBANK"]).
        objective: "max_sharpe" or "min_vol".
        period: yfinance period string.
        rf: annual risk-free rate (default 0.065).

    Returns:
        dict with optimal weights, expected annual return, annual vol, Sharpe;
        or {"error": ...} on failure.
    """
    try:
        from scipy.optimize import minimize

        symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        if len(symbols) < 2:
            return {"error": "Portfolio optimization needs at least 2 symbols."}
        if objective not in ("max_sharpe", "min_vol"):
            return {"error": f"Unknown objective '{objective}'. Use 'max_sharpe' or 'min_vol'."}

        # Build an aligned matrix of daily returns.
        series = {}
        for sym in symbols:
            close = _fetch_close(sym, period)
            series[sym] = _daily_returns(close)
        rets_df = pd.DataFrame(series).dropna()
        if len(rets_df) < 30:
            return {"error": f"Insufficient overlapping data: only {len(rets_df)} aligned observations (need >= 30)."}
        if rets_df.shape[1] < 2:
            return {"error": "Fewer than 2 symbols had usable data."}

        used_symbols = list(rets_df.columns)
        n = len(used_symbols)
        mean_daily = rets_df.mean().values
        cov_daily = rets_df.cov().values
        ann_mean = mean_daily * TRADING_DAYS
        ann_cov = cov_daily * TRADING_DAYS

        def port_perf(weights):
            w = np.asarray(weights)
            ret = float(w @ ann_mean)
            vol = float(np.sqrt(w @ ann_cov @ w))
            return ret, vol

        def neg_sharpe(weights):
            ret, vol = port_perf(weights)
            if vol <= 0:
                return 1e9
            return -(ret - rf) / vol

        def port_vol(weights):
            _ret, vol = port_perf(weights)
            return vol

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        x0 = np.array([1.0 / n] * n)

        obj_fn = neg_sharpe if objective == "max_sharpe" else port_vol
        opt = minimize(
            obj_fn,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if not opt.success:
            return {"error": f"Optimization did not converge: {opt.message}"}

        weights = np.clip(opt.x, 0.0, 1.0)
        total = weights.sum()
        if total <= 0:
            return {"error": "Optimization produced degenerate weights."}
        weights = weights / total

        exp_ret, exp_vol = port_perf(weights)
        sharpe = _f((exp_ret - rf) / exp_vol) if exp_vol > 0 else None

        weight_dict = {sym: _f(round(w, 6)) for sym, w in zip(used_symbols, weights)}

        return {
            "symbols": used_symbols,
            "objective": objective,
            "period": period,
            "observations": int(len(rets_df)),
            "weights": weight_dict,
            "expected_annual_return": _f(exp_ret),
            "annual_volatility": _f(exp_vol),
            "sharpe_ratio": sharpe,
            "risk_free_rate": rf,
        }
    except Exception as e:
        return {"error": str(e)}


def forecast_volatility(symbol: str, horizon: int = 5, period: str = "2y") -> dict:
    """Forecast volatility with a GARCH(1,1) model (arch package).

    Fits GARCH(1,1) on daily percent returns and reports the current annualized
    volatility plus the annualized volatility forecast `horizon` trading days out.

    Args:
        symbol: NSE symbol (e.g. "RELIANCE").
        horizon: forecast horizon in trading days (default 5).
        period: yfinance period string (default "2y").

    Returns:
        dict with current and forecast annualized volatility, or {"error": ...}.
    """
    try:
        from arch import arch_model

        if horizon < 1:
            return {"error": "Horizon must be >= 1 trading day."}

        close = _fetch_close(symbol, period)
        # arch works best on percent returns (scaled by 100).
        pct_returns = close.pct_change().dropna() * 100.0
        if len(pct_returns) < 100:
            return {"error": f"Insufficient data for GARCH on '{symbol}': only {len(pct_returns)} observations (need >= 100)."}

        am = arch_model(pct_returns, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
        res = am.fit(disp="off")

        # Conditional volatility is in percent (daily). Annualize and de-scale (/100).
        current_daily_vol = _f(res.conditional_volatility[-1]) / 100.0
        current_annual_vol = _f(current_daily_vol * np.sqrt(TRADING_DAYS))

        fc = res.forecast(horizon=horizon, reindex=False)
        # Variance forecast for the last row; take the horizon-th day.
        var_row = np.asarray(fc.variance.values[-1])
        horizon_daily_var = _f(var_row[horizon - 1]) / (100.0 ** 2)
        horizon_daily_vol = float(np.sqrt(horizon_daily_var))
        horizon_annual_vol = _f(horizon_daily_vol * np.sqrt(TRADING_DAYS))

        return {
            "symbol": symbol.upper(),
            "model": "GARCH(1,1)",
            "period": period,
            "observations": int(len(pct_returns)),
            "horizon_days": int(horizon),
            "current_annual_volatility": current_annual_vol,
            "forecast_annual_volatility": horizon_annual_vol,
        }
    except Exception as e:
        return {"error": str(e)}


def correlation_matrix(symbols: list, period: str = "1y") -> dict:
    """Compute the return-correlation matrix for a basket of symbols.

    Returns the full correlation matrix as a nested dict, the average pairwise
    correlation, and a simple diversification note.

    Args:
        symbols: list of NSE symbols (e.g. ["RELIANCE", "TCS", "HDFCBANK"]).
        period: yfinance period string.

    Returns:
        dict with correlation matrix, average pairwise correlation and a note,
        or {"error": ...} on failure.
    """
    try:
        symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        if len(symbols) < 2:
            return {"error": "Correlation needs at least 2 symbols."}

        series = {}
        for sym in symbols:
            close = _fetch_close(sym, period)
            series[sym] = _daily_returns(close)
        rets_df = pd.DataFrame(series).dropna()
        if len(rets_df) < 30:
            return {"error": f"Insufficient overlapping data: only {len(rets_df)} aligned observations (need >= 30)."}
        if rets_df.shape[1] < 2:
            return {"error": "Fewer than 2 symbols had usable data."}

        corr = rets_df.corr()
        used = list(corr.columns)
        matrix = {
            row: {col: _f(round(corr.loc[row, col], 4)) for col in used}
            for row in used
        }

        # Average of the strictly-upper-triangle (unique pairwise) correlations.
        n = len(used)
        pair_vals = [corr.iloc[i, j] for i in range(n) for j in range(i + 1, n)]
        avg_corr = _f(np.mean(pair_vals)) if pair_vals else 0.0

        if avg_corr >= 0.7:
            note = "High average correlation — limited diversification benefit; holdings move together."
        elif avg_corr >= 0.4:
            note = "Moderate correlation — some diversification, but meaningful shared risk remains."
        else:
            note = "Low average correlation — good diversification across these holdings."

        return {
            "symbols": used,
            "period": period,
            "observations": int(len(rets_df)),
            "correlation_matrix": matrix,
            "average_pairwise_correlation": _f(round(avg_corr, 4)),
            "diversification_note": note,
        }
    except Exception as e:
        return {"error": str(e)}


def pairs_cointegration(symbol1: str, symbol2: str, period: str = "2y") -> dict:
    """Test two symbols for cointegration and derive a pairs-trading signal.

    Runs the statsmodels Engle-Granger cointegration test (p-value), estimates
    the OLS hedge ratio (symbol1 ~ symbol2), computes the current spread z-score,
    and emits a signal (LONG_SPREAD / SHORT_SPREAD / NEUTRAL). The pair is flagged
    `cointegrated` when p < 0.05.

    Args:
        symbol1: first NSE symbol (the dependent leg).
        symbol2: second NSE symbol (the hedge leg).
        period: yfinance period string (default "2y").

    Returns:
        dict with p-value, hedge ratio, spread z-score, signal and a
        `cointegrated` boolean, or {"error": ...} on failure.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.tsa.stattools import coint

        close1 = _fetch_close(symbol1, period)
        close2 = _fetch_close(symbol2, period)
        aligned = pd.concat([close1, close2], axis=1, join="inner").dropna()
        aligned.columns = ["y", "x"]
        if len(aligned) < 60:
            return {"error": f"Insufficient overlapping data: only {len(aligned)} aligned observations (need >= 60)."}

        y = aligned["y"].values
        x = aligned["x"].values

        # Engle-Granger cointegration test.
        _t_stat, p_value, _crit = coint(y, x)
        cointegrated = bool(p_value < 0.05)

        # OLS hedge ratio: y = alpha + beta * x.
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        intercept = _f(model.params[0])
        hedge_ratio = _f(model.params[1])

        # Spread and its z-score.
        spread = aligned["y"] - (intercept + hedge_ratio * aligned["x"])
        spread_mean = _f(spread.mean())
        spread_std = _f(spread.std(ddof=1))
        if spread_std <= 0:
            return {"error": "Spread has zero variance; cannot compute z-score."}
        current_spread = _f(spread.iloc[-1])
        z_score = _f((current_spread - spread_mean) / spread_std)

        # Signal: spread = y - beta*x. High z => y rich vs x => short the spread.
        if z_score > 2.0:
            signal = "SHORT_SPREAD"
        elif z_score < -2.0:
            signal = "LONG_SPREAD"
        else:
            signal = "NEUTRAL"

        return {
            "symbol1": symbol1.upper(),
            "symbol2": symbol2.upper(),
            "period": period,
            "observations": int(len(aligned)),
            "coint_p_value": _f(round(p_value, 6)),
            "cointegrated": cointegrated,
            "hedge_ratio": _f(round(hedge_ratio, 6)),
            "intercept": _f(round(intercept, 6)),
            "current_spread": _f(round(current_spread, 6)),
            "spread_zscore": _f(round(z_score, 4)),
            "signal": signal,
            "signal_note": (
                "Signal only actionable when cointegrated (p < 0.05). "
                "|z| > 2 indicates a mean-reversion entry; z reverting toward 0 is the exit."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Forecasting (probabilistic — distributions/intervals, NOT point predictions) ──

def _hurst(ts: np.ndarray) -> float | None:
    """Hurst exponent via rescaled-range across lags. <0.5 mean-reverting, >0.5 trending."""
    try:
        ts = np.asarray(ts, dtype=float)
        lags = range(2, min(40, len(ts) // 2))
        tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
        tau = [t for t in tau if t > 0]
        if len(tau) < 5:
            return None
        poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
        return round(float(poly[0]), 3)
    except Exception:
        return None


def monte_carlo(symbol: str, horizon: int = 21, sims: int = 5000, period: str = "3y") -> dict:
    """GBM Monte-Carlo price distribution `horizon` trading days out."""
    close = _fetch_close(symbol, period)
    rets = _daily_returns(close)
    if len(rets) < 60:
        return {"error": f"insufficient history for {symbol}"}
    mu, sd, s0 = float(rets.mean()), float(rets.std()), float(close.iloc[-1])
    rng = np.random.default_rng(7)
    drift = mu - 0.5 * sd * sd
    shocks = rng.normal(drift, sd, size=(int(sims), int(horizon)))
    final = s0 * np.exp(shocks.sum(axis=1))
    pc = lambda p: round(float(np.percentile(final, p)), 2)
    return {
        "model": "monte_carlo_gbm", "symbol": symbol.upper(), "current_price": round(s0, 2),
        "horizon_days": int(horizon), "simulations": int(sims),
        "expected_price": round(float(final.mean()), 2),
        "expected_return_pct": round(float((final.mean() / s0 - 1) * 100), 2),
        "percentiles": {"p5": pc(5), "p25": pc(25), "p50": pc(50), "p75": pc(75), "p95": pc(95)},
        "prob_above_current_pct": round(float((final > s0).mean() * 100), 1),
        "daily_drift": round(mu, 5), "daily_vol": round(sd, 5),
        "disclaimer": "Probability distribution from historical drift/vol — NOT a prediction.",
    }


def mean_reversion(symbol: str, period: str = "2y") -> dict:
    """AR(1) half-life + Hurst + z-score: is the stock mean-reverting, and where in the range?"""
    close = _fetch_close(symbol, period)
    if len(close) < 60:
        return {"error": f"insufficient history for {symbol}"}
    lp = np.log(close.values.astype(float))
    b, _a = np.polyfit(lp[:-1], lp[1:], 1)
    half_life = round(float(-np.log(2) / np.log(b)), 1) if 0 < b < 1 else None
    m = float(close.rolling(50).mean().iloc[-1]); sd = float(close.rolling(50).std().iloc[-1])
    z = round((float(close.iloc[-1]) - m) / sd, 2) if sd else None
    h = _hurst(lp)
    regime = "mean_reverting" if (h is not None and h < 0.45) else ("trending" if h is not None and h > 0.55 else "random_walk")
    sig = "NEUTRAL"
    if z is not None and regime == "mean_reverting":
        sig = "BUY_DIP" if z < -1.5 else ("SELL_RIP" if z > 1.5 else "NEUTRAL")
    return {
        "model": "mean_reversion", "symbol": symbol.upper(), "current_price": round(float(close.iloc[-1]), 2),
        "hurst_exponent": h, "regime": regime, "half_life_days": half_life,
        "zscore_50d": z, "fair_value_50d": round(m, 2), "signal": sig,
        "note": "Hurst<0.45 mean-reverting, >0.55 trending. Half-life = days to revert halfway.",
    }


def drift_forecast(symbol: str, horizon: int = 252, period: str = "3y") -> dict:
    """Annualized drift + vol projected to a 90% expected price band `horizon` days out."""
    close = _fetch_close(symbol, period)
    rets = _daily_returns(close)
    if len(rets) < 60:
        return {"error": f"insufficient history for {symbol}"}
    mu = float(rets.mean()) * 252
    sd = float(rets.std()) * np.sqrt(252)
    s0 = float(close.iloc[-1]); h = int(horizon) / 252
    exp_ret = mu * h
    half = 1.645 * sd * np.sqrt(h)  # 90% CI
    return {
        "model": "drift_projection", "symbol": symbol.upper(), "current_price": round(s0, 2),
        "horizon_days": int(horizon), "annualized_drift_pct": round(mu * 100, 2),
        "annualized_vol_pct": round(sd * 100, 2),
        "expected_return_pct": round(exp_ret * 100, 2),
        "expected_price": round(s0 * (1 + exp_ret), 2),
        "ci90_price": [round(s0 * (1 + exp_ret - half), 2), round(s0 * (1 + exp_ret + half), 2)],
        "disclaimer": "Drift extrapolation with a 90% band — assumes past drift/vol persist; NOT a guarantee.",
    }
