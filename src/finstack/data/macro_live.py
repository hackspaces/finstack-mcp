"""
FinStack Live Macro (datacenter-safe, key-free)

Replaces the old hardcoded/stale macro tools with LIVE official data from
sources that do NOT IP-block cloud hosts and need NO API key:
  - World Bank Open Data API  (annual official macro)
  - DBnomics (OECD/IMF series) (rates)

Every datum is stamped with `value`, `as_of`, `source`, `source_url` and an
`is_stale` flag — so a value can never silently masquerade as current.
"""

from __future__ import annotations

from datetime import datetime

import httpx

# friendly name -> (World Bank indicator code, unit, human label)
WB_INDICATORS = {
    "cpi_inflation": ("FP.CPI.TOTL.ZG", "% YoY", "CPI inflation"),
    "gdp_growth": ("NY.GDP.MKTP.KD.ZG", "% YoY", "Real GDP growth"),
    "gdp_usd": ("NY.GDP.MKTP.CD", "USD", "GDP (current US$)"),
    "current_account_pct_gdp": ("BN.CAB.XOKA.GD.ZS", "% of GDP", "Current account balance"),
    "unemployment": ("SL.UEM.TOTL.ZS", "%", "Unemployment rate"),
    "real_interest_rate": ("FR.INR.RINR", "%", "Real interest rate"),
    "lending_rate": ("FR.INR.LEND", "%", "Bank lending rate"),
    "broad_money_growth": ("FM.LBL.BMNY.ZG", "% YoY", "Broad money growth"),
    "fdi_pct_gdp": ("BX.KLT.DINV.WD.GD.ZS", "% of GDP", "FDI net inflows"),
    "gross_capital_formation_pct": ("NE.GDI.TOTL.ZS", "% of GDP", "Gross capital formation"),
    "gni_per_capita_usd": ("NY.GNP.PCAP.CD", "USD", "GNI per capita"),
    "govt_debt_pct_gdp": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central govt debt"),
}

# friendly name -> (DBnomics series_id, unit, label)
DBNOMICS_RATES = {
    "short_term_rate": ("OECD/KEI/IR3TIB01.IND.ST.M", "%", "3-month interbank rate"),
}

DEFAULT_SET = ["cpi_inflation", "gdp_growth", "current_account_pct_gdp",
               "unemployment", "short_term_rate", "real_interest_rate"]

_STALE_AFTER_YEARS = 2


def _is_stale(as_of: str) -> bool:
    """Annual/monthly periods older than _STALE_AFTER_YEARS are flagged stale."""
    try:
        yr = int(str(as_of)[:4])
        return (datetime.now().year - yr) > _STALE_AFTER_YEARS
    except Exception:
        return False


def _wb_latest(indicator: str, country: str) -> dict:
    # mrv=N (most-recent N values) is far more reliable than mrnev across
    # indicators; pull a few and pick the latest non-null ourselves.
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    with httpx.Client(timeout=15) as c:
        r = c.get(url, params={"format": "json", "mrv": "8"})
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return {"value": None, "error": "no data"}
    for row in data[1]:  # newest first
        if row.get("value") is not None:
            return {"value": row.get("value"), "as_of": row.get("date")}
    return {"value": None, "error": "no non-null observation"}


def _dbnomics_latest(series_id: str) -> dict:
    url = "https://api.db.nomics.world/v22/series"
    with httpx.Client(timeout=15) as c:
        r = c.get(url, params={"series_ids": series_id, "observations": "1", "format": "json"})
        r.raise_for_status()
        docs = r.json().get("series", {}).get("docs", [])
    if not docs:
        return {"value": None, "error": "no data"}
    s = docs[0]
    pairs = [(p, v) for p, v in zip(s["period"], s["value"]) if v is not None]
    if not pairs:
        return {"value": None, "error": "no observations"}
    period, value = pairs[-1]
    return {"value": value, "as_of": period}


def get_macro(indicators: list | None = None, country: str = "IN") -> dict:
    """Fetch live macro indicators with provenance + freshness stamps.

    indicators: list of friendly names (see WB_INDICATORS / DBNOMICS_RATES).
                None -> a sensible default set.
    """
    wanted = indicators or DEFAULT_SET
    out: dict[str, object] = {}
    for name in wanted:
        try:
            if name in WB_INDICATORS:
                code, unit, label = WB_INDICATORS[name]
                res = _wb_latest(code, country)
                if res.get("value") is None:
                    out[name] = {"value": None, "label": label,
                                 "source": "World Bank", "error": res.get("error", "unavailable")}
                else:
                    out[name] = {
                        "value": res["value"], "as_of": res["as_of"], "unit": unit,
                        "label": label, "source": "World Bank Open Data",
                        "source_url": f"https://data.worldbank.org/indicator/{code}?locations={country}",
                        "is_stale": _is_stale(res["as_of"]),
                    }
            elif name in DBNOMICS_RATES:
                sid, unit, label = DBNOMICS_RATES[name]
                res = _dbnomics_latest(sid)
                if res.get("value") is None:
                    out[name] = {"value": None, "label": label, "source": "DBnomics",
                                 "error": res.get("error", "unavailable")}
                else:
                    out[name] = {
                        "value": res["value"], "as_of": res["as_of"], "unit": unit,
                        "label": label, "source": f"DBnomics ({sid.split('/')[0]})",
                        "source_url": f"https://db.nomics.world/{sid.rsplit('/', 1)[0]}",
                        "is_stale": _is_stale(res["as_of"]),
                    }
            else:
                out[name] = {"error": f"unknown indicator '{name}'",
                             "valid": sorted(list(WB_INDICATORS) + list(DBNOMICS_RATES))}
        except Exception as e:
            out[name] = {"value": None, "error": f"{type(e).__name__}: {e}"}

    # RBI policy/repo rate has no reliable key-free live feed via these aggregators —
    # be honest rather than ship a stale number (the bug we are fixing).
    if "policy_rate" in (indicators or []):
        out["policy_rate"] = {
            "value": None,
            "label": "RBI policy repo rate",
            "note": "No reliable key-free live feed; check RBI directly to avoid a stale value.",
            "source_url": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
        }

    return {
        "country": country,
        "fetched_at": datetime.now().isoformat(),
        "indicators": out,
        "disclaimer": "Official annual/periodic data; see each item's as_of and is_stale.",
    }
