"""
Context-engineering response layer.

One place that makes every tool's JSON cheap for an LLM to read without losing
signal:
  - rounds floats to 4 dp (kills 14-digit yfinance noise / false precision)
  - drops None / NaN keys (omit, don't emit "field":null)
  - compact separators (no indent whitespace)
  - ensure_ascii=False so ₹/€ are 1 char, not \\uXXXX

Use `dumps(obj)` as a drop-in for `json.dumps(obj, ...)` (extra kwargs ignored).
`downsample(seq, n)` evenly thins long series for chart/history payloads.
"""

import json
import math


def _slim(x):
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 4)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            sv = _slim(v)
            if sv is not None:
                out[k] = sv
        return out
    if isinstance(x, list):
        return [_slim(v) for v in x]
    return x


def dumps(obj, **_kwargs) -> str:
    """Compact, slimmed JSON. Drop-in for json.dumps (indent/default ignored)."""
    return json.dumps(_slim(obj), separators=(",", ":"), default=str, ensure_ascii=False)


def downsample(seq, n: int):
    """Evenly sample a list down to ~n points (keeps first & last shape)."""
    if not isinstance(seq, list) or n <= 0 or len(seq) <= n:
        return seq
    step = len(seq) / n
    out = [seq[int(i * step)] for i in range(n)]
    if out and out[-1] is not seq[-1]:
        out[-1] = seq[-1]
    return out
