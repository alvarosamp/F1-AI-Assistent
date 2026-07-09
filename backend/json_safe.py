"""JSON-safe serialization helpers.

FastF1 telemetry/summary computations can legitimately produce NaN or
Infinity (e.g. mean() over an all-NaN column). Starlette's default
JSONResponse uses allow_nan=False, so any such value crashes the request
with "ValueError: Out of range float values are not JSON compliant".
"""

from __future__ import annotations

import math
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj
