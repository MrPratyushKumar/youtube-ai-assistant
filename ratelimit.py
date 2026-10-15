"""Shared free-tier etiquette: space requests out and honour Google's retryDelay."""

from __future__ import annotations

import os
import re
import time

MIN_GAP_SECONDS = float(os.environ.get("RATE_LIMIT_GAP_SECONDS", "6"))
MAX_RETRIES = int(os.environ.get("RATE_LIMIT_RETRIES", "5"))

_last_request = 0.0


def is_rate_limit(error: Exception) -> bool:
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def retry_delay(error: Exception, attempt: int) -> float:
    """Prefer the server's own retry hint, else exponential backoff capped at 60s."""
    match = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 1
    return min(2 ** (attempt + 2), 60)


def throttle(gap: float = MIN_GAP_SECONDS) -> None:
    global _last_request
    wait = gap - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def with_retry(call, *, retries: int = MAX_RETRIES, gap: float = MIN_GAP_SECONDS):
    """Run `call()` under the shared throttle, retrying only on rate limits."""
    for attempt in range(retries):
        throttle(gap)
        try:
            return call()
        except Exception as error:
            if not is_rate_limit(error) or attempt == retries - 1:
                raise
            time.sleep(retry_delay(error, attempt))
    raise RuntimeError("Unreachable: the retry loop always returns or raises.")
