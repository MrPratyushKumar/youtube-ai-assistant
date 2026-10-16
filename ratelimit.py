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


def is_daily_quota(error: Exception) -> bool:
    """True when the whole day's allowance is gone, so waiting cannot help.

    Google reports per-minute and per-day exhaustion identically (429), but the
    per-day one carries a `...PerDay...` quota id. Retrying that turns a one-second
    error into a five-minute hang in the middle of a demo.
    """
    return "PerDay" in str(error)


def daily_quota_message(error: Exception) -> str:
    model = re.search(r"model: ([\w.\-]+)", str(error))
    limit = re.search(r"limit: (\d+)", str(error))
    which = model.group(1) if model else "the current model"
    cap = f" ({limit.group(1)}/day)" if limit else ""
    return (
        f"Google's free tier has used up today's requests for {which}{cap}. "
        "The allowance resets daily. Wait for the reset, or set "
        "GEMINI_CHAT_MODEL=gemini-flash-lite-latest in the app secrets to use a "
        "model with a larger free quota."
    )


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
            if is_daily_quota(error):
                raise RuntimeError(daily_quota_message(error)) from error
            if not is_rate_limit(error) or attempt == retries - 1:
                raise
            time.sleep(retry_delay(error, attempt))
    raise RuntimeError("Unreachable: the retry loop always returns or raises.")
