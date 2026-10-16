"""Shared free-tier etiquette: space requests out, honour Google's hints, fail fast when waiting cannot help."""

from __future__ import annotations

import os
import re
import time

MIN_GAP_SECONDS = float(os.environ.get("RATE_LIMIT_GAP_SECONDS", "6"))
MAX_RETRIES = int(os.environ.get("RATE_LIMIT_RETRIES", "5"))
MAX_WAIT_SECONDS = float(os.environ.get("RATE_LIMIT_MAX_WAIT_SECONDS", "30"))

_last_request = 0.0


class QuotaError(RuntimeError):
    """Google's free tier declined to serve this request.

    Typed so callers can tell "wait, or switch model" apart from a genuine bug: the
    message is already a clean sentence, which means the 429 markers that
    `is_rate_limit()` looks for are gone by then.
    """


class ConfigError(RuntimeError):
    """The key or the model name is wrong — only a settings change fixes this one."""


def describe(error: Exception, hint: str = "") -> Exception:
    """Replace a raw Google API failure with something a user can act on.

    Unknown errors pass through untouched: dressing a genuine bug up as a config
    problem is how a project ends up with a silent failure in production.
    """
    text = str(error).lower()
    which = hint or "the configured model"
    if any(mark in text for mark in ("unauthenticated", "401", "api key not valid", "api_key_invalid")):
        return ConfigError(
            "The Gemini API key was rejected. Paste the current key into the app "
            "secrets (or .env) and restart the app — a rotated key is not picked up "
            "until the container restarts."
        )
    if any(mark in text for mark in ("permission_denied", "403")):
        return ConfigError(
            "Google refused this key (403). Enable the Generative Language API for the "
            "project it belongs to, or create a fresh key at aistudio.google.com/apikey."
        )
    if any(mark in text for mark in ("404", "not found for api version", "is not found")):
        return ConfigError(
            f"{which} is not available to this key. Google retires dated snapshots for "
            "new keys — set GEMINI_CHAT_MODEL=gemini-flash-latest in the app secrets."
        )
    if any(mark in text for mark in ("invalid_argument", "400")):
        return ConfigError(
            f"{which} rejected the request as invalid (400). The usual cause is a "
            "parameter that model does not accept; try another GEMINI_CHAT_MODEL."
        )
    return error

# 429 covers quota and burst limits; 503/"high demand" is Google telling us to come back.
_TRANSIENT_MARKS = ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "high demand", "try again")

_RETRY_HINT = re.compile(r"retry(?:\s+\w+)?(?:Delay)?['\"]?\s*[:=]?\s*['\"]?([\d.]+)s", re.IGNORECASE)
_MODEL_MENTION = re.compile(r"(?:models?/|(?:error encountered with model|model: ))([\w.\-]+)", re.IGNORECASE)
_DAILY_METRIC = re.compile(r"([\w\-]*PerDay[\w\-]*)")


def is_rate_limit(error: Exception) -> bool:
    """True for anything worth retrying: per-minute bursts and temporary server overload."""
    text = str(error)
    return any(mark in text for mark in _TRANSIENT_MARKS)


def is_daily_quota(error: Exception) -> bool:
    """True when the whole day's allowance is gone, so waiting cannot help.

    Google reports per-minute and per-day exhaustion identically (429), but the
    per-day one carries a `...PerDay...` quota id. Retrying that turns a one-second
    error into a five-minute hang in the middle of a demo.
    """
    return bool(_DAILY_METRIC.search(str(error)))


def is_hard_quota(error: Exception) -> bool:
    """The 'check your plan and billing' reply — exhaustion with no retry hint attached.

    A per-minute limit always carries a hint; when none is offered the only honest
    move is to give up quickly and tell the user which quota to go look at.
    """
    text = str(error)
    return "plan and billing" in text.lower() and not _RETRY_HINT.search(text)


def quota_message(error: Exception, hint: str = "") -> str:
    """Turn a raw Google quota payload into one sentence a user can act on.

    `hint` names the model when the response omits it — embeddings errors never do,
    and "the model in use" is useless advice when the fix is to switch models.
    """
    text = str(error)
    model = _MODEL_MENTION.search(text)
    daily = _DAILY_METRIC.search(text)
    limit = re.search(r"limit: (\d+)", text)
    which = model.group(1) if model else (hint or (daily.group(1) if daily else "that model"))
    cap = f" ({limit.group(1)} per day)" if limit else ""
    if "high demand" in text.lower() or "503" in text:
        return (
            f"{which} is temporarily overloaded on Google's side (503). Nothing is "
            "wrong with the app — try again in a minute."
        )
    return (
        f"Google's free tier is not accepting more requests for {which}{cap}. "
        "This is quota, not a bug in the app: free allowances reset at midnight Pacific "
        "and per-minute limits clear within a minute. Check live usage at "
        "https://ai.dev/rate-limit, or point the app at a model that still has budget "
        "with GEMINI_CHAT_MODEL in the app secrets."
    )


def retry_delay(error: Exception, attempt: int) -> float:
    """Prefer the server's own retry hint, else exponential backoff capped at 60s."""
    match = _RETRY_HINT.search(str(error))
    if match:
        return float(match.group(1)) + 1
    return min(2 ** (attempt + 2), 60)


def throttle(gap: float = MIN_GAP_SECONDS) -> None:
    global _last_request
    wait = gap - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def with_retry(call, *, retries: int = MAX_RETRIES, gap: float = MIN_GAP_SECONDS, hint: str = ""):
    """Run `call()` under the shared throttle, retrying only what a retry can fix.

    Quota failures leave as `quota_message()` so the UI never prints a JSON blob, and
    the whole loop is capped by MAX_WAIT_SECONDS — a demo must not hang for minutes.
    """
    waited = 0.0
    for attempt in range(retries):
        throttle(gap)
        try:
            return call()
        except Exception as error:
            if is_rate_limit(error):
                if is_daily_quota(error) or (is_hard_quota(error) and attempt >= 1):
                    # Waiting cannot help a daily cap, and a hard quota rarely survives
                    # one retry: cloud bursts read exactly like exhaustion otherwise.
                    raise QuotaError(quota_message(error, hint)) from error
                if attempt == retries - 1:
                    raise QuotaError(quota_message(error, hint)) from error
                delay = retry_delay(error, attempt)
                if waited + delay > MAX_WAIT_SECONDS:
                    # Google's own hint may ask for minutes. Honouring it is how a
                    # demo stalls, so give up and let a sibling model answer instead.
                    raise QuotaError(quota_message(error, hint)) from error
                waited += delay
                time.sleep(delay)
                continue
            cleaned = describe(error, hint)
            if cleaned is error:
                raise
            raise cleaned from error
    raise RuntimeError("Unreachable: the retry loop always returns or raises.")
