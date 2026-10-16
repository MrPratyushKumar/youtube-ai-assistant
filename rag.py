"""Gemini chat model factory and the RAG answer chain."""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

import ratelimit

SYSTEM_RULES = """
You are a YouTube Video Assistant.
Use ONLY the provided transcript context.
Never invent facts that are not in the context.
If the context does not contain the answer, reply exactly:
"The answer is not available in the video transcript."
"""

# Pinned to an alias rather than a dated snapshot: Google retires those for new
# API keys, which leaves a deployed app returning 404 on every question.
DEFAULT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-flash-latest")

# RAG answers restate transcript context rather than solve problems, so the
# model's hidden reasoning is pure latency: 432 of 500 output tokens on one
# question, ~40s instead of ~2s.
DEFAULT_THINKING_BUDGET = "0"


def thinking_kwargs(model: str) -> dict:
    """`thinking_budget` only exists on thinking models — the lite ones answer 400.

    Set GEMINI_THINKING_BUDGET to an empty value to leave the parameter off.
    """
    raw = os.environ.get("GEMINI_THINKING_BUDGET", DEFAULT_THINKING_BUDGET).strip()
    if not raw or "lite" in model:
        return {}
    return {"thinking_budget": int(raw)}


def make_llm(api_key: str, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=api_key,
        temperature=temperature,
        **thinking_kwargs(DEFAULT_MODEL),
    )


# Free-tier budgets are counted per model, so a sibling that still has room can keep
# a demo alive on the day the primary model runs out.
FALLBACK_MODELS = ("gemini-flash-lite-latest", "gemini-flash-latest")

# Set by call_llm() when a fallback answered; the UI mentions it so notes written by a
# cheaper model are not mistaken for a quality regression in the primary one.
last_model = None


def candidate_models(primary: str) -> list[str]:
    models = [primary, *FALLBACK_MODELS]
    return list(dict.fromkeys(m for m in models if m))


def _rebuild(llm, model: str) -> ChatGoogleGenerativeAI:
    api_key = llm.google_api_key
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=getattr(api_key, "get_secret_value", lambda: api_key)(),
        temperature=llm.temperature,
        **thinking_kwargs(model),
    )


def response_text(content) -> str:
    """Flatten a chat response to plain text.

    Recent versions of langchain-google-genai hand back a list of content blocks
    instead of a string, which breaks every `.strip()` and `+` downstream.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)


def call_llm(llm, prompt: str) -> str:
    """One chat completion: throttled, retried, then tried on sibling models.

    A quota error on one Gemini model says nothing about the others — each carries its
    own free-tier budget — so the app keeps answering on the day the primary runs out,
    instead of dying on the first 429.
    """
    global last_model
    primary = getattr(llm, "model", DEFAULT_MODEL)
    tried: list[str] = []
    for model in candidate_models(primary):
        attempt = llm if model == primary else _rebuild(llm, model)
        try:
            answer = response_text(
                ratelimit.with_retry(
                    lambda: attempt.invoke(prompt),
                    retries=1 if tried else ratelimit.MAX_RETRIES,
                    hint=model,
                ).content
            )
        except ratelimit.QuotaError:
            tried.append(model)
            continue
        last_model = model
        return answer
    raise RuntimeError(
        f"Gemini has no free-tier capacity left for {', '.join(tried)}. "
        "Allowances reset at midnight Pacific — check https://ai.dev/rate-limit, "
        "wait for the reset, or attach a billed key to the app secrets."
    )



def retrieve(vector_store, question: str, k: int = 4):
    return vector_store.similarity_search(question, k=k)


def answer_question(llm, question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""{SYSTEM_RULES}

Each context block starts with a timestamp range taken from the video.
Quote that range when it helps the user find the moment.

Context:
{context}

Question:
{question}

Answer in clear Markdown with short bullet points where useful.
"""
    return call_llm(llm, prompt)
