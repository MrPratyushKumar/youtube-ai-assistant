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


def make_llm(api_key: str, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=api_key,
        temperature=temperature,
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
    """One chat completion, spaced and retried for the free tier's ~10 requests/min."""
    return response_text(ratelimit.with_retry(lambda: llm.invoke(prompt)).content)



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
