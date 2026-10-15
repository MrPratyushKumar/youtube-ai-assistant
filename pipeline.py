"""Transcript fetching, timestamped chunking and vector-store building."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi

OEMBED_URL = "https://www.youtube.com/oembed"
_NOISE = re.compile(r"\[Music\]|\[Applause\]|\[\s*\]", re.IGNORECASE)
_TIMESTAMP = re.compile(r"^\[(\d+)-(\d+)\]\s*(\S+)\s*-\s*(\S+)")


@dataclass
class Snippet:
    start: int
    text: str


def extract_video_id(url: str) -> str | None:
    if re.fullmatch(r"[\w-]{11}", url):
        return url

    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0] or None

    parsed = urlparse(url)
    if parsed.hostname not in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        return None

    parts = parsed.path.split("/")
    if parsed.path.startswith("/embed/") or parsed.path.startswith("/shorts/"):
        return parts[2] if len(parts) > 2 and parts[2] else None

    return parse_qs(parsed.query).get("v", [None])[0]


def format_time(seconds: int) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def timestamp_link(video_id: str, seconds: int) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={max(0, int(seconds))}s"


def fetch_video_title(video_id: str) -> str:
    """Best-effort title lookup; falls back to the video id."""
    try:
        response = requests.get(
            OEMBED_URL,
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        if response.ok:
            return json.loads(response.text).get("title") or f"Video {video_id}"
    except Exception:
        pass
    return f"Video {video_id}"


def fetch_snippets(video_id: str, languages: list[str] | None = None) -> list[Snippet]:
    """Return cleaned transcript snippets ordered by start time."""
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    snippets: list[Snippet] = []
    for item in transcript:
        text = _NOISE.sub(" ", item.text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            snippets.append(Snippet(start=int(item.start), text=text))
    snippets.sort(key=lambda s: s.start)
    return snippets


def chunk_snippets(
    snippets: list[Snippet], chunk_chars: int = 1200, overlap: int = 3
) -> list[list[Snippet]]:
    """Group caption snippets into overlapping windows of roughly `chunk_chars`."""
    if not snippets:
        return []

    chunks: list[list[Snippet]] = []
    start_idx = 0
    total = len(snippets)

    while start_idx < total:
        end_idx = start_idx
        size = 0
        while end_idx < total:
            size += len(snippets[end_idx].text) + 1
            end_idx += 1
            if size >= chunk_chars:
                break
        chunks.append(snippets[start_idx:end_idx])
        if end_idx >= total:
            break
        start_idx = max(end_idx - overlap, start_idx + 1)

    return chunks


def chunk_to_text(batch: list[Snippet], batch_index: int) -> str:
    """Timestamped chunk text; the header survives retrieval for citations."""
    header = f"[{batch_index}-{len(batch) - 1}] {format_time(batch[0].start)} - {format_time(batch[-1].start)}"
    body = " ".join(s.text for s in batch)
    return f"{header}\n{body}"


def chunk_label(chunk_text: str) -> tuple[int, int, str]:
    """Return (start_index, end_index, 'mm:ss - mm:ss') for a chunk header line."""
    first_line = chunk_text.splitlines()[0] if chunk_text else ""
    match = _TIMESTAMP.match(first_line)
    if not match:
        return 0, 0, ""
    return int(match.group(1)), int(match.group(2)), f"{match.group(3)} - {match.group(4)}"


def build_vector_store(chunks: list[str], api_key: str):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=api_key
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings)
