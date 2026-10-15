"""Transcript fetching, timestamped chunking and vector-store building."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import requests
from langchain_core.embeddings import Embeddings
from youtube_transcript_api import YouTubeTranscriptApi

import ratelimit

OEMBED_URL = "https://www.youtube.com/oembed"
EMBED_GAP_SECONDS = float(os.environ.get("RATE_LIMIT_EMBED_GAP_SECONDS", "0.8"))
_NOISE = re.compile(r"\[Music\]|\[Applause\]|\[\s*\]", re.IGNORECASE)
_TIMESTAMP = re.compile(r"^\[(\d+)-(\d+)\]\s*(\S+)\s*-\s*(\S+)")


@dataclass
class Snippet:
    start: int | None
    text: str


class TranscriptBlockedError(RuntimeError):
    """Raised when YouTube refuses this IP (normal for cloud hosts)."""


def _transcript_api():
    """Build the client, routing through a proxy when one is configured.

    YouTube blocks datacenter IPs, so a deployed app needs `TRANSCRIPT_PROXY_HTTPS_URL`
    (a residential proxy from Webshare/Bright Data etc.) to fetch transcripts at all.
    """
    https_url = os.environ.get("TRANSCRIPT_PROXY_HTTPS_URL")
    http_url = os.environ.get("TRANSCRIPT_PROXY_HTTP_URL", https_url)
    if not https_url:
        return YouTubeTranscriptApi()

    from youtube_transcript_api.proxies import GenericProxyConfig

    return YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(http_url=http_url, https_url=https_url)
    )


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


def format_time(seconds: int | None) -> str:
    if seconds is None:
        return ""
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
    from youtube_transcript_api import IpBlocked, RequestBlocked

    try:
        transcript = _transcript_api().fetch(video_id, languages or ["en"])
    except (IpBlocked, RequestBlocked) as blocked:
        raise TranscriptBlockedError(
            "YouTube blocked this server's IP, which is normal for cloud hosts. "
            "Either set TRANSCRIPT_PROXY_HTTPS_URL in the app secrets, or switch to "
            "'Paste / upload transcript' and feed the app a .srt you downloaded yourself."
        ) from blocked
    return _clean_snippets((item.start for item in transcript), (item.text for item in transcript))


def _clean_snippets(starts, texts) -> list[Snippet]:
    snippets: list[Snippet] = []
    for start, raw in zip(starts, texts):
        text = re.sub(r"<[^>]+>", " ", raw)  # VTT timing tags and <c> styling runs
        text = re.sub(r"\s+", " ", _NOISE.sub(" ", text)).strip()
        if text:
            snippets.append(Snippet(start=None if start is None else int(start), text=text))
    snippets.sort(key=lambda s: s.start if s.start is not None else 0)
    return snippets


def _dedupe_rolling(snippets: list[Snippet]) -> list[Snippet]:
    """Collapse YouTube's rolling captions, where each cue repeats the previous line.

    Auto-generated VTT emits every cue as a sliding window: the tail of the last
    caption plus a little new text. Keeping only the words past the overlap stops
    the transcript being written out two or three times over.
    """
    kept: list[Snippet] = []
    keys: list[str] = []
    for snippet in snippets:
        words = snippet.text.split()
        word_keys = [_token_key(word) for word in words]
        size = _overlap_size(keys, word_keys) if kept else 0
        remainder = words[size:]
        if size:
            tail = words[:size]
            previous_words = kept[-1].text.split()
            merged = " ".join(previous_words[:-size] + tail)
            kept[-1] = Snippet(kept[-1].start, merged)
            keys = [_token_key(word) for word in merged.split()]
        if remainder:
            kept.append(Snippet(start=snippet.start, text=" ".join(remainder)))
            keys = [_token_key(word) for word in remainder]
    return kept


def _token_key(word: str) -> str:
    return re.sub(r"[^0-9a-z]", "", word.lower())


def _overlap_size(previous: list[str], current: list[str]) -> int:
    for size in range(min(len(previous), len(current)), 0, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0


_CUE = re.compile(
    r"^(?P<start>(?:\d{1,2}:)?\d{1,2}:\d{2})[.,]\d{3}\s*-->\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{3}"
)


def _cue_to_seconds(value: str) -> int:
    parts = [int(p) for p in value.replace(",", ".").split(":")]
    if len(parts) == 2:
        minutes, secs = parts
        return minutes * 60 + secs
    hours, minutes, secs = parts
    return hours * 3600 + minutes * 60 + secs


def parse_uploaded_transcript(raw: str) -> list[Snippet]:
    """Accept a .srt/.vtt export, the 'Show transcript' panel text, or plain text."""
    if "-->" in raw:
        return _parse_cues(raw)
    lines = raw.splitlines()
    marks = [line for line in lines if _PANEL_TIME.match(line.strip())]
    if len(marks) >= 2 and len(marks) * 4 >= sum(1 for line in lines if line.strip()):
        return _parse_panel(lines)
    return _plain_snippets(raw)


def _parse_cues(raw: str) -> list[Snippet]:
    starts: list[int | None] = []
    texts: list[str] = []
    pending_start: int | None = None
    buffer: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip().replace("\ufeff", "")
        cue = _CUE.match(stripped)
        if cue:
            pending_start = _cue_to_seconds(cue.group("start"))
            continue
        if not stripped:
            if buffer and pending_start is not None:
                starts.append(pending_start)
                texts.append(" ".join(buffer))
            buffer, pending_start = [], None
            continue
        if stripped.isdigit() or stripped.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        buffer.append(stripped)

    if buffer and pending_start is not None:
        starts.append(pending_start)
        texts.append(" ".join(buffer))

    return _dedupe_rolling(_clean_snippets(iter(starts), iter(texts)))


_PANEL_TIME = re.compile(r"^(?:\d{1,2}:)?\d{1,2}:\d{2}$")


def _parse_panel(lines: list[str]) -> list[Snippet]:
    """Parse the 'Show transcript' panel, which alternates a bare timestamp and its lines."""
    snippets: list[Snippet] = []
    pending_start: int | None = None
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _PANEL_TIME.match(stripped):
            if buffer and pending_start is not None:
                snippets.append(Snippet(start=pending_start, text=" ".join(buffer)))
            buffer = []
            pending_start = _cue_to_seconds(stripped)
            continue
        if pending_start is None:
            continue
        buffer.append(stripped)
    if buffer and pending_start is not None:
        snippets.append(Snippet(start=pending_start, text=" ".join(buffer)))
    return _clean_snippets((s.start for s in snippets), (s.text for s in snippets))


def _plain_snippets(raw: str, target_chars: int = 220) -> list[Snippet]:
    """Split timestamp-free text on sentence boundaries so chunking stays uniform."""
    text = re.sub(r"\s+", " ", raw).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    snippets: list[Snippet] = []
    buffer = ""
    for sentence in sentences:
        if not sentence:
            continue
        if buffer and len(buffer) + len(sentence) > target_chars:
            snippets.append(Snippet(start=None, text=buffer))
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        snippets.append(Snippet(start=None, text=buffer))
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
    index_range = f"[{batch_index}-{len(batch) - 1}]"
    first, last = batch[0].start, batch[-1].start
    header = index_range if first is None else f"{index_range} {format_time(first)} - {format_time(last)}"
    body = " ".join(s.text for s in batch)
    return f"{header}\n{body}"


def chunk_label(chunk_text: str) -> tuple[int, int, str]:
    """Return (start_index, end_index, 'mm:ss - mm:ss') for a chunk header line."""
    first_line = chunk_text.splitlines()[0] if chunk_text else ""
    match = _TIMESTAMP.match(first_line)
    if not match:
        return 0, 0, ""
    return int(match.group(1)), int(match.group(2)), f"{match.group(3)} - {match.group(4)}"


def build_vector_store(chunks: list[str], vectors: list[list[float]], api_key: str):
    """Assemble FAISS from pre-computed vectors, so re-runs cost zero embedding calls."""
    from langchain_community.vectorstores import FAISS

    return FAISS.from_embeddings(
        texts=chunks, embedding=GeminiBatchEmbeddings(api_key), embeddings=vectors
    )


class GeminiBatchEmbeddings(Embeddings):
    """Gemini embeddings that send many texts per request.

    LangChain's own wrapper issues one request per document, which blows through
    the free tier's 100 embed-requests-per-minute cap on any long transcript.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-001",
        batch_size: int = 90,
    ) -> None:
        from google.genai import Client

        self._client = Client(api_key=api_key)
        self._model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Embeddings allow 100 requests/min, so they need a much lighter gap than chat.
        return ratelimit.with_retry(
            lambda: self._call(texts), gap=EMBED_GAP_SECONDS
        )

    def _call(self, texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(model=self._model, contents=texts)
        return [list(item.values) for item in response.embeddings]
