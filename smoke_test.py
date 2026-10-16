"""Offline smoke test for the pure logic (no API keys, no network needed)."""

import os
import time

os.environ["RATE_LIMIT_GAP_SECONDS"] = "0"  # the stub LLM never hits a real quota
os.environ["RATE_LIMIT_EMBED_GAP_SECONDS"] = "0"

import pipeline
import notes


def test_rate_limit():
    import ratelimit

    err = Exception("429 RESOURCE_EXHAUSTED ... 'retryDelay': '41s'")
    assert ratelimit.is_rate_limit(err)
    assert ratelimit.retry_delay(err, 0) == 42
    assert not ratelimit.is_rate_limit(Exception("bad request"))
    assert ratelimit.retry_delay(Exception("bad request"), 0) == 4  # exponential fallback
    assert ratelimit.retry_delay(Exception("plain 429"), 4) == 60  # capped

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert ratelimit.with_retry(flaky) == "ok"
    assert attempts["n"] == 3

    def broken():
        raise ValueError("not a quota problem")

    try:
        ratelimit.with_retry(broken)
        raise AssertionError("should have re-raised")
    except ValueError:
        pass

    # Captured verbatim from the API: same 429, but the whole day is gone.
    daily = Exception(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
        "current quota ... * Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
        "model: gemini-3.8-flash \\nPlease retry in 53.482125167s.', 'status': "
        "'RESOURCE_EXHAUSTED', 'details': [{'quotaId': "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}}"
    )
    assert ratelimit.is_rate_limit(daily) and ratelimit.is_daily_quota(daily)
    assert not ratelimit.is_daily_quota(Exception("429 RESOURCE_EXHAUSTED limit: 10"))

    def exhausted():
        raise daily

    started = time.monotonic()
    try:
        ratelimit.with_retry(exhausted)
        raise AssertionError("daily quota should fail fast, not retry")
    except RuntimeError as fast:
        assert "gemini-3.8-flash" in str(fast) and "20/day" in str(fast), fast
    assert time.monotonic() - started < 1, "must not sit in the retry loop"
    print("rate-limit handling ok (retryDelay, 429-only retries, daily quota fails fast)")


def test_video_ids():
    cases = {
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=42": "dQw4w9WgXcQ",
        "https://m.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://example.com/watch?v=abc": None,
    }
    for url, expected in cases.items():
        assert pipeline.extract_video_id(url) == expected, url
    print("video id parsing ok")


def test_chunking():
    snippets = [pipeline.Snippet(i * 7, f"word word word line number {i} " * 4) for i in range(40)]
    chunks = pipeline.chunk_snippets(snippets, chunk_chars=400, overlap=2)
    texts = [pipeline.chunk_to_text(c, i) for i, c in enumerate(chunks)]
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) >= len(snippets)
    for i, text in enumerate(texts):
        start, end, label = pipeline.chunk_label(text)
        assert label and start == i and end == len(chunks[i]) - 1, text.splitlines()[0]
        assert notes.seconds_from_label(label) == chunks[i][0].start
    print(f"chunking ok ({len(chunks)} chunks, labels round-trip)")


class StubLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls <= 2:
            return type("R", (), {"content": "- **[0:00]** first point\n- **[0:14]** second point"})()
        return type("R", (), {"content": "# Notes\n\n## 2. Key Takeaways\n- **[0:21]** a point"})()


def test_notes_flow():
    snippets = [pipeline.Snippet(i * 7, f"line {i} " * 30) for i in range(30)]
    texts = [
        pipeline.chunk_to_text(c, i) for i, c in enumerate(pipeline.chunk_snippets(snippets, 400))
    ]
    steps = []
    llm = StubLLM()
    markdown = notes.generate_notes(
        llm, "Demo Video", texts, progress=lambda d, t, label: steps.append(label)
    )
    linked = notes.add_timestamp_links(markdown, "dQw4w9WgXcQ")
    assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=21s" in linked, linked
    assert steps and steps[-1] == "Notes ready"
    print(f"notes pipeline ok ({llm.calls} llm calls, {len(steps)} progress updates)")
    print("\n".join(linked.splitlines()[:6]))


NOTES_SAMPLE = """# Demo Video

## 2. Key Takeaways
- **[0:21]** first point
- **[1:05]** second point

## 4. Concepts & Definitions
Term | Meaning
--- | ---
Embedding | A numeric vector representing meaning

## 7. Self-Test Questions
**Q: Why overlap chunks?**
A: So an idea split across a boundary still appears intact.
"""


def test_upload_parsing():
    srt = (
        "1\n00:00:12,340 --> 00:00:15,000\nHello there\n\n"
        "2\n00:01:02,000 --> 00:01:05,000\nSecond cue\n"
    )
    cues = pipeline.parse_uploaded_transcript(srt)
    assert [c.start for c in cues] == [12, 62], cues
    assert [c.text for c in cues] == ["Hello there", "Second cue"], cues

    vtt = "WEBVTT\nKind: captions\nLanguage: en\n\n00:00:04.100 --> 00:00:07.000\nA caption\n"
    assert pipeline.parse_uploaded_transcript(vtt)[0].start == 4

    rolling = (
        "WEBVTT\n\n00:00:05.000 --> 00:00:08.000\n"
        "Well, think of an API like a waiter\n\n"
        "00:00:08.000 --> 00:00:11.000\n"
        "think of an API like a waiter at a restaurant\n\n"
        "00:00:11.000 --> 00:00:14.000\n"
        "at a restaurant. You place your order.\n"
    )
    rolled = pipeline.parse_uploaded_transcript(rolling)
    assert " ".join(c.text for c in rolled) == (
        "Well, think of an API like a waiter at a restaurant. You place your order."
    ), [c.text for c in rolled]
    assert [c.start for c in rolled] == [5, 8, 11], rolled

    panel = "0:00\nhey everyone\n0:07\nwelcome back\n0:12\ntoday we talk about APIs\n"
    stamps = pipeline.parse_uploaded_transcript(panel)
    assert [s.start for s in stamps] == [0, 7, 12], stamps
    assert stamps[-1].text == "today we talk about APIs", stamps

    plain = pipeline.parse_uploaded_transcript("First sentence here. Second one runs longer. " * 20)
    assert len(plain) > 1 and all(s.start is None for s in plain)
    texts = [
        pipeline.chunk_to_text(batch, i)
        for i, batch in enumerate(pipeline.chunk_snippets(plain, chunk_chars=300))
    ]
    assert all(pipeline.chunk_label(text)[2] == "" for text in texts)  # no fake timestamps
    print(f"upload parsing ok ({len(cues)} srt cues, {len(plain)} plain snippets, {len(texts)} chunks)")


def test_pdf_export():
    import pdf_export

    linked = notes.add_timestamp_links(NOTES_SAMPLE, "dQw4w9WgXcQ")
    data = pdf_export.notes_to_pdf(
        linked, "Demo Video", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    assert data[:5] == b"%PDF-", data[:8]
    assert len(data) > 2000
    assert not pdf_export._prepare(linked).startswith("# ")  # title moves to the page header
    assert notes.add_timestamp_links(NOTES_SAMPLE, "") == NOTES_SAMPLE  # upload mode has no video
    assert pdf_export.notes_to_pdf(linked, "Demo Video", "")[:5] == b"%PDF-"
    print(f"pdf export ok ({len(data) / 1024:.0f} KB, font={pdf_export._find_font('') or 'core latin-1'})")


def test_llm_response_shapes():
    """Current Google chat wrappers return content blocks, not a plain string."""
    import rag

    assert rag.response_text("plain") == "plain"
    blocks = [
        {"type": "text", "text": "part one "},
        {"type": "thinking", "text": "ignored"},
        {"type": "text", "text": "part two"},
    ]
    assert rag.response_text(blocks) == "part one part two"
    assert rag.response_text(None) == ""
    assert rag.DEFAULT_MODEL and "2.5" not in rag.DEFAULT_MODEL  # retired snapshots 404
    print(f"chat response parsing ok (model={rag.DEFAULT_MODEL})")


def test_vector_store():
    """FAISS must accept pre-computed vectors, otherwise re-runs pay for embeddings twice."""
    chunks = ["[0-2] 0:00 - 0:20 rest apis use http", "[3-5] 0:21 - 0:40 graphql field selection"]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store = pipeline.build_vector_store(chunks, vectors, "smoke-test-key-not-used")
    assert len(store.docstore._dict) == 2
    hit = store.similarity_search_by_vector([0.9, 0.1, 0.0], k=1)[0].page_content
    assert hit == chunks[0], hit
    print("vector store ok (2 chunks, cached vectors reused)")


test_video_ids()
test_chunking()
test_upload_parsing()
test_rate_limit()
test_llm_response_shapes()
test_vector_store()
test_notes_flow()
test_pdf_export()
print("\nALL CHECKS PASSED")
