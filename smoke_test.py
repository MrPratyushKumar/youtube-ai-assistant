"""Offline smoke test for the pure logic (no API keys, no network needed)."""

import os

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
    print("rate-limit handling ok (parses retryDelay, retries 429 only)")


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


def test_pdf_export():
    import pdf_export

    linked = notes.add_timestamp_links(NOTES_SAMPLE, "dQw4w9WgXcQ")
    data = pdf_export.notes_to_pdf(
        linked, "Demo Video", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    assert data[:5] == b"%PDF-", data[:8]
    assert len(data) > 2000
    assert not pdf_export._prepare(linked).startswith("# ")  # title moves to the page header
    print(f"pdf export ok ({len(data) / 1024:.0f} KB, font={pdf_export._find_font('') or 'core latin-1'})")


test_video_ids()
test_chunking()
test_rate_limit()
test_notes_flow()
test_pdf_export()
print("\nALL CHECKS PASSED")
