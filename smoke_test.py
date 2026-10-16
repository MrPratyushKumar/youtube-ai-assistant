"""Smoke test for the whole app: no API key needed, and the one network probe self-skips."""

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
    except ratelimit.QuotaError as fast:
        assert "gemini-3.8-flash" in str(fast) and "20 per day" in str(fast), fast
    assert time.monotonic() - started < 1, "must not sit in the retry loop"

    # Captured verbatim from a deployed run: a hard quota with no retryDelay attached.
    hard = Exception(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
        "current quota, please check your plan and billing details. For more information "
        "on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To "
        "monitor your current usage, head to: https://ai.dev/rate-limit. ', 'status': "
        "'RESOURCE_EXHAUSTED', 'details': [{'@type': "
        "'type.googleapis.com/google.rpc.Help'}]}}"
    )
    assert ratelimit.is_rate_limit(hard) and ratelimit.is_hard_quota(hard)
    assert not ratelimit.is_daily_quota(hard)

    attempts = {"n": 0}

    def billed():
        attempts["n"] += 1
        raise hard

    started = time.monotonic()
    try:
        ratelimit.with_retry(billed, gap=0)
        raise AssertionError("a hard quota must not consume the whole retry budget")
    except ratelimit.QuotaError as clean:
        assert attempts["n"] == 2, attempts  # one proving retry, then stop
        assert "{'error'" not in str(clean) and "ai.dev/rate-limit" in str(clean), clean
        assert "that model" in str(clean)
    assert time.monotonic() - started < 20, "bounded waiting"

    def hinted():
        raise Exception(str(hard) + " model: gemini-embedding-001")

    try:
        ratelimit.with_retry(hinted, retries=1, gap=0, hint="gemini-embedding-001")
        raise AssertionError("should have raised")
    except ratelimit.QuotaError as named:
        assert "gemini-embedding-001" in str(named), named

    # 503 is Google's own "come back later", so it belongs in the retry loop.
    overload = {"n": 0}

    def busy():
        overload["n"] += 1
        if overload["n"] < 2:
            raise Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return "served"

    assert ratelimit.is_rate_limit(Exception("503 UNAVAILABLE"))
    assert ratelimit.with_retry(busy, gap=0) == "served"
    assert overload["n"] == 2

    # A rejected key or a retired model is a settings problem, not a quota story.
    assert "API key was rejected" in str(ratelimit.describe(Exception("401 UNAUTHENTICATED"), "gemini-flash-latest"))
    assert "Generative Language API" in str(ratelimit.describe(Exception("403 PERMISSION_DENIED"), ""))
    assert isinstance(ratelimit.describe(Exception("404 models/gemini-2.5-flash is not found"), "x"), ratelimit.ConfigError)
    assert isinstance(ratelimit.describe(ValueError("boom"), ""), ValueError)

    def rejected():
        raise Exception("401 UNAUTHENTICATED. API key not valid.")

    try:
        ratelimit.with_retry(rejected, gap=0)
        raise AssertionError("should have raised")
    except ratelimit.ConfigError as why:
        assert "restart" in str(why), why
    print("rate-limit handling ok (retryDelay, 503 retries, quota and config errors all read as sentences)")


def test_model_fallback():
    """A quota error on one Gemini model must not end the demo — siblings have their own budget."""
    import ratelimit
    import rag

    assert rag.candidate_models("gemini-flash-lite-latest") == [
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
    ]
    assert rag.candidate_models("gemini-custom") == ["gemini-custom", "gemini-flash-lite-latest", "gemini-flash-latest"]

    quota = Exception(
        "429 RESOURCE_EXHAUSTED. You exceeded your current quota, check your plan and billing details."
    )

    class Flaky:
        model = "gemini-flash-latest"
        temperature = 0.2
        google_api_key = "fake"

        def __init__(self):
            self.seen = []

        def invoke(self, prompt):
            self.seen.append(self.model)
            if self.model == "gemini-flash-latest":
                raise ratelimit.QuotaError("quota gone")
            return type("R", (), {"content": [{"type": "text", "text": "from lite"}]})()

    llm = Flaky()

    def fake_rebuild(source, model):
        replacement = Flaky()
        replacement.model = model
        replacement.seen = llm.seen
        return replacement

    original, original_last = rag._rebuild, rag.last_model
    rag._rebuild = fake_rebuild
    try:
        assert rag.call_llm(llm, "question") == "from lite"
        assert rag.last_model == "gemini-flash-lite-latest", rag.last_model
        assert llm.seen == ["gemini-flash-latest", "gemini-flash-lite-latest"], llm.seen

        class Broken(Flaky):
            model = "gemini-flash-lite-latest"

            def invoke(self, prompt):
                raise ValueError("some unexpected state")

        try:
            rag.call_llm(Broken(), "question")
            raise AssertionError("a real bug must not be reported as a quota problem")
        except ValueError:
            pass
    finally:
        rag._rebuild = original
        rag.last_model = original_last
    print(f"model fallback ok ({llm.seen})")


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
        return type("R", (), {"content": (
            "# Notes\n\n## 2. Key Takeaways\n- **[0:21]** a point\n\n"
            "## 4. Concepts\nTerm | Meaning\n--- | ---\n**REST** | an *architecture*\n"
        )})()


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
**Embedding** | A *numeric vector* representing meaning

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
    # Lite models answer 400 INVALID_ARGUMENT for this parameter, so it must be omitted.
    assert rag.thinking_kwargs("gemini-flash-lite-latest") == {}
    assert rag.thinking_kwargs("gemini-flash-latest") == {"thinking_budget": 0}
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


def test_pdf_font_fallback():
    """A cloud container may have no TTF at all — that must degrade, not abort."""
    import pdf_export

    saved_index = pdf_export._index
    saved_download = pdf_export._download_font
    regular = pdf_export._find_font("")
    assert regular, "needs a real TTF on the test machine"
    body = "**bold point** and *emphasis* and [0:12](https://youtu.be/x?t=12)"
    try:
        pdf_export._index = {}  # nothing installed
        pdf_export._download_font = lambda style: None  # and no egress either
        assert pdf_export._resolve_family() == pdf_export._CORE_FAMILY
        assert pdf_export.notes_to_pdf(body, "T", "https://youtu.be/x")[:5] == b"%PDF-"
        assert pdf_export.notes_to_pdf("café — ₹ → ✅", "T", "")[:5] == b"%PDF-"

        pdf_export._download_font = lambda style: regular  # downloaded Noto Sans
        assert pdf_export._resolve_family() == "Notes"
        assert pdf_export.notes_to_pdf("café — ₹ → ✅", "T", "")[:5] == b"%PDF-"

        pdf_export._download_font = saved_download
        all_names = [n for names in pdf_export._FONT_CANDIDATES.values() for n in names]
        pdf_export._index = {name.lower(): regular for name in all_names}  # bold/italic files missing
        assert pdf_export._resolve_family() == "Notes"
        assert pdf_export.notes_to_pdf(body, "T", "https://youtu.be/x")[:5] == b"%PDF-"
        print("pdf font fallback ok (no fonts -> core family, downloaded font -> Notes, missing bold -> single TTF)")
    finally:
        pdf_export._index = saved_index
        pdf_export._download_font = saved_download


def test_font_download():
    """The Noto Sans URLs the cloud fallback depends on must still serve real TTFs."""
    import pdf_export

    paths = {style: pdf_export._download_font(style) for style in pdf_export._NOTO_FILES}
    if not any(paths.values()):  # offline run — nothing to check
        print("font download SKIPPED (no network)")
        return
    for style, path in paths.items():
        assert path and os.path.isfile(path), f"{style or 'regular'} font download failed"
        with open(path, "rb") as handle:
            assert handle.read(4) in pdf_export._TTF_MAGICS, path
    assert pdf_export._download_font("") == paths[""]  # second call reads the cache
    print(f"font download ok (4 TTFs cached in {os.path.dirname(paths[''])})")



def test_pdf_table_cells():
    """fpdf2 aborts on emphasis in a table cell and loses links to nested tags."""
    import pdf_export

    markdown = (
        "## Glossary\n| Term | When |\n|---|---|\n"
        "| **REST** | **[`0:21`](https://youtu.be/x?t=21)** |\n\n"
        "## Points\n- **[`2:40`](https://youtu.be/x?t=160)** body link\n"
    )
    html = pdf_export._to_html(markdown, "helvetica")
    table = html[html.index("<table>") : html.index("</table>")]
    assert "<td>REST</td>" in table and "<td>0:21</td>" in table, table
    assert "<strong" not in table and "<a " not in table, table
    assert '<a href="https://youtu.be/x?t=160"' in html  # links outside tables stay links
    assert pdf_export.notes_to_pdf(markdown, "T", "")[:5] == b"%PDF-"
    print("pdf table cells ok (bold/link cells flatten to text, export survives)")


def test_app_flow():
    """Headless click through app.py: a one-chunk upload must reach a downloadable PDF."""
    import rag
    from streamlit.testing.v1 import AppTest

    os.environ["GEMINI_API_KEY"] = "smoke-test-key-not-used"

    class StubEmbeddings:
        def __init__(self, api_key):
            pass

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [0.9, 0.1, 0.0]

    pipeline.GeminiBatchEmbeddings = StubEmbeddings
    rag.make_llm = lambda api_key, temperature=0.2: StubLLM()

    def click(at, label):
        labels = [button.label for button in at.button]
        assert label in labels, labels
        at.button[labels.index(label)].click()
        at.run()

    at = AppTest.from_file(os.path.join(os.path.dirname(__file__) or ".", "app.py"), default_timeout=120)
    at.run()
    at.radio[0].set_value("📄 Paste / upload transcript")
    at.run()
    at.text_area[0].set_value("0:00\nAPIs are messengers.\n0:07\nThey hide complexity.\n")
    at.run()
    click(at, "🚀 Process video")
    assert not at.exception, at.exception
    assert [s.value for s in at.success][0].endswith("(1 chunks)"), [s.value for s in at.success]
    click(at, "📝 Generate notes")
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    downloads = [button.label for button in at.download_button]
    assert any("PDF" in label for label in downloads), downloads
    # Button labels carry emoji, which Windows' default console code page cannot print.
    print(f"app flow ok (1-chunk upload -> notes -> {len(downloads)} PDF download button)")


test_video_ids()
test_chunking()
test_upload_parsing()
test_rate_limit()
test_model_fallback()
test_llm_response_shapes()
test_vector_store()
test_notes_flow()
test_pdf_export()
test_pdf_font_fallback()
test_pdf_table_cells()
test_font_download()
test_app_flow()
print("\nALL CHECKS PASSED")
