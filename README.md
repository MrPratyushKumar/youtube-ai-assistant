# 🎥 YouTube AI Assistant — Chat + Auto Notes (RAG)

Ask a YouTube video questions, and turn any video into downloadable, timestamped study notes.
Transcript-based Retrieval-Augmented Generation with LangChain, Gemini embeddings, FAISS and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![LangChain](https://img.shields.io/badge/RAG-LangChain-green) ![Gemini](https://img.shields.io/badge/LLM-Gemini--Flash-blueviolet) ![FAISS](https://img.shields.io/badge/Vector__DB-FAISS-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

**🔴 Live demo:** <https://youtube-ai-assistant-dxevhsgtntnggmg4aibzee.streamlit.app/>
(Cloud IPs are blocked by YouTube, so on the deployed app use **📄 Paste / upload transcript** — the
*YouTube blocks cloud IPs* note under **☁️ Deploy** explains why and what to do.)

> Extended from the original *YouTube Video Chatbot* by
> [Harsha](https://youtube-ai-assistant-1926.streamlit.app/), which only answered questions.
> **What I added:** timestamped chunks and citations, the Auto Notes pipeline, PDF export,
> quota-safe batched embeddings, vector caching, chat history, multi-language transcripts,
> offline tests and the module structure.

---

## 🎯 30-second tour

1. Paste a 5–15 minute talk → **🚀 Process video**. (On the deployed app, YouTube may block the server's IP — switch to **📄 Paste / upload transcript** and feed it a `.srt` instead.)
2. **💬 Chat** → ask *"What are the three main points?"* → the answer ends with a `[mm:ss]` link that jumps into the video.
3. **📝 Auto notes** → **Generate notes** → TL;DR, topic sections, definitions table, action items and a self-test quiz.
4. **📄 Download notes (PDF)** → open it; every timestamp is a clickable deep link.
5. Re-process the same video → instant, because the transcript and vectors are cached.

---

## 🚀 Features

| Feature | Status |
|---|---|
| Paste any YouTube URL / short link / Shorts / embed / raw video ID | ✅ |
| Transcript extraction (`youtube-transcript-api`), language selectable | ✅ |
| Timestamp-aware chunking (overlapping windows of caption snippets) | ✅ |
| Gemini embeddings stored in a FAISS vector index | ✅ |
| Similarity search with `k` relevant chunks | ✅ |
| Grounded Q&A with Gemini Flash | ✅ |
| 💬 Chat history kept across reruns, clear button | ⭐ added |
| 🔗 Clickable `[mm:ss]` citations that jump to that moment in the video | ⭐ added |
| 📝 **Auto Notes**: map/reduce summarisation → TL;DR, takeaways, topics, definitions table, action items, quotes, self-test quiz | ⭐ added |
| 📄 **PDF export** of those notes — styled headings, definition table, page numbers and clickable timestamp links | ⭐ added |
| ⬇️ Download notes as Markdown **or** PDF — [sample output](docs/sample_notes.pdf) | ⭐ added |
| 🧾 Full transcript browser by timestamp | ⭐ added |
| 📤 Transcript upload (`.srt` / `.vtt` / `.txt`) for when YouTube blocks the server's IP | ⭐ added |
| ⚡ `st.cache_data` so re-processing a video costs zero API calls | ⭐ added |
| 🚦 Batched embeddings + throttled chat calls so free-tier quotas don't break a run | ⭐ added |

---

## 🏗️ Architecture

```text
YouTube URL
   └─> Video ID parsing (pipeline.extract_video_id)
         └─> Transcript snippets (start, text)
               └─> Overlapping chunk windows  ──►  "[3-17] 04:12 - 09:30  <text>"
                     ├─► Gemini embedding-001 ──► FAISS index ──► similarity search
                     │                                             └─► Gemini Flash ─► answer + citation
                     └─► map (per batch: timestamped bullets) ─► reduce ─► Markdown notes + quiz
```

Why timestamps live *inside* the chunk text: retrieval returns the chunk verbatim, so both
the chat answer and the notes can cite the exact moment in the video — no extra lookup needed.
The same `**[mm:ss]**` markers survive into the PDF, where `fpdf2` renders them as real
`?t=<seconds>` hyperlinks, so a printed note still jumps back to the source moment.

## 📂 Project structure

```text
├── app.py           # Streamlit UI, session state, tabs, caching
├── pipeline.py      # URL parsing, transcript fetch, chunking, FAISS build
├── rag.py           # Gemini LLM factory + grounded answer prompt
├── notes.py         # map/reduce note generation, Markdown templates, deep links
├── pdf_export.py    # Markdown -> PDF (headings, tables, links, page numbers, font fallback)
├── ratelimit.py     # shared throttle + 429 retry that honours Google's retryDelay
├── smoke_test.py    # offline checks (no API key needed): python smoke_test.py
├── docs/            # sample PDF output
├── requirements.txt
├── .env.example     # copy to .env and add your key
└── .streamlit/secrets.toml   # cloud only, never committed
```

---

## 🧠 Design decisions

| Choice | Why |
|---|---|
| Timestamps live **inside** the chunk text | Retrieval returns the chunk verbatim, so citations come back for free — no parallel metadata store to keep in sync |
| Chunk on caption-snippet boundaries, not raw characters | Cuts never split mid-sentence, so an embedding represents a complete thought |
| 3-snippet overlap between chunks | Without it, an idea straddling a boundary becomes unretrievable |
| Batched `embed_content` (90 texts/request) | LangChain's wrapper issues one request per document and dies at the free tier's 100 requests/min on any long video |
| Map/reduce for notes instead of one giant prompt | Transcripts exceed the practical context window; batching keeps each call small and cost linear |
| Vectors cached per video | Re-indexing is a pure FAISS rebuild — zero API spend |
| Chat model = `gemini-flash-latest`, not a dated snapshot | Google retires pinned versions for new API keys — a `gemini-2.5-flash` deployment now 404s on every question. Override with `GEMINI_CHAT_MODEL` |
| `thinking_budget=0` on chat calls | Retrieval-augmented answers restate context instead of reasoning over it; measured 432 of 500 output tokens were hidden reasoning, ~40s per question instead of ~2s |
| Daily quota fails fast instead of retrying | Per-minute and per-day exhaustion both arrive as 429, but waiting out a *daily* cap is pointless — it would hang a demo for five minutes to no purpose |
| Cloud PDFs download Noto Sans once | Streamlit's image installs no TrueType fonts, and the core PDF fonts are latin-1 only, so `café` and `₹` would vanish silently. One cached download per container covers the rest of Unicode; a host with no egress still renders, just plainer |
| Fixed "not in the transcript" reply | A RAG demo is only trustworthy when it refuses instead of inventing |

---

## ⚙️ Quickstart (local)

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

copy .env.example .env           # then paste your Gemini key into .env
streamlit run app.py             # opens http://localhost:8501
```

Get a free Gemini key at <https://aistudio.google.com/apikey>.

Verify the core logic without spending any API credits:

```bash
python smoke_test.py
```

## ☁️ Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repository.
2. Open <https://share.streamlit.io> → **New app** → pick the repo, `Main script: app.py`.
3. **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY="your-key"
   ```
4. Deploy. The app reads the key from `st.secrets` in the cloud and from `.env` locally.

### ⚠️ YouTube blocks cloud IPs

`youtube-transcript-api` works fine on your laptop, but YouTube refuses requests from
datacenter ranges — so a Streamlit Cloud deployment gets *"YouTube is blocking requests from
your IP"* the moment you press Process. Two supported ways around it:

| Option | Cost | What you do |
|---|---|---|
| **Paste / upload transcript** mode | free | Download captions with `yt-dlp --write-auto-subs --sub-langs en --skip-download <url>` (it writes an `.en.vtt` file), or copy the text straight out of YouTube's *Show transcript* panel and paste it in. `.srt`/`.vtt` keep timestamps, so citations and PDF deep links still work — including YouTube's rolling auto-captions, which the parser stitches back into one clean pass. |
| **Residential proxy** | paid trial | Add `TRANSCRIPT_PROXY_HTTPS_URL` to the app secrets; `pipeline._transcript_api()` routes requests through it and URL mode works again. |

Without either, the deployed app still runs — it just cannot reach YouTube itself.

## 🛠️ Tech stack

Python · Streamlit · LangChain · Gemini Embedding 001 · Gemini Flash · FAISS · youtube-transcript-api

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `No transcripts were provided` / `TranscriptsDisabled` | Video has no captions — pick another, or change the language selector |
| `YouTube is blocking requests from your IP` | Expected on cloud hosts — use **Paste / upload transcript**, or set `TRANSCRIPT_PROXY_HTTPS_URL` |
| `Could not get API key` | `.env` missing `GEMINI_API_KEY=...`, or secrets not set on Streamlit Cloud |
| Rate limit (429, per minute) | The app already spaces requests out and honours Google's `retryDelay` — let it finish | 
| "used up today's requests for ... (20/day)" | The free tier allows **20 chat requests per model per day**. Wait for the reset, or add `GEMINI_CHAT_MODEL="gemini-flash-lite-latest"` to secrets for a larger allowance |
| Notes cover only part of a long video | Increase the chunk slider; cost grows linearly with batches of 12 |
| `faiss-cpu` will not install | Use Python 3.11–3.13; pre-built wheels lag behind brand-new Python releases |
| `FAISS.from_embeddings() missing ... 'text_embeddings'` | A LangChain version drifted out from under the code — `requirements.txt` is pinned to the versions this repo is tested against, so reinstall with `pip install -r requirements.txt --force-reinstall` |
| PDF text looks plain / some characters missing | `pdf_export.py` auto-picks Arial → Calibri → DejaVu; a host with **no** TrueType fonts (Streamlit Cloud) downloads Noto Sans once and caches it in the temp dir. Force one with `PDF_FONT_PATH=C:\Windows\Fonts\arial.ttf` (or `PDF_FONT_DIR`) |
| `Undefined font: notesB` | The page header needs the font family while `add_page()` is still running, so it has to be decided before any page exists. Current `pdf_export.py` does that (`_resolve_family()`); if you hit this, pull the latest version of that file |

## 📄 License

MIT for this work — see [LICENSE](LICENSE). The base chatbot it extends is credited above; check
its own terms before reusing it commercially. API usage costs are yours; the free Gemini tier is
enough for demos.

## 🧪 Evidence

The AI calls cost API credits, so the deterministic parts of the pipeline are covered by
[`smoke_test.py`](smoke_test.py) — URL parsing, chunk/label round-trips, note batching, 429
retry/backoff behaviour and PDF rendering. Run it with `python smoke_test.py`; it needs no key
and makes no network calls.
