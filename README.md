# 🎥 YouTube AI Assistant — Chat + Auto Notes (RAG)

Ask a YouTube video questions, and turn any video into downloadable, timestamped study notes.
Transcript-based Retrieval-Augmented Generation with LangChain, Gemini embeddings, FAISS and Streamlit.

> Forked and extended from the original *YouTube Video Chatbot* by Harsha
> ([live demo](https://youtube-ai-assistant-1926.streamlit.app/)).
> **What I added:** timestamped chunks, the Auto Notes pipeline, clickable citations,
> chat history, caching, multi-language transcripts and project structure.

---

## 🚀 Features

| Feature | Status |
|---|---|
| Paste any YouTube URL / short link / Shorts / embed / raw video ID | ✅ |
| Transcript extraction (`youtube-transcript-api`), language selectable | ✅ |
| Timestamp-aware chunking (overlapping windows of caption snippets) | ✅ |
| Gemini embeddings stored in a FAISS vector index | ✅ |
| Similarity search with `k` relevant chunks | ✅ |
| Grounded Q&A with Gemini 2.5 Flash | ✅ |
| 💬 Chat history kept across reruns, clear button | ⭐ added |
| 🔗 Clickable `[mm:ss]` citations that jump to that moment in the video | ⭐ added |
| 📝 **Auto Notes**: map/reduce summarisation → TL;DR, takeaways, topics, definitions table, action items, quotes, self-test quiz | ⭐ added |
| 📄 **PDF export** of those notes — styled headings, definition table, page numbers and clickable timestamp links | ⭐ added |
| ⬇️ Download notes as Markdown **or** PDF — [sample output](docs/sample_notes.pdf) | ⭐ added |
| 🧾 Full transcript browser by timestamp | ⭐ added |
| ⚡ `st.cache_data` so re-processing a video costs zero API calls | ⭐ added |

---

## 🏗️ Architecture

```text
YouTube URL
   └─> Video ID parsing (pipeline.extract_video_id)
         └─> Transcript snippets (start, text)
               └─> Overlapping chunk windows  ──►  "[3-17] 04:12 - 09:30  <text>"
                     ├─► Gemini embedding-001 ──► FAISS index ──► similarity search
                     │                                             └─► Gemini 2.5 Flash ─► answer + citation
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
├── smoke_test.py    # offline checks (no API key needed): python smoke_test.py
├── requirements.txt
├── .env.example     # copy to .env and add your key
└── .streamlit/secrets.toml   # cloud only, never committed
```

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

## 🛠️ Tech stack

Python · Streamlit · LangChain · Gemini Embedding 001 · Gemini 2.5 Flash · FAISS · youtube-transcript-api

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `No transcripts were provided` / `TranscriptsDisabled` | Video has no captions — pick another, or change the language selector |
| `Could not get API key` | `.env` missing `GEMINI_API_KEY=...`, or secrets not set on Streamlit Cloud |
| Rate limit (429) | Lower "How many transcript chunks to read" in the Notes tab |
| Notes cover only part of a long video | Increase the chunk slider; cost grows linearly with batches of 6 |
| `faiss-cpu` will not install | Use Python 3.11–3.13; pre-built wheels lag behind brand-new Python releases |
| PDF text looks plain / some characters missing | `pdf_export.py` auto-picks Arial → Calibri → DejaVu. Force one with `PDF_FONT_PATH=C:\Windows\Fonts\arial.ttf` (or `PDF_FONT_DIR`) |

## 📄 License

MIT (original project's terms). API usage costs are yours; the free Gemini tier is enough for demos.

## 🧪 Evidence

The AI calls cost API credits, so the deterministic parts of the pipeline are covered by
[`smoke_test.py`](smoke_test.py) — URL parsing, chunk/label round-trips, note batching and PDF
rendering. Run it with `python smoke_test.py`; it needs no key and makes no network calls.
