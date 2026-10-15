"""YouTube AI Assistant - chat with any video and generate timestamped study notes."""

from __future__ import annotations

import os

import streamlit as st

import notes as notes_module
import pdf_export as pdf_module
import pipeline
import rag

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


st.set_page_config(page_title="YouTube AI Assistant", page_icon="🎥", layout="wide")


# ---------------------------------------------------------------- helpers ----
def get_api_key() -> str | None:
    """Read the Gemini key from Streamlit secrets (cloud) or the environment (local)."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@st.cache_data(show_spinner="📄 Building your PDF...")
def render_pdf(markdown_text: str, title: str, video_url: str) -> bytes:
    return pdf_module.notes_to_pdf(markdown_text, title, video_url)


@st.cache_data(show_spinner=False)
def load_transcript(video_id: str, languages: tuple[str, ...]) -> list[tuple[int, str]]:
    return [(s.start, s.text) for s in pipeline.fetch_snippets(video_id, list(languages))]


@st.cache_data(show_spinner=False)
def build_chunks(video_id: str, languages: tuple[str, ...], chunk_chars: int) -> list[str]:
    snippets = [pipeline.Snippet(start, text) for start, text in load_transcript(video_id, languages)]
    batches = pipeline.chunk_snippets(snippets, chunk_chars=chunk_chars)
    return [pipeline.chunk_to_text(batch, i) for i, batch in enumerate(batches)]


def init_state() -> None:
    st.session_state.setdefault("video_id", None)
    st.session_state.setdefault("video_title", "")
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("notes", "")


def citation_line(chunk_text: str, video_id: str) -> str:
    _, _, label = pipeline.chunk_label(chunk_text)
    if not label:
        return ""
    seconds = notes_module.seconds_from_label(label)
    if seconds is None:
        return f"> 🕒 `{label}`"
    return f"> 🕒 [`{label}`](https://www.youtube.com/watch?v={video_id}&t={seconds}s)"


# ------------------------------------------------------------------- ui ----
init_state()
st.title("🎥 YouTube AI Assistant")
st.caption("Chat with any YouTube video, or download timestamped study notes generated from its transcript.")

api_key = get_api_key()
if not api_key:
    st.info(
        "Set your Gemini API key first: create a `.env` file in this folder containing "
        "`GEMINI_API_KEY=your_key` (local) or add it in Streamlit Cloud secrets."
    )

left, right = st.columns([3, 1])
with left:
    youtube_url = st.text_input("YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=...")
with right:
    language = st.selectbox(
        "Transcript language",
        ["en", "hi", "es", "fr", "de", "pt", "ja", "zh-CN"],
        help="Fallback order: the selected language, then English.",
    )
    chunk_size = st.slider("Chunk size (characters)", 600, 2000, 1200, step=100)

if youtube_url:
    video_id = pipeline.extract_video_id(youtube_url.strip())
    if video_id:
        st.video(youtube_url)
    else:
        st.error("That does not look like a YouTube URL.")
        video_id = None
else:
    video_id = None

if st.button("🚀 Process video", type="primary", disabled=not video_id or not api_key):
    languages = tuple(dict.fromkeys([language, "en"]))
    try:
        with st.spinner("🎬 Loading transcript..."):
            chunks = build_chunks(video_id, languages, chunk_size)
            if not chunks:
                raise ValueError("No transcript found for this video. Try another language or video.")

        with st.spinner("🧠 Creating embeddings and the FAISS index..."):
            vector_store = pipeline.build_vector_store(chunks, api_key)

        st.session_state.update(
            video_id=video_id,
            video_title=pipeline.fetch_video_title(video_id),
            chunks=chunks,
            vector_store=vector_store,
            messages=[],
            notes="",
        )
        st.success(f"✅ Ready — {st.session_state.video_title} ({len(chunks)} chunks)")
    except Exception as error:
        st.error(f"Could not process this video: {error}")

if not st.session_state.vector_store:
    st.stop()

ready_title = st.session_state.video_title
chat_tab, notes_tab, transcript_tab = st.tabs(["💬 Chat", "📝 Auto notes", "🧾 Transcript"])

# ----------------------------------------------------------------- chat ----
with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about this video...")
    if question:
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        try:
            with st.spinner("🔍 Searching the transcript..."):
                docs = rag.retrieve(st.session_state.vector_store, question, k=4)

            with st.spinner("🤔 Thinking..."):
                answer = rag.answer_question(rag.make_llm(api_key), question, [d.page_content for d in docs])

            sources = "\n\n".join(
                line for line in (citation_line(d.page_content, st.session_state.video_id) for d in docs) if line
            )
            display = answer + ("\n\n" + sources if sources else "")
            st.chat_message("assistant").markdown(display)
            st.session_state.messages.append({"role": "assistant", "content": display})
        except Exception as error:
            st.error(f"Error: {error}")

    if st.session_state.messages and st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- notes ----
with notes_tab:
    st.markdown(f"### Study notes for **{ready_title}**")
    st.caption(notes_module.estimate_cost_note(len(st.session_state.chunks)))
    coverage = st.slider(
        "How many transcript chunks to read",
        5,
        max(5, len(st.session_state.chunks)),
        min(max(5, len(st.session_state.chunks)), 80),
        help="More chunks = better notes but more API calls. The transcript is processed in batches.",
    )

    if st.button("📝 Generate notes", type="primary"):
        progress_bar = st.progress(0.0, text="Preparing...")
        try:
            llm = rag.make_llm(api_key, temperature=0.3)
            used = st.session_state.chunks[:coverage]
            markdown_notes = notes_module.add_timestamp_links(
                notes_module.generate_notes(
                    llm,
                    ready_title,
                    used,
                    progress=lambda done, total, label: progress_bar.progress(
                        min(done / total, 1.0), text=label
                    ),
                ),
                st.session_state.video_id,
            )
            st.session_state.notes = markdown_notes
        except Exception as error:
            st.error(f"Could not generate notes: {error}")

    if st.session_state.notes:
        st.markdown(st.session_state.notes)
        slug = "".join(c if c.isalnum() or c in " -_" else "" for c in ready_title)[:60].strip().replace(" ", "_")
        file_stem = f"{slug or 'video'}_notes"
        left_md, left_pdf = st.columns(2)

        with left_md:
            st.download_button(
                "⬇️ Download notes (Markdown)",
                data=st.session_state.notes,
                file_name=f"{file_stem}.md",
                mime="text/markdown",
                width="stretch",
            )
        with left_pdf:
            try:
                pdf_bytes = render_pdf(
                    st.session_state.notes,
                    ready_title,
                    f"https://www.youtube.com/watch?v={st.session_state.video_id}",
                )
            except Exception as error:
                pdf_bytes = None
                st.error(f"PDF export failed: {error}")
            if pdf_bytes:
                st.download_button(
                    "📄 Download notes (PDF)",
                    data=pdf_bytes,
                    file_name=f"{file_stem}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

# ------------------------------------------------------------ transcript ----
with transcript_tab:
    st.caption("Every chunk carries the timestamp range it covers — that is what makes answers and notes citable.")
    for chunk in st.session_state.chunks:
        _, _, label = pipeline.chunk_label(chunk)
        with st.expander(f"🕒 {label}"):
            st.write("\n".join(chunk.splitlines()[1:]))
