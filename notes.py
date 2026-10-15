"""Auto-notes: map/reduce summarisation of timestamped transcript chunks."""

from __future__ import annotations

import re

BATCH_SIZE = 6

_MAP_PROMPT = """You are preparing study notes from one section of a YouTube transcript.
Every block below is prefixed with its in-video timestamp range.

Return ONLY Markdown bullet points for this section. Each bullet must:
- state one concrete idea, definition, step, number, tool or example
- start with the block's timestamp in the form `**[mm:ss]**`
- be self-contained (readable without seeing the transcript)
Keep 3-8 bullets. No intro, no headings, no conclusions.

Transcript section:
"""

_REDUCE_PROMPT_HEADER = """You are turning timestamped bullet points from a YouTube video into
clean, exam-ready study notes in Markdown.

Rules:
- Use ONLY the supplied points; never add outside facts.
- Merge duplicates, keep the most precise timestamps, and preserve every `**[mm:ss]**` marker.
- Write in the same language as the points.
- Output Markdown only, with exactly these headings in this order:

# {title}

## 1. TL;DR
3 sentences maximum.

## 2. Key Takeaways
5-10 bullets, each starting with a timestamp.

## 3. Notes by Topic
Group the content under `### Topic name` sub-headings with timestamped bullets.

## 4. Concepts & Definitions
Table with columns: Term | Meaning (only terms the video actually defines).

## 5. Action Items / Steps
Numbered list, or "Not applicable" if the video is not instructional.

## 6. Notable Quotes
Up to 3 short quotes with timestamps, or "None".

## 7. Self-Test Questions
5 question + answer pairs based only on the video content.

Video title: {title}

Timestamped points:
"""


def seconds_from_label(label: str) -> int | None:
    """`'1:02:03 - 1:05:10'` or `'2:15 - 4:30'` -> seconds of the first time."""
    first = label.split("-")[0].strip()
    if not first:
        return None
    parts = first.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        return None
    try:
        return int(h) * 3600 + int(m) * 60 + int(s)
    except ValueError:
        return None


def _chunk_body(chunk_text: str) -> str:
    return "\n".join(chunk_text.splitlines()[1:]).strip()


def generate_notes(
    llm,
    title: str,
    chunks: list[str],
    max_chunks: int | None = None,
    progress=None,
) -> str:
    """Map each batch of chunks to timestamped points, then reduce into notes."""
    if not chunks:
        raise ValueError("There is no transcript to summarise.")

    used = chunks[:max_chunks] if max_chunks else chunks
    batches = [used[i : i + BATCH_SIZE] for i in range(0, len(used), BATCH_SIZE)]

    points: list[str] = []
    total_steps = len(batches) + 1
    for index, batch in enumerate(batches):
        section = "\n\n".join(f"{line_first(chunk)}\n{_chunk_body(chunk)}" for chunk in batch)
        result = llm.invoke(_MAP_PROMPT + section)
        points.append(result.content.strip())
        if progress:
            progress(index + 1, total_steps, f"Reading section {index + 1} of {len(batches)}...")

    if progress:
        progress(len(batches), total_steps, "Writing your notes...")
    reduce_prompt = _REDUCE_PROMPT_HEADER.format(title=title) + "\n\n".join(points)
    final = llm.invoke(reduce_prompt).content.strip()
    if progress:
        progress(total_steps, total_steps, "Notes ready")
    return final


def line_first(chunk_text: str) -> str:
    return chunk_text.splitlines()[0] if chunk_text else ""


def add_timestamp_links(markdown: str, video_id: str) -> str:
    """Turn `**[mm:ss]**` markers into clickable YouTube deep links."""
    pattern = re.compile(r"\*\*\[(\d{1,2}(?::\d{2}){1,2})\]\*\*")

    def replace(match: re.Match) -> str:
        stamp = match.group(1)
        seconds = seconds_from_label(stamp)
        if seconds is None:
            return match.group(0)
        return f"**[`{stamp}`](https://www.youtube.com/watch?v={video_id}&t={seconds}s)**"

    return pattern.sub(replace, markdown)


def estimate_cost_note(chunk_count: int) -> str:
    batches = max(1, -(-chunk_count // BATCH_SIZE))
    return f"{chunk_count} transcript chunks -> {batches} summariser calls + 1 notes call"
