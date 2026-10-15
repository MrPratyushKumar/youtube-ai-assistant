"""Render generated Markdown notes into a styled, clickable PDF."""

from __future__ import annotations

import os
import re

import markdown as md
from fpdf import FPDF

ACCENT = (37, 99, 235)
TEXT = (28, 32, 38)
MUTED = (110, 118, 128)

_FONT_CANDIDATES = {
    "": ["arial.ttf", "calibri.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"],
    "B": ["arialbd.ttf", "calibrib.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
    "I": ["ariali.ttf", "calibrii.ttf", "LiberationSans-Italic.ttf", "DejaVuSans-Oblique.ttf"],
    "BI": ["arialbi.ttf", "LiberationSans-BoldItalic.ttf", "DejaVuSans-BoldOblique.ttf"],
}
_FONT_DIRS = [
    os.environ.get("PDF_FONT_DIR", ""),
    r"C:\Windows\Fonts",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/System/Library/Fonts",
]

_LINK_TIMESTAMP = re.compile(r"\*\*\[`?(\d{1,2}(?::\d{2}){1,2})`?\]\((https?://[^)]+)\)\*\*")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002190-\U00002BFF\U0000FE0F\U0000200D]")


def _find_font(style: str) -> str | None:
    override = os.environ.get("PDF_FONT_PATH")
    if override and style == "":
        return override
    for folder in _FONT_DIRS:
        if not folder:
            continue
        for name in _FONT_CANDIDATES[style]:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    return None


def _prepare(markdown_text: str) -> str:
    """Normalise the note Markdown for the HTML subset fpdf2 understands."""
    text = _LINK_TIMESTAMP.sub(r"[\1](\2)", markdown_text)
    text = re.sub(r"^# (.+)$", "", text, count=1, flags=re.MULTILINE)  # title goes in the page header
    return text.strip()


def _to_html(markdown_text: str) -> str:
    body = md.markdown(_prepare(markdown_text), extensions=["tables", "fenced_code"])
    return f'<body style="font-family:Notes">{body}</body>'


class _NotesPDF(FPDF):
    def __init__(self, title: str, video_url: str) -> None:
        super().__init__(format="A4")
        self._title = title
        self._video_url = video_url
        self.set_margins(18, 20, 18)
        self.set_auto_page_break(True, margin=22)
        self.set_text_color(*TEXT)

    def header(self) -> None:
        self.set_font("Notes", "B", 14)
        self.set_text_color(*TEXT)
        self.cell(0, 8, self._title[:110], new_x="LMARGIN", new_y="NEXT")
        self.set_font("Notes", "", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "Study notes generated from the video transcript", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.5)
        self.line(18, self.get_y() + 2, 192, self.get_y() + 2)
        self.ln(8)

    def footer(self) -> None:
        self.set_y(-16)
        self.set_font("Notes", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", align="R")

    def open_video_link(self) -> None:
        self.set_font("Notes", "", 9)
        self.set_text_color(*ACCENT)
        self.multi_cell(
            0,
            5,
            f"Watch: {self._video_url}",
            link=self._video_url,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_text_color(*TEXT)
        self.ln(4)


def notes_to_pdf(markdown_text: str, title: str, video_url: str) -> bytes:
    """Convert notes Markdown into PDF bytes (clickable timestamps included)."""
    pdf = _NotesPDF(title=title, video_url=video_url)

    regular = _find_font("")
    if regular:
        for style in ("", "B", "I", "BI"):
            path = _find_font(style) or regular
            try:
                pdf.add_font("Notes", style, path)
            except Exception:
                if style == "":
                    raise
        font_family, unicode_ok = "Notes", True
    else:
        # Core fonts are latin-1 only, so drop anything they cannot draw.
        font_family, unicode_ok = "helvetica", False

    pdf.set_font(font_family, "", 10.5)
    pdf.add_page()
    pdf.open_video_link()

    body = _to_html(markdown_text)
    if not unicode_ok:
        body = _EMOJI.sub("", body).encode("latin-1", "replace").decode("latin-1")
    pdf.write_html(body)

    return bytes(pdf.output())
