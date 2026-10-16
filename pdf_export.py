"""Render generated Markdown notes into a styled, clickable PDF."""

from __future__ import annotations

import os
import re
import tempfile

import markdown as md
import requests
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

_NOTO_URL = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/"
_NOTO_FILES = {
    "": "NotoSans-Regular.ttf",
    "B": "NotoSans-Bold.ttf",
    "I": "NotoSans-Italic.ttf",
    "BI": "NotoSans-BoldItalic.ttf",
}
_TTF_MAGICS = (b"\x00\x01\x00\x00", b"true", b"OTTO")
_DOWNLOAD_TIMEOUT = 30
_download_unavailable: set[str] = set()


def _download_font(style: str) -> str | None:
    """Fetch the Noto Sans file for `style`, cached in the temp directory.

    Cloud images ship no TrueType fonts, which leaves only the core fonts — latin-1,
    so they drop accented letters and currency symbols from the notes silently.
    Downloading once per container keeps the full Unicode range; a host with no
    egress just gets the core fonts back.
    """
    if style in _download_unavailable:
        return None
    name = _NOTO_FILES[style]
    path = os.path.join(tempfile.gettempdir(), "youtube-ai-assistant-fonts", name)
    if os.path.isfile(path) and os.path.getsize(path) > 10_000:
        return path
    try:
        response = requests.get(_NOTO_URL + name, timeout=_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        data = response.content
        if data[:4] not in _TTF_MAGICS:
            raise ValueError(f"{name} is not a TrueType file")
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        partial = path + ".part"
        with open(partial, "wb") as handle:
            handle.write(data)
        os.replace(partial, path)  # never leave a truncated font in the cache
        return path
    except Exception:
        _download_unavailable.add(style)
        return None


def _find_font(style: str) -> str | None:
    override = os.environ.get("PDF_FONT_PATH")
    if override and style == "":
        return override
    index = _font_index()
    for name in _FONT_CANDIDATES[style]:
        path = index.get(name.lower())
        if path:
            return path
    return _download_font(style)


_index: dict[str, str] | None = None


def _font_index() -> dict[str, str]:
    """Map lowercased TTF filenames to paths, searching font trees recursively.

    Distributions bury fonts at unpredictable depths — `/usr/share/fonts/truetype/...`,
    `.../X11/`, `.../noto/...` — so a fixed list of directories silently finds nothing
    and the PDF loses every non-latin character.
    """
    global _index
    if _index is None:
        roots = [folder for folder in _FONT_DIRS if folder]
        roots += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
        found: dict[str, str] = {}
        for root in roots:
            for folder, _, files in os.walk(root):
                for name in files:
                    if name.lower().endswith(".ttf") and name.lower() not in found:
                        found[name.lower()] = os.path.join(folder, name)
        _index = found
    return _index


def _prepare(markdown_text: str) -> str:
    """Normalise the note Markdown for the HTML subset fpdf2 understands."""
    text = _LINK_TIMESTAMP.sub(r"[\1](\2)", markdown_text)
    text = re.sub(r"^# (.+)$", "", text, count=1, flags=re.MULTILINE)  # title goes in the page header
    return text.strip()


def _to_html(markdown_text: str, family: str) -> str:
    body = md.markdown(_prepare(markdown_text), extensions=["tables", "fenced_code"])
    style = f' style="font-family:{family}"' if family != _CORE_FAMILY else ""
    return f"<body{style}>{body}</body>"


_CORE_FAMILY = "helvetica"


class _NotesPDF(FPDF):
    """Header and footer run inside add_page(), before the caller can set a font,
    so the resolved family has to live on the instance."""

    def __init__(self, title: str, video_url: str, family: str) -> None:
        super().__init__(format="A4")
        self._title = title
        self._video_url = video_url
        self._family = family
        self.set_margins(18, 20, 18)
        self.set_auto_page_break(True, margin=22)
        self.set_text_color(*TEXT)

    def header(self) -> None:
        self.set_font(self._family, "B", 14)
        self.set_text_color(*TEXT)
        self.cell(0, 8, self._title[:110], new_x="LMARGIN", new_y="NEXT")
        self.set_font(self._family, "", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "Study notes generated from the video transcript", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.5)
        self.line(18, self.get_y() + 2, 192, self.get_y() + 2)
        self.ln(8)

    def footer(self) -> None:
        self.set_y(-16)
        self.set_font(self._family, "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", align="R")

    def open_video_link(self) -> None:
        if not self._video_url:  # uploaded transcripts have no source video
            return
        self.set_font(self._family, "", 9)
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


def _resolve_family() -> str:
    """Pick the font family the whole document can use, before any page is drawn.

    A half-registered family is worse than none: write_html asks for bold the moment
    it meets `**text**`, and an undefined `NotesB` aborts the export mid-render.
    """
    regular = _find_font("")
    if not regular:
        return _CORE_FAMILY
    probe = FPDF()
    try:
        for style in ("", "B", "I", "BI"):
            probe.add_font("Notes", style, _find_font(style) or regular)
    except Exception:
        return _CORE_FAMILY
    return "Notes"


def notes_to_pdf(markdown_text: str, title: str, video_url: str) -> bytes:
    """Convert notes Markdown into PDF bytes (clickable timestamps included)."""
    family = _resolve_family()
    pdf = _NotesPDF(title=title, video_url=video_url, family=family)

    if family == "Notes":
        regular = _find_font("")
        for style in ("", "B", "I", "BI"):
            pdf.add_font("Notes", style, _find_font(style) or regular)

    pdf.set_font(family, "", 10.5)
    pdf.add_page()
    pdf.open_video_link()

    body = _to_html(markdown_text, family)
    if family == _CORE_FAMILY:
        # Core fonts are latin-1 only, so drop anything they cannot draw.
        body = _EMOJI.sub("", body).encode("latin-1", "replace").decode("latin-1")
    pdf.write_html(body)

    return bytes(pdf.output())
