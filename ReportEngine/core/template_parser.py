"""
Markdown template slicing tool.

LLM needs "chapter-by-chapter invocation", so Markdown templates must be parsed into structured chapter queues.
Here uses lightweight regex and indentation heuristics, compatible with "# Title" and
"- **1.0 Title** /   - 1.1 Subtitle" and other formats.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

SECTION_ORDER_STEP = 10


@dataclass
class TemplateSection:
    """
    Template section entity.

    Records title, slug, order, depth, raw title, chapter number and outline,
    facilitating subsequent nodes to reference in prompts and maintain anchor consistency.
    """

    title: str
    slug: str
    order: int
    depth: int
    raw_title: str
    number: str = ""
    chapter_id: str = ""
    outline: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Serialize section entity to dictionary.

        This structure is widely used for prompt context and layout/word budget node inputs.
        """
        return {
            "title": self.title,
            "slug": self.slug,
            "order": self.order,
            "depth": self.depth,
            "number": self.number,
            "chapterId": self.chapter_id,
            "outline": self.outline,
        }


# Parsing expressions deliberately avoid using `.*` to maintain matching determinism,
# and avoid regex DoS risks common in untrusted template text.
heading_pattern = re.compile(
    r"""
    (?P<marker>\#{1,6})       # Markdown heading marker
    [ \t]+                    # Required whitespace characters
    (?P<title>[^\r\n]+)       # Title text not containing newlines
    """,
    re.VERBOSE,
)
bullet_pattern = re.compile(
    r"""
    (?P<marker>[-*+])         # List bullet marker
    [ \t]+
    (?P<title>[^\r\n]+)
    """,
    re.VERBOSE,
)
number_pattern = re.compile(
    r"""
    (?P<num>
        (?:0|[1-9]\d*)
        (?:\.(?:0|[1-9]\d*))*
    )
    (?:
        (?:[ \t\u00A0\u3000:：-]+|\.(?!\d))+
        (?P<label>[^\r\n]*)
    )?
    """,
    re.VERBOSE,
)


def parse_template_sections(template_md: str) -> List[TemplateSection]:
    """
    Slice Markdown template into chapter list (by main headings).

    Each returned TemplateSection carries slug/order/chapter number,
    facilitating subsequent chapter-by-chapter invocation and anchor generation. Parsing is compatible with
    "# Title", "unsigned numbering", "list outline" and other formats.

    Args:
        template_md: Full template Markdown text.

    Returns:
        list[TemplateSection]: Structured chapter sequence.
    """

    sections: List[TemplateSection] = []
    current: Optional[TemplateSection] = None
    order = SECTION_ORDER_STEP
    used_slugs = set()

    for raw_line in template_md.splitlines():
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        meta = _classify_line(stripped, indent)
        if not meta:
            continue

        if meta["is_section"]:
            slug = _ensure_unique_slug(meta["slug"], used_slugs)
            section = TemplateSection(
                title=meta["title"],
                slug=slug,
                order=order,
                depth=meta["depth"],
                raw_title=meta["raw"],
                number=meta["number"],
            )
            sections.append(section)
            current = section
            order += SECTION_ORDER_STEP
            continue

        # Outline entries
        if current:
            current.outline.append(meta["title"])

    for idx, section in enumerate(sections, start=1):
        # Generate stable chapter_id for each section for subsequent reference
        section.chapter_id = f"S{idx}"

    return sections


def _classify_line(stripped: str, indent: int) -> Optional[dict]:
    """
    Classify lines based on indentation and symbols.

    Uses regex to determine if current line is chapter title, outline or normal list item,
    and derives depth/slug/number and other derived information.

    Args:
        stripped: Original line with leading/trailing whitespace removed.
        indent: Number of leading spaces, used to distinguish levels.

    Returns:
        dict | None: Recognized metadata; returns None when unrecognizable.
    """

    heading_match = heading_pattern.fullmatch(stripped)
    if heading_match:
        level = len(heading_match.group("marker"))
        payload = _strip_markup(heading_match.group("title").strip())
        title_info = _split_number(payload)
        slug = _build_slug(title_info["number"], title_info["title"])
        return {
            "is_section": level <= 2,
            "depth": level,
            "title": title_info["display"],
            "raw": payload,
            "number": title_info["number"],
            "slug": slug,
        }

    bullet_match = bullet_pattern.fullmatch(stripped)
    if bullet_match:
        payload = _strip_markup(bullet_match.group("title").strip())
        title_info = _split_number(payload)
        slug = _build_slug(title_info["number"], title_info["title"])
        is_section = indent <= 1 and (
            bool(title_info["number"]) or _looks_like_unnumbered_section_title(title_info["title"])
        )
        depth = 1 if indent <= 1 else 2
        return {
            "is_section": is_section,
            "depth": depth,
            "title": title_info["display"],
            "raw": payload,
            "number": title_info["number"],
            "slug": slug,
        }

    # Compatible with lines like "1.1 ..." without prefix symbols
    number_match = number_pattern.fullmatch(stripped)
    if number_match and number_match.group("label"):
        payload = stripped
        title = number_match.group("label").strip()
        number = number_match.group("num")
        slug = _build_slug(number, title)
        is_section = indent == 0 and number.count(".") <= 1
        depth = 1 if is_section else 2
        display = f"{number} {title}" if title else number
        return {
            "is_section": is_section,
            "depth": depth,
            "title": display,
            "raw": payload,
            "number": number,
            "slug": slug,
        }

    return None


def _strip_markup(text: str) -> str:
    """Remove wrapping **, __ and other emphasis markers to avoid interfering with title matching."""
    if text.startswith(("**", "__")) and text.endswith(("**", "__")) and len(text) > 4:
        return text[2:-2].strip()
    return text


def _looks_like_unnumbered_section_title(title: str) -> bool:
    """
    Distinguish real unnumbered section headings from instruction bullets.

    Custom report templates often use top-level bullets as instructions under a
    numbered chapter, for example "Summarize the overall finding...". Those
    should become outline notes, not standalone chapters.
    """
    normalized = (title or "").strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    if lowered in {"appendix", "references", "source list", "executive summary", "conclusion", "conclusions"}:
        return True
    if normalized.endswith((".", ";", ":")):
        return False
    first_word = re.split(r"\s+", lowered, maxsplit=1)[0]
    imperative_verbs = {
        "summarize",
        "state",
        "explain",
        "include",
        "present",
        "preserve",
        "distinguish",
        "provide",
        "describe",
        "show",
        "list",
        "compare",
        "analyze",
        "note",
    }
    if first_word in imperative_verbs:
        return False
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    return 0 < len(words) <= 6


def _split_number(payload: str) -> dict:
    """
    Split number and title.

    For example `1.2 Market Trends` will be split into number=1.2, label=Market Trends,
    and provides display for title filling.

    Args:
        payload: Original title string.

    Returns:
        dict: Contains number/title/display.
    """
    match = number_pattern.fullmatch(payload)
    number = match.group("num") if match else ""
    label = match.group("label") if match else payload
    label = (label or "").strip()
    display = f"{number} {label}".strip() if number else label or payload
    title_core = label or payload
    return {
        "number": number,
        "title": title_core,
        "display": display,
    }


def _build_slug(number: str, title: str) -> str:
    """
    Generate anchor based on number/title, prefer reusing number, slugify title when missing.

    Args:
        number: Chapter number.
        title: Title text.

    Returns:
        str: Slug in format `section-1-0`.
    """
    if number:
        token = number.replace(".", "-")
    else:
        token = _slugify_text(title)
    token = token or "section"
    return f"section-{token}"


def _slugify_text(text: str) -> str:
    """
    Perform noise reduction and transliteration on arbitrary text to get URL-friendly slug fragment.

    Normalizes case, removes special symbols and preserves Chinese characters, ensuring anchor readability.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("·", "-").replace(" ", "-")
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-").lower()


def _ensure_unique_slug(slug: str, used: set) -> str:
    """
    If slug is duplicated, automatically append number until unique in used set.

    Uses `-2/-3...` approach to ensure identical titles don't produce duplicate anchors.

    Args:
        slug: Initial slug.
        used: Used set.

    Returns:
        str: Deduplicated slug.
    """
    if slug not in used:
        used.add(slug)
        return slug
    base = slug
    idx = 2
    while slug in used:
        slug = f"{base}-{idx}"
        idx += 1
    used.add(slug)
    return slug


__all__ = ["TemplateSection", "parse_template_sections"]
