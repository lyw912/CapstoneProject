"""
Chapter-level JSON generation node.

Each chapter independently calls LLM based on Markdown template slices, streams to raw files,
validates and persists standardized JSON upon completion. This node is solely responsible for obtaining compliant chapters.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple, Callable, Optional, Set

from loguru import logger

from ..core import TemplateSection, ChapterStorage
from ..ir import (
    ALLOWED_BLOCK_TYPES,
    ALLOWED_INLINE_MARKS,
    ENGINE_AGENT_TITLES,
    IRValidator,
)
from ..prompts import (
    SYSTEM_PROMPT_CHAPTER_JSON,
    SYSTEM_PROMPT_CHAPTER_JSON_REPAIR,
    SYSTEM_PROMPT_CHAPTER_JSON_RECOVERY,
    build_chapter_repair_prompt,
    build_chapter_recovery_payload,
    build_chapter_user_prompt,
)
from ..utils.json_parser import RobustJSONParser, JSONParseError
from .base_node import BaseNode

try:
    from json_repair import repair_json as _json_repair_fn
except ImportError:  # pragma: no cover - optional dependency
    _json_repair_fn = None


class ChapterJsonParseError(ValueError):
    """Exception raised when chapter LLM output cannot be parsed as valid JSON, with raw text attached for troubleshooting."""

    def __init__(self, message: str, raw_text: Optional[str] = None):
        """
        Construct exception with original output attached for logging.

        Args:
            message: Human-readable error description.
            raw_text: Complete LLM output that triggered the exception.
        """
        super().__init__(message)
        self.raw_text = raw_text


class ChapterContentError(ValueError):
    """
    Chapter content sparse exception.

    Triggered when LLM outputs only titles or insufficient body content to support a chapter,
    driving retries to ensure report quality.
    """

    def __init__(
        self,
        message: str,
        chapter: Optional[Dict[str, Any]] = None,
        body_characters: int = 0,
        narrative_characters: int = 0,
        non_heading_blocks: int = 0,
    ):
        """Save body content characteristics for retry and fallback strategy reference."""
        super().__init__(message)
        self.chapter_payload: Optional[Dict[str, Any]] = chapter
        self.body_characters: int = int(body_characters or 0)
        self.narrative_characters: int = int(narrative_characters or 0)
        self.non_heading_blocks: int = int(non_heading_blocks or 0)


class ChapterValidationError(ValueError):
    """
    Exception raised when chapter structure still fails validation after local and LLM repair.

    This exception triggers single-chapter retries at the Agent layer without restarting the entire report.
    """

    def __init__(self, message: str, errors: Optional[List[str]] | None = None):
        super().__init__(message)
        self.errors: List[str] = list(errors or [])


class ChapterGenerationNode(BaseNode):
    """
    Responsible for calling LLM per chapter and validating JSON structure.

    Core capabilities:
        - Construct chapter-level payload and prompts;
        - Stream to raw files and pass through deltas;
        - Attempt repair/parse LLM output and validate using IRValidator;
        - Fault-tolerant repair of block structure to ensure final JSON is renderable.
    """

    _COLON_EQUALS_PATTERN = re.compile(r'(":\s*)=')
    _LINE_BREAK_SENTINEL = "__LINE_BREAK__"
    _INLINE_MARK_ALIASES = {
        "strong": "bold",
        "b": "bold",
        "em": "italic",
        "emphasis": "italic",
        "i": "italic",
        "u": "underline",
        "strike-through": "strike",
        "strikethrough": "strike",
        "s": "strike",
        "codeblock": "code",
        "monospace": "code",
        "hyperlink": "link",
        "url": "link",
        "colour": "color",
        "textcolor": "color",
        "bgcolor": "highlight",
        "background": "highlight",
        "highlightcolor": "highlight",
        "sub": "subscript",
        "sup": "superscript",
    }
    # Chapter is considered failed if it contains only headings or too few characters, forcing LLM regeneration
    _MIN_NON_HEADING_BLOCKS = 2
    _MIN_BODY_CHARACTERS = 600
    _MIN_NARRATIVE_CHARACTERS = 300
    _PARAGRAPH_FRAGMENT_MAX_CHARS = 80
    _PARAGRAPH_FRAGMENT_NO_TERMINATOR_MAX_CHARS = 240
    _TERMINATION_PUNCTUATION = set("。！？!?；;……")

    def __init__(
        self,
        llm_client,
        validator: IRValidator,
        storage: ChapterStorage,
        fallback_llm_clients: Optional[List[Tuple[str, Any]]] = None,
        error_log_dir: Optional[str | Path] = None,
    ):
        """
        Record LLM client/validator/chapter storage for run method scheduling.

        Args:
            llm_client: Client for calling large models
            validator: IR structure validator
            storage: Storage handler for chapter streaming persistence
        """
        super().__init__(llm_client, "ChapterGenerationNode")
        self.validator = validator
        self.storage = storage
        self.fallback_llm_clients: List[Tuple[str, Any]] = fallback_llm_clients or [
            ("report_engine", llm_client)
        ]
        error_dir = Path(error_log_dir or "logs/json_repair_failures")
        error_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_dir = error_dir
        self._failed_block_counter = 0
        self._active_run_id: Optional[str] = None
        self._rescue_attempted_labels: Dict[str, Set[str]] = {}
        self._skipped_placeholder_chapters: Set[str] = set()
        self._archived_failed_json: Dict[str, str] = {}
        # Use more robust JSON parser as fallback, extract valid chunks as much as possible
        self._robust_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,
        )

    def run(
        self,
        section: TemplateSection,
        context: Dict[str, Any],
        run_dir: Path,
        stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Call LLM for a single chapter, validate/persist chapter JSON and return structured result.

        Args:
            section: Chapter object generated from template slice, containing title/order/slug.
            context: Agent-constructed shared context (topic, word count, layout, etc.).
            run_dir: Chapter storage directory, returned by `ChapterStorage.start_session`.
            stream_callback: Optional streaming callback to push LLM deltas to frontend.
            **kwargs: Pass-through sampling parameters like temperature, top_p, etc.

        Returns:
            dict: Chapter JSON that has passed IR validation.

        Raises:
            ChapterJsonParseError: Still unable to parse valid JSON after multiple attempts.
            ChapterContentError: Body content density insufficient or only headings, requiring retry.
        """
        chapter_meta = {
            "chapterId": section.chapter_id,
            "slug": section.slug,
            "title": section.title,
            "order": section.order,
        }
        chapter_dir = self.storage.begin_chapter(run_dir, chapter_meta)
        run_id = run_dir.name
        self._ensure_run_state(run_id)
        llm_payload = self._build_payload(section, context)
        user_message = build_chapter_user_prompt(llm_payload)

        raw_text = self._stream_llm(
            user_message,
            chapter_dir,
            stream_callback=stream_callback,
            section_meta=chapter_meta,
            **kwargs,
        )
        parse_context: List[str] = []
        placeholder_created = False
        try:
            chapter_json = self._parse_chapter(raw_text)
        except ChapterJsonParseError as parse_error:
            logger.warning(f"{section.title} chapter JSON parsing failed, attempting cross-engine repair: {parse_error}")
            parse_context.append(str(parse_error))
            self._archive_failed_output(section, raw_text)
            recovered = self._attempt_cross_engine_json_rescue(
                section,
                llm_payload,
                raw_text,
                run_id,
            )
            if recovered:
                chapter_json = recovered
                logger.info(f"{section.title} chapter JSON repaired via cross-engine")
            else:
                placeholder = self._build_placeholder_chapter(section, raw_text, parse_error)
                if not placeholder:
                    raise
                chapter_json, placeholder_notes = placeholder
                parse_context.extend(placeholder_notes)
                placeholder_created = True

        # Auto-complete key fields before validation
        chapter_json.setdefault("chapterId", section.chapter_id)
        chapter_json.setdefault("anchor", section.slug)
        chapter_json.setdefault("title", section.title)
        chapter_json.setdefault("order", section.order)
        self._sanitize_chapter_blocks(chapter_json)

        valid, errors = self.validator.validate_chapter(chapter_json)
        if not valid and errors:
            repaired = self._attempt_llm_structural_repair(
                chapter_json,
                errors,
                raw_text=raw_text,
            )
            if repaired:
                chapter_json = repaired
                chapter_json.setdefault("chapterId", section.chapter_id)
                chapter_json.setdefault("anchor", section.slug)
                chapter_json.setdefault("title", section.title)
                chapter_json.setdefault("order", section.order)
                self._sanitize_chapter_blocks(chapter_json)
                valid, errors = self.validator.validate_chapter(chapter_json)
        content_error: ChapterContentError | None = None
        if valid and not placeholder_created:
            try:
                self._ensure_content_density(chapter_json)
            except ChapterContentError as exc:
                content_error = exc

        error_messages: List[str] = parse_context.copy()
        if not valid and errors:
            error_messages.extend(errors)
        if content_error:
            error_messages.append(str(content_error))

        self.storage.persist_chapter(
            run_dir,
            chapter_meta,
            chapter_json,
            errors=None if not error_messages else error_messages,
        )

        if not valid:
            raise ChapterValidationError(
                f"{section.title} chapter JSON validation failed: {'; '.join(errors[:5])}",
                errors=errors,
            )
        if content_error:
            raise content_error

        return chapter_json

    # ====== Internal Methods ======

    def _build_payload(self, section: TemplateSection, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct LLM input payload.

        Args:
            section: Current chapter to generate, providing title/number/outline.
            context: Global context dictionary containing topic, dual-engine reports, word budget plan, etc.

        Returns:
            dict: Payload that can be directly serialized into prompts, balancing chapter info and global constraints.
        """
        reports = context.get("reports", {})
        # Chapter word budget plan (from WordBudgetNode), used to guide word count and emphasis
        chapter_plan_map = context.get("chapter_directives", {})
        chapter_plan = chapter_plan_map.get(section.chapter_id) if chapter_plan_map else {}

        # Check from layout's tocPlan whether this chapter is allowed to use SWOT and PEST blocks
        allow_swot = self._get_chapter_swot_permission(section.chapter_id, context)
        allow_pest = self._get_chapter_pest_permission(section.chapter_id, context)

        payload = {
            "section": {
                "chapterId": section.chapter_id,
                "title": section.title,
                "slug": section.slug,
                "order": section.order,
                "number": section.number,
                "outline": section.outline,
            },
            "globalContext": {
                "query": context.get("query"),
                "templateName": context.get("template_name"),
                "themeTokens": context.get("theme_tokens", {}),
                "styleDirectives": context.get("style_directives", {}),
                # layout contains title/toc/hero etc., helping chapters maintain consistent visual tone
                "layout": context.get("layout"),
                "templateOverview": context.get("template_overview", {}),
            },
            "reports": {
                "query_engine": reports.get("query_engine", ""),
                "media_engine": reports.get("media_engine", ""),
            },
            "forumLogs": context.get("forum_logs", ""),
            "dataBundles": context.get("data_bundles", []),
            "constraints": {
                "language": (
                    (context.get("style_directives") or {}).get("language")
                    or "en-US"
                ),
                "languageRule": (context.get("style_directives") or {}).get(
                    "language_rule", ""
                ),
                "maxTokens": context.get("max_tokens", 4096),
                "allowedBlocks": ALLOWED_BLOCK_TYPES,
                "allowSwot": allow_swot,
                "allowPest": allow_pest,
                "styleHints": {
                    "expectWidgets": True,
                    "forceHeadingAnchors": True,
                    "allowInlineMix": True,
                },
            },
            "chapterPlan": chapter_plan,
            "wordPlan": context.get("word_plan"),
        }
        if chapter_plan:
            constraints = payload["constraints"]
            if chapter_plan.get("targetWords"):
                constraints["wordTarget"] = chapter_plan["targetWords"]
            if chapter_plan.get("minWords"):
                constraints["minWords"] = chapter_plan["minWords"]
            if chapter_plan.get("maxWords"):
                constraints["maxWords"] = chapter_plan["maxWords"]
            if chapter_plan.get("emphasis"):
                constraints["emphasis"] = chapter_plan["emphasis"]
            if chapter_plan.get("sections"):
                constraints["sectionBudgets"] = chapter_plan["sections"]
                payload["globalContext"]["sectionBudgets"] = chapter_plan["sections"]
        return payload

    def _get_chapter_swot_permission(self, chapter_id: str, context: Dict[str, Any]) -> bool:
        """
        Check from layout's tocPlan whether the specified chapter is allowed to use SWOT blocks.

        At most one chapter in the entire document is allowed to use SWOT blocks,
        marked by the allowSwot field in tocPlan during document design phase.

        Args:
            chapter_id: Current chapter ID.
            context: Global context dictionary.

        Returns:
            bool: True if the chapter is allowed to use SWOT blocks, False otherwise.
        """
        layout = context.get("layout")
        if not isinstance(layout, dict):
            return False

        toc_plan = layout.get("tocPlan")
        if not isinstance(toc_plan, list):
            return False

        for entry in toc_plan:
            if not isinstance(entry, dict):
                continue
            if entry.get("chapterId") == chapter_id:
                return bool(entry.get("allowSwot", False))

        return False

    def _get_chapter_pest_permission(self, chapter_id: str, context: Dict[str, Any]) -> bool:
        """
        Check from layout's tocPlan whether the specified chapter is allowed to use PEST blocks.

        At most one chapter in the entire document is allowed to use PEST blocks,
        marked by the allowPest field in tocPlan during document design phase.

        PEST blocks are used for macro-environment analysis:
        - Political factors
        - Economic factors
        - Social factors
        - Technological factors

        Args:
            chapter_id: Current chapter ID.
            context: Global context dictionary.

        Returns:
            bool: True if the chapter is allowed to use PEST blocks, False otherwise.
        """
        layout = context.get("layout")
        if not isinstance(layout, dict):
            return False

        toc_plan = layout.get("tocPlan")
        if not isinstance(toc_plan, list):
            return False

        for entry in toc_plan:
            if not isinstance(entry, dict):
                continue
            if entry.get("chapterId") == chapter_id:
                return bool(entry.get("allowPest", False))

        return False

    def _stream_llm(
        self,
        user_message: str,
        chapter_dir: Path,
        stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        section_meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Stream call LLM and write to raw file in real-time, while pushing deltas through callback.

        Args:
            user_message: Assembled user prompt.
            chapter_dir: Local cache directory for the chapter, used to store stream.raw.
            stream_callback: SSE streaming push callback function.
            section_meta: Attached chapter ID/title for callback payload.
            **kwargs: Pass-through parameters like temperature, top_p, etc.

        Returns:
            str: Raw text concatenated from all deltas.
        """
        chunks: List[str] = []
        with self.storage.capture_stream(chapter_dir) as stream_fp:
            stream = self.llm_client.stream_invoke(
                SYSTEM_PROMPT_CHAPTER_JSON,
                user_message,
                temperature=kwargs.get("temperature", 0.2),
                top_p=kwargs.get("top_p", 0.95),
            )
            for delta in stream:
                stream_fp.write(delta)
                chunks.append(delta)
                if stream_callback:
                    meta = section_meta or {}
                    try:
                        stream_callback(delta, meta)
                    except Exception as callback_error:  # pragma: no cover - log only, don't block main flow
                        logger.warning(f"Chapter streaming callback failed: {callback_error}")
        return "".join(chunks)

    def _attempt_cross_engine_json_rescue(
        self,
        section: TemplateSection,
        generation_payload: Dict[str, Any],
        raw_text: str,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Sequentially call Report/Forum/Media APIs to attempt repairing unparsable JSON.

        Returns:
            dict | None: Chapter JSON if repair successful, None otherwise.
        """
        if not self.fallback_llm_clients:
            return None
        if self._chapter_already_skipped(section):
            logger.info(f"[{run_id}] {section.title} already marked as placeholder, skipping cross-engine repair")
            return None
        section_payload = {
            "chapterId": section.chapter_id,
            "title": section.title,
            "slug": section.slug,
            "order": section.order,
            "number": section.number,
            "outline": section.outline,
        }
        repair_prompt = build_chapter_recovery_payload(
            section_payload,
            generation_payload,
            raw_text,
        )
        attempted_labels = self._rescue_attempted_labels.setdefault(section.chapter_id, set())
        for label, client in self.fallback_llm_clients:
            if label in attempted_labels:
                continue
            attempt_index = len(attempted_labels) + 1
            attempted_labels.add(label)
            logger.info(
                f"[{run_id}] Chapter {section.title} triggered {label} API JSON rescue (attempt #{attempt_index})"
            )
            try:
                response = client.invoke(
                    SYSTEM_PROMPT_CHAPTER_JSON_RECOVERY,
                    repair_prompt,
                    temperature=0.0,
                    top_p=0.05,
                )
            except Exception as exc:
                logger.warning(f"{label} JSON repair call failed: {exc}")
                continue
            if not response:
                continue
            try:
                repaired = self._parse_chapter(response)
            except Exception as exc:
                logger.warning(f"{label} JSON repair output still unparsable: {exc}")
                continue
            logger.warning(f"[{run_id}] {label} API repaired chapter JSON")
            self._archived_failed_json.pop(section.chapter_id, None)
            return repaired
        return None

    def _ensure_run_state(self, run_id: str):
        """Ensure repair state isolation for each report run to prevent previous task records from affecting new tasks."""
        if self._active_run_id == run_id:
            return
        self._active_run_id = run_id
        self._rescue_attempted_labels = {}
        self._skipped_placeholder_chapters = set()
        self._archived_failed_json = {}

    def _archive_failed_output(self, section: TemplateSection, raw_text: str):
        """Cache current chapter's raw error JSON for subsequent placeholder creation or manual review."""
        if not raw_text:
            return
        self._archived_failed_json[section.chapter_id] = raw_text

    def _get_archived_failed_output(self, section: TemplateSection) -> Optional[str]:
        """Get the most recent failed raw output for the chapter."""
        return self._archived_failed_json.get(section.chapter_id)

    def _mark_chapter_skipped(self, section: TemplateSection):
        """Record that this chapter has been downgraded to placeholder to avoid repeated cross-engine repair triggers."""
        self._skipped_placeholder_chapters.add(section.chapter_id)

    def _chapter_already_skipped(self, section: TemplateSection) -> bool:
        """Check if chapter has already been marked as placeholder."""
        return section.chapter_id in self._skipped_placeholder_chapters

    def _build_placeholder_chapter(
        self,
        section: TemplateSection,
        raw_text: str,
        parse_error: Exception,
    ) -> Optional[Tuple[Dict[str, Any], List[str]]]:
        """
        Construct renderable placeholder chapter when all repairs fail, and log files for subsequent troubleshooting.
        """
        snapshot = self._get_archived_failed_output(section) or raw_text
        log_ref = self._persist_error_payload(section, snapshot, parse_error)
        if not log_ref:
            logger.error(f"{section.title} chapter JSON completely corrupted and cannot write log")
            return None
        importance = "critical" if self._is_section_critical(section) else "standard"
        message = (
            f"LLM returned block parse error, see {log_ref['relativeFile']} entry {log_ref['entryId']} for details."
        )
        heading_block = {
            "type": "heading",
            "level": 2 if importance == "critical" else 3,
            "text": section.title,
            "anchor": section.slug,
        }
        callout_block = {
            "type": "callout",
            "tone": "danger" if importance == "critical" else "warning",
            "title": "LLM Block Parse Error",
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "text": message,
                        }
                    ],
                }
            ],
            "meta": {
                "errorLogRef": log_ref,
                "rawJsonPreview": (snapshot or "")[:2000],
                "errorMessage": message,
                "importance": importance,
            },
        }
        placeholder = {
            "chapterId": section.chapter_id,
            "title": section.title,
            "anchor": section.slug,
            "order": section.order,
            "blocks": [heading_block, callout_block],
            "errorPlaceholder": True,
        }
        errors = [
            f"{section.title} chapter JSON parsing failed, downgraded to placeholder. Reference: {log_ref['relativeFile']}#{log_ref['entryId']}"
        ]
        self._mark_chapter_skipped(section)
        return placeholder, errors

    def _parse_chapter(self, raw_text: str) -> Dict[str, Any]:
        """
        Clean LLM output and parse JSON.

        Args:
            raw_text: Raw LLM output (may contain ``` wrapping or extra notes).

        Returns:
            dict: Chapter JSON object, containing at least chapterId/title/blocks.

        Raises:
            ChapterJsonParseError: Still unable to parse valid JSON after multiple repair strategies.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if not cleaned:
            raise ChapterJsonParseError("LLM returned empty content", raw_text=raw_text)

        candidate_payloads = [cleaned]
        repaired = self._repair_llm_json(cleaned)
        if repaired != cleaned:
            candidate_payloads.append(repaired)

        data: Dict[str, Any] | None = None
        try:
            data = self._parse_with_candidates(candidate_payloads)
        except json.JSONDecodeError as exc:
            repaired_payload = self._attempt_json_repair(cleaned)
            if repaired_payload:
                candidate_payloads.append(repaired_payload)
                try:
                    data = self._parse_with_candidates(candidate_payloads[-1:])
                except json.JSONDecodeError:
                    data = None
            if data is None:
                try:
                    data = self._robust_parser.parse(
                        cleaned,
                        context_name="ChapterJSON",
                        expected_keys=["chapter", "blocks", "chapterId", "title"],
                    )
                except JSONParseError as robust_exc:
                    raise ChapterJsonParseError(
                        f"Chapter JSON parsing failed: {robust_exc}", raw_text=cleaned
                    ) from robust_exc

        if "chapter" in data and isinstance(data["chapter"], dict):
            return data["chapter"]
        if isinstance(data, dict) and all(
            key in data for key in ("chapterId", "title", "blocks")
        ):
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "chapter" in item and isinstance(item["chapter"], dict):
                        return item["chapter"]
                    if all(key in item for key in ("chapterId", "title", "blocks")):
                        return item
        raise ChapterJsonParseError("Chapter JSON missing chapter field or incomplete structure", raw_text=cleaned)

    def _persist_error_payload(
        self,
        section: TemplateSection,
        raw_text: str,
        parse_error: Exception,
    ) -> Optional[Dict[str, str]]:
        """Persist unparsable JSON text to disk for referencing specific files in HTML."""
        try:
            self._failed_block_counter += 1
            entry_id = f"E{self._failed_block_counter:04d}"
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            slug = section.slug or "section"
            filename = f"{timestamp}-{slug}-{entry_id}.json"
            file_path = self.error_log_dir / filename
            payload = {
                "chapterId": section.chapter_id,
                "title": section.title,
                "slug": section.slug,
                "order": section.order,
                "rawOutput": raw_text,
                "error": str(parse_error),
                "loggedAt": timestamp,
            }
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                relative_path = str(file_path.relative_to(Path.cwd()))
            except ValueError:
                relative_path = str(file_path)
            return {
                "file": str(file_path),
                "relativeFile": relative_path,
                "entryId": entry_id,
                "timestamp": timestamp,
            }
        except Exception as exc:
            logger.error(f"Failed to log chapter JSON error: {exc}")
            return None

    def _is_section_critical(self, section: TemplateSection) -> bool:
        """Determine if section affects table of contents based on depth/number, deciding prompt intensity."""
        if not section:
            return False
        if section.depth <= 2:
            return True
        number = section.number or ""
        if number and number.count(".") <= 1:
            return True
        return False

    def _repair_llm_json(self, text: str) -> str:
        """
        Handle common LLM errors (like ":= causing invalid JSON).

        Args:
            text: Raw chapter JSON text.

        Returns:
            str: Repaired text; returns original if no changes made.
        """
        repaired = text
        mutated = False

        new_text = self._COLON_EQUALS_PATTERN.sub(r"\1", repaired)
        if new_text != repaired:
            logger.warning('Detected ":=" characters in chapter JSON, auto-removed redundant "="')
            repaired = new_text
            mutated = True

        repaired, escaped = self._escape_in_string_controls(repaired)
        if escaped:
            logger.warning("Detected unescaped control characters in chapter JSON string, auto-converted to escape sequences")
            mutated = True

        repaired, balanced = self._balance_brackets(repaired)
        if balanced:
            logger.warning("Detected unbalanced brackets in chapter JSON, auto-fixed abnormal brackets")
            mutated = True

        repaired, commas_fixed = self._fix_missing_commas(repaired)
        if commas_fixed:
            logger.warning("Detected missing commas between chapter JSON objects/arrays, auto-completed")
            mutated = True

        return repaired if mutated else text

    def _escape_in_string_controls(self, text: str) -> Tuple[str, bool]:
        """
        Replace raw newlines/tabs/control characters in string literals with JSON-legal escape sequences.
        """
        if not text:
            return text, False

        result: List[str] = []
        in_string = False
        escaped = False
        mutated = False
        control_map = {"\n": "\\n", "\r": "\\n", "\t": "\\t"}

        for ch in text:
            if escaped:
                result.append(ch)
                escaped = False
                continue

            if ch == "\\":
                result.append(ch)
                escaped = True
                continue

            if ch == '"':
                result.append(ch)
                in_string = not in_string
                continue

            if in_string and ch in control_map:
                result.append(control_map[ch])
                mutated = True
                continue

            if in_string and ord(ch) < 0x20:
                result.append(f"\\u{ord(ch):04x}")
                mutated = True
                continue

            result.append(ch)

        return "".join(result), mutated

    def _fix_missing_commas(self, text: str) -> Tuple[str, bool]:
        """Auto-add commas when objects/arrays appear consecutively"""
        if not text:
            return text, False

        chars: List[str] = []
        mutated = False
        in_string = False
        escaped = False
        length = len(text)
        i = 0
        while i < length:
            ch = text[i]
            chars.append(ch)
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if ch == '"':
                in_string = not in_string
                i += 1
                continue
            if not in_string and ch in "}]":
                j = i + 1
                while j < length and text[j] in " \t\r\n":
                    j += 1
                if j < length:
                    next_ch = text[j]
                    if next_ch in "{[":
                        chars.append(",")
                        mutated = True
            i += 1
        return "".join(chars), mutated

    def _balance_brackets(self, text: str) -> Tuple[str, bool]:
        """Attempt to fix unbalanced bracket structures caused by LLM writing too many/few brackets"""
        if not text:
            return text, False

        result: List[str] = []
        stack: List[str] = []
        mutated = False
        in_string = False
        escaped = False

        opener_map = {"{": "}", "[": "]"}

        for ch in text:
            if escaped:
                result.append(ch)
                escaped = False
                continue

            if ch == "\\":
                result.append(ch)
                escaped = True
                continue

            if ch == '"':
                result.append(ch)
                in_string = not in_string
                continue

            if in_string:
                result.append(ch)
                continue

            if ch in "{[":
                stack.append(ch)
                result.append(ch)
                continue

            if ch in "}]":
                if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                    stack.pop()
                    result.append(ch)
                else:
                    mutated = True
                continue

            result.append(ch)

        while stack:
            opener = stack.pop()
            result.append(opener_map[opener])
            mutated = True

        return "".join(result), mutated

    def _attempt_json_repair(self, text: str) -> str | None:
        """Use optional json_repair library to further fix complex syntax errors"""
        if not _json_repair_fn:
            return None
        try:
            fixed = _json_repair_fn(text)
        except Exception as exc:  # pragma: no cover - library failure
            logger.warning(f"json_repair failed to fix chapter JSON: {exc}")
            return None
        if fixed == text:
            return None
        logger.warning("Auto-fixed chapter JSON syntax using json_repair")
        return fixed

    def _attempt_llm_structural_repair(
        self,
        chapter: Dict[str, Any],
        validation_errors: List[str],
        raw_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Delegate structurally erroneous chapters to LLM for fallback repair, maintaining same API settings as Report Engine."""
        if not validation_errors:
            return None
        payload = build_chapter_repair_prompt(chapter, validation_errors, raw_text)
        try:
            response = self.llm_client.invoke(
                SYSTEM_PROMPT_CHAPTER_JSON_REPAIR,
                payload,
                temperature=0.0,
                top_p=0.05,
            )
        except Exception as exc:  # pragma: no cover - network or API exceptions logged only
            logger.error(f"Chapter JSON LLM repair call failed: {exc}")
            return None
        if not response:
            return None
        try:
            repaired = self._parse_chapter(response)
        except Exception as exc:
            logger.error(f"Failed to parse chapter JSON after LLM repair: {exc}")
            return None
        logger.warning("Chapter JSON still non-compliant after multiple local repairs, successfully enabled LLM fallback repair")
        return repaired

    def _sanitize_chapter_blocks(self, chapter: Dict[str, Any]):
        """
        Fix common structural errors (e.g., list.items nested too deep).

        Args:
            chapter: Chapter JSON object, cleaned and normalized in place.
        """

        def walk(blocks: List[Dict[str, Any]] | None):
            """Recursively check and fix nested structures, ensuring each block is valid"""
            if not isinstance(blocks, list):
                return
            # First filter out abnormal blocks that are not dict types
            valid_indices = []
            for idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    # Try to convert string to paragraph
                    if isinstance(block, str) and block.strip():
                        blocks[idx] = self._as_paragraph_block(block)
                        valid_indices.append(idx)
                        logger.warning(f"walk: converted string block to paragraph")
                    elif isinstance(block, list):
                        # Try to extract valid dict from list
                        for item in block:
                            if isinstance(item, dict):
                                self._ensure_block_type(item)
                                blocks[idx] = item
                                valid_indices.append(idx)
                                logger.warning(f"walk: extracted dict block from list")
                                break
                        else:
                            logger.warning(f"walk: skipped invalid list block: {block}")
                    else:
                        logger.warning(f"walk: skipped invalid block (type: {type(block).__name__})")
                else:
                    valid_indices.append(idx)

            for idx in valid_indices:
                block = blocks[idx]
                if not isinstance(block, dict):
                    continue
                self._ensure_block_type(block)
                self._sanitize_block_content(block)
                block_type = block.get("type")
                if block_type == "list":
                    # Auto-fix listType: ensure it's a valid value
                    self._normalize_list_type(block)
                    items = block.get("items")
                    normalized = self._normalize_list_items(items)
                    if normalized:
                        block["items"] = normalized
                    for entry in block.get("items", []):
                        walk(entry)
                elif block_type in {"callout", "blockquote", "engineQuote"}:
                    walk(block.get("blocks"))
                elif block_type == "table":
                    for row in block.get("rows", []):
                        if not isinstance(row, dict):
                            continue
                        cells = row.get("cells") or []
                        for cell in cells:
                            if not isinstance(cell, dict):
                                continue
                            walk(cell.get("blocks"))
                elif block_type == "widget":
                    self._normalize_widget_block(block)
                else:
                    nested = block.get("blocks")
                    if isinstance(nested, list):
                        walk(nested)

        walk(chapter.get("blocks"))

        blocks = chapter.get("blocks")
        if isinstance(blocks, list):
            # Filter out all non-dict blocks before merging
            filtered_blocks = [b for b in blocks if isinstance(b, dict)]
            chapter["blocks"] = self._merge_fragment_sequences(filtered_blocks)

    def _ensure_content_density(self, chapter: Dict[str, Any]):
        """
        Validate chapter body content density.

        If blocks are missing, no valid blocks except headings, or body character count is below threshold,
        the chapter content is considered abnormal, triggering ChapterContentError for upstream retry.

        Args:
            chapter: Current chapter JSON.

        Raises:
            ChapterContentError: When body block count or character count does not meet minimum requirements.
        """
        blocks = chapter.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ChapterContentError(
                "Chapter missing body blocks, cannot output content",
                chapter=chapter,
                body_characters=0,
                narrative_characters=0,
                non_heading_blocks=0,
            )

        non_heading_blocks = [
            block
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") not in {"heading", "divider", "toc"}
        ]
        valid_block_count = len(non_heading_blocks)
        body_characters = self._count_body_characters(blocks)
        narrative_characters = self._count_narrative_characters(blocks)

        if (
            valid_block_count < self._MIN_NON_HEADING_BLOCKS
            or body_characters < self._MIN_BODY_CHARACTERS
            or narrative_characters < self._MIN_NARRATIVE_CHARACTERS
        ):
            raise ChapterContentError(
                f"{chapter.get('title') or 'This chapter'} insufficient body content: {valid_block_count} valid blocks, {body_characters} estimated characters, {narrative_characters} narrative characters",
                chapter=chapter,
                body_characters=body_characters,
                narrative_characters=narrative_characters,
                non_heading_blocks=valid_block_count,
            )

    def _count_body_characters(self, blocks: Any) -> int:
        """
        Recursively count body characters.

        - Ignore non-body types like heading/divider/widget;
        - Extract nested text from paragraph/list/table/callout structures;
        - Only used for coarse-grained length reasonableness judgment.

        Args:
            blocks: Chapter's blocks list or subtree.

        Returns:
            int: Estimated body character count.
        """

        def walk(node: Any) -> int:
            """Recursively traverse block tree and return character estimate, skipping non-body types"""
            if node is None:
                return 0
            if isinstance(node, list):
                return sum(walk(item) for item in node)
            if isinstance(node, str):
                return len(node.strip())
            if not isinstance(node, dict):
                return 0

            block_type = node.get("type")
            if block_type in {"heading", "divider", "toc", "widget"}:
                return 0

            if block_type == "paragraph":
                return self._estimate_paragraph_characters(node)

            if block_type == "list":
                total = 0
                for item in node.get("items", []):
                    total += walk(item)
                return total

            if block_type in {"blockquote", "callout", "engineQuote"}:
                return walk(node.get("blocks"))

            if block_type == "table":
                total = 0
                for row in node.get("rows", []):
                    cells = row.get("cells") or []
                    for cell in cells:
                        total += walk(cell.get("blocks"))
                return total

            nested = node.get("blocks")
            if isinstance(nested, list):
                return walk(nested)

            return len(self._extract_block_text(node).strip())

        return walk(blocks)

    def _count_narrative_characters(self, blocks: Any) -> int:
        """
        Count characters in narrative structures like paragraph/callout/list/blockquote/engineQuote,
        """

        def walk(node: Any) -> int:
            """Recursively traverse narrative nodes, ignoring charts/toc and other non-body structures"""
            if node is None:
                return 0
            if isinstance(node, list):
                return sum(walk(item) for item in node)
            if isinstance(node, str):
                return len(node.strip())
            if not isinstance(node, dict):
                return 0

            block_type = node.get("type")
            if block_type == "paragraph":
                return self._estimate_paragraph_characters(node)
            if block_type == "list":
                total = 0
                for item in node.get("items", []):
                    total += walk(item)
                return total
            if block_type in {"callout", "blockquote", "engineQuote"}:
                return walk(node.get("blocks"))

            # list items may be anonymous dict, traverse for compatibility
            if block_type is None:
                nested = node.get("blocks")
                if isinstance(nested, list):
                    return walk(nested)
            return 0

        return walk(blocks)

    def _estimate_paragraph_characters(self, block: Dict[str, Any]) -> int:
        """Extract paragraph text length, reused in various statistics."""
        inlines = block.get("inlines")
        if isinstance(inlines, list):
            total = 0
            for run in inlines:
                if isinstance(run, dict):
                    text = run.get("text")
                    if isinstance(text, str):
                        total += len(text.strip())
            return total
        text_value = block.get("text")
        if isinstance(text_value, str):
            return len(text_value.strip())
        return len(self._extract_block_text(block).strip())

    def _sanitize_block_content(self, block: Dict[str, Any]):
        """Perform fine-grained repair based on type, e.g., cleaning illegal inline marks in paragraph"""
        block_type = block.get("type")
        if block_type == "paragraph":
            self._normalize_paragraph_block(block)
        elif block_type == "table":
            self._sanitize_table_block(block)
        elif block_type == "engineQuote":
            self._sanitize_engine_quote_block(block)

    def _sanitize_table_block(self, block: Dict[str, Any]):
        """Ensure table rows/cells structure is valid and each cell contains at least one block"""
        raw_rows = block.get("rows")
        # First check if there is nested row structure problem (only 1 row but nested cells)
        if isinstance(raw_rows, list) and len(raw_rows) == 1:
            first_row = raw_rows[0]
            if isinstance(first_row, dict):
                cells = first_row.get("cells", [])
                # Check if nested structure exists
                has_nested = any(
                    isinstance(cell, dict) and "cells" in cell and "blocks" not in cell
                    for cell in cells
                    if isinstance(cell, dict)
                )
                if has_nested:
                    # Fix nested row structure
                    fixed_rows = self._fix_nested_rows_structure(raw_rows)
                    block["rows"] = fixed_rows
                    return
        # Normal case, use standard normalization
        rows = self._normalize_table_rows(raw_rows)
        block["rows"] = rows

    def _fix_nested_rows_structure(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix incorrectly nested table row structures.

        When LLM generates a table with only 1 row but all data is nested in cells,
        this method flattens all cells and reorganizes them into correct multi-row structure.

        Args:
            rows: Original table row array (should only have 1 row).

        Returns:
            List[Dict]: Fixed multi-row table structure.
        """
        if not rows or len(rows) != 1:
            return self._normalize_table_rows(rows)

        first_row = rows[0]
        original_cells = first_row.get("cells", [])

        # Recursively flatten all nested cells
        all_cells = self._flatten_all_cells_recursive(original_cells)

        if len(all_cells) <= 1:
            return self._normalize_table_rows(rows)

        # Helper function: get cell text
        def _get_cell_text(cell: Dict[str, Any]) -> str:
            blocks = cell.get("blocks", [])
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "paragraph":
                    inlines = block.get("inlines", [])
                    for inline in inlines:
                        if isinstance(inline, dict):
                            text = inline.get("text", "")
                            if text:
                                return str(text).strip()
            return ""

        def _is_placeholder_cell(cell: Dict[str, Any]) -> bool:
            """Determine if cell is a placeholder"""
            text = _get_cell_text(cell)
            return text in ("--", "-", "—", "——", "", "N/A", "n/a")

        def _is_header_cell(cell: Dict[str, Any]) -> bool:
            """Determine if cell looks like a header (usually has bold mark or typical header words)"""
            blocks = cell.get("blocks", [])
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "paragraph":
                    inlines = block.get("inlines", [])
                    for inline in inlines:
                        if isinstance(inline, dict):
                            marks = inline.get("marks", [])
                            if any(isinstance(m, dict) and m.get("type") == "bold" for m in marks):
                                return True
            # Also check typical header words
            text = _get_cell_text(cell)
            header_keywords = {
                "Time", "Date", "Name", "Type", "Status", "Quantity", "Amount", "Ratio", "Indicator",
                "Platform", "Channel", "Source", "Description", "Note", "Remark", "No.", "ID",
                "Event", "Key", "Data", "Support", "Reaction", "Market", "Sentiment", "Node",
                "Dimension", "Point", "Details", "Tag", "Impact", "Trend", "Weight", "Category",
                "Info", "Content", "Style", "Preference", "Main", "User", "Core", "Feature",
                "Classification", "Scope", "Object", "Item", "Phase", "Cycle", "Frequency", "Level",
            }
            return any(kw in text for kw in header_keywords) and len(text) <= 20

        # Filter out placeholder cells
        valid_cells = [c for c in all_cells if not _is_placeholder_cell(c)]

        if len(valid_cells) <= 1:
            return self._normalize_table_rows(rows)

        # Detect header column count: count consecutive header cells
        header_count = 0
        for cell in valid_cells:
            if _is_header_cell(cell):
                header_count += 1
            else:
                break

        # If no header detected, use heuristic method
        if header_count == 0:
            total = len(valid_cells)
            for possible_cols in [4, 5, 3, 6, 2]:
                if total % possible_cols == 0:
                    header_count = possible_cols
                    break
            else:
                # Try to find closest divisible column count
                for possible_cols in [4, 5, 3, 6, 2]:
                    remainder = total % possible_cols
                    if remainder <= 3:
                        header_count = possible_cols
                        break
                else:
                    # Cannot determine column count, use original data
                    return self._normalize_table_rows(rows)

        # Calculate valid cell count
        total = len(valid_cells)
        remainder = total % header_count
        if remainder > 0 and remainder <= 3:
            # Truncate excess tail cells
            valid_cells = valid_cells[:total - remainder]
        elif remainder > 3:
            # Remainder too large, possible column count detection error
            return self._normalize_table_rows(rows)

        # Reorganize into multiple rows
        fixed_rows: List[Dict[str, Any]] = []
        for i in range(0, len(valid_cells), header_count):
            row_cells = valid_cells[i:i + header_count]
            # Mark first row as header
            if i == 0:
                for cell in row_cells:
                    cell["header"] = True
            fixed_rows.append({"cells": row_cells})

        return fixed_rows if fixed_rows else self._normalize_table_rows(rows)

    def _flatten_all_cells_recursive(self, cells: List[Any]) -> List[Dict[str, Any]]:
        """
        Recursively flatten all nested cell structures.

        Args:
            cells: Cell array that may contain nested structures.

        Returns:
            List[Dict]: Flattened cell array, each cell has blocks.
        """
        if not cells:
            return []

        flattened: List[Dict[str, Any]] = []

        def _extract_cells(cell_or_list: Any) -> None:
            if not isinstance(cell_or_list, dict):
                if isinstance(cell_or_list, (str, int, float)):
                    flattened.append({"blocks": [self._as_paragraph_block(str(cell_or_list))]})
                return

            # If current object has blocks, it's a valid cell
            if "blocks" in cell_or_list:
                # Create cell copy, remove nested cells
                clean_cell = {
                    k: v for k, v in cell_or_list.items()
                    if k != "cells"
                }
                # Ensure blocks are valid
                blocks = clean_cell.get("blocks")
                if not isinstance(blocks, list) or not blocks:
                    clean_cell["blocks"] = [self._as_paragraph_block("")]
                flattened.append(clean_cell)

            # If current object has nested cells, process recursively
            nested_cells = cell_or_list.get("cells")
            if isinstance(nested_cells, list):
                for nested_cell in nested_cells:
                    _extract_cells(nested_cell)

        for cell in cells:
            _extract_cells(cell)

        return flattened

    def _sanitize_engine_quote_block(self, block: Dict[str, Any]):
        """engineQuote is only used for single Agent speech, only allows paragraph internally and title must lock Agent name"""
        engine_raw = block.get("engine")
        engine = engine_raw.lower() if isinstance(engine_raw, str) else None
        if engine not in ENGINE_AGENT_TITLES:
            engine = "query"
        block["engine"] = engine
        block["title"] = ENGINE_AGENT_TITLES[engine]
        allowed_marks = {"bold", "italic"}
        raw_blocks = block.get("blocks")
        candidates = raw_blocks if isinstance(raw_blocks, list) else ([raw_blocks] if raw_blocks else [])
        sanitized_blocks: List[Dict[str, Any]] = []

        for item in candidates:
            if isinstance(item, dict) and item.get("type") == "paragraph":
                para = dict(item)
            else:
                text = self._extract_block_text(item) if isinstance(item, dict) else (item or "")
                para = self._as_paragraph_block(str(text))

            inlines = para.get("inlines")
            if not isinstance(inlines, list) or not inlines:
                inlines = [self._as_inline_run(self._extract_block_text(para))]

            cleaned_inlines: List[Dict[str, Any]] = []
            for run in inlines:
                if isinstance(run, dict):
                    text_val = run.get("text")
                    text_str = text_val if isinstance(text_val, str) else ("" if text_val is None else str(text_val))
                    marks_raw = run.get("marks") if isinstance(run.get("marks"), list) else []
                    marks_filtered: List[Dict[str, Any]] = []
                    for mark in marks_raw:
                        if not isinstance(mark, dict):
                            continue
                        mark_type = mark.get("type")
                        if mark_type in allowed_marks:
                            marks_filtered.append({"type": mark_type})
                    cleaned_inlines.append({"text": text_str, "marks": marks_filtered})
                else:
                    cleaned_inlines.append(self._as_inline_run(str(run)))

            if not cleaned_inlines:
                cleaned_inlines.append(self._as_inline_run(""))
            para["inlines"] = cleaned_inlines
            para["type"] = "paragraph"
            para.pop("blocks", None)
            sanitized_blocks.append(para)

        if not sanitized_blocks:
            sanitized_blocks.append(self._as_paragraph_block(""))
        block["blocks"] = sanitized_blocks

    def _normalize_table_rows(self, rows: Any) -> List[Dict[str, Any]]:
        """Ensure rows is always a list composed of row objects"""
        if rows is None:
            rows_iterable: List[Any] = []
        elif isinstance(rows, list):
            rows_iterable = rows
        else:
            rows_iterable = [rows]

        normalized_rows: List[Dict[str, Any]] = []
        for row in rows_iterable:
            sanitized_row = self._normalize_table_row(row)
            if sanitized_row:
                normalized_rows.append(sanitized_row)

        if not normalized_rows:
            normalized_rows.append({"cells": [self._build_default_table_cell()]})
        return normalized_rows

    def _normalize_table_row(self, row: Any) -> Dict[str, Any] | None:
        """Unify various row expressions into {'cells': [...]} structure"""
        if row is None:
            return None
        if isinstance(row, dict):
            result = dict(row)
            cells_value = result.get("cells")
        else:
            result = {}
            cells_value = row

        cells = self._normalize_table_cells(cells_value)
        if not cells:
            cells = [self._build_default_table_cell()]
        result["cells"] = cells
        return result

    def _normalize_table_cells(self, cells: Any) -> List[Dict[str, Any]]:
        """Clean cells, ensure each cell has non-empty blocks"""
        if cells is None:
            cell_entries: List[Any] = []
        elif isinstance(cells, list):
            cell_entries = cells
        else:
            cell_entries = [cells]

        normalized_cells: List[Dict[str, Any]] = []
        for cell in cell_entries:
            # Detect incorrectly nested cells structure: has cells but no blocks
            # Need to flatten into multiple independent cells
            if isinstance(cell, dict) and "cells" in cell and "blocks" not in cell:
                flattened = self._flatten_all_nested_cells(cell)
                normalized_cells.extend(flattened)
            else:
                sanitized = self._normalize_table_cell(cell)
                if sanitized:
                    normalized_cells.append(sanitized)

        return normalized_cells

    def _flatten_all_nested_cells(self, cell: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten incorrectly nested cells structure, return all flattened cells.

        LLM sometimes generates error structures like:
        { "cells": [
            { "blocks": [...] },
            { "cells": [
                { "blocks": [...] },
                { "cells": [...] }
              ]
            }
          ]
        }

        Should be flattened to independent cells list.
        """
        nested_cells = cell.get("cells")
        if not isinstance(nested_cells, list) or not nested_cells:
            return [{"blocks": [self._as_paragraph_block("")]}]

        result: List[Dict[str, Any]] = []
        for nested in nested_cells:
            if isinstance(nested, dict):
                if "blocks" in nested and "cells" not in nested:
                    # Normal cell, normalize and add directly
                    sanitized = self._normalize_table_cell(nested)
                    if sanitized:
                        result.append(sanitized)
                elif "cells" in nested and "blocks" not in nested:
                    # Continue recursively flattening nested cells
                    result.extend(self._flatten_all_nested_cells(nested))
                else:
                    # Other cases, try to normalize
                    sanitized = self._normalize_table_cell(nested)
                    if sanitized:
                        result.append(sanitized)
            elif isinstance(nested, (str, int, float)):
                result.append({"blocks": [self._as_paragraph_block(str(nested))]})

        return result if result else [{"blocks": [self._as_paragraph_block("")]}]

    def _normalize_table_cell(self, cell: Any) -> Dict[str, Any] | None:
        """Normalize various cell writing styles to schema-recognized form"""
        if cell is None:
            return {"blocks": [self._as_paragraph_block("")]}

        if isinstance(cell, dict):
            # Detect incorrectly nested cells structure: has cells but no blocks
            # This is a common LLM error, nesting sibling cells into cells array
            if "cells" in cell and "blocks" not in cell:
                # Flatten nested cells and return first valid cell
                # Note: remaining nested cells will be processed in _normalize_table_cells
                return self._flatten_nested_cell(cell)

            normalized = dict(cell)
            blocks = self._coerce_cell_blocks(normalized.get("blocks"), normalized)
        elif isinstance(cell, list):
            normalized = {}
            blocks = self._coerce_cell_blocks(cell, None)
        elif isinstance(cell, (str, int, float)):
            normalized = {}
            blocks = [self._as_paragraph_block(str(cell))]
        else:
            normalized = {}
            blocks = [self._as_paragraph_block(str(cell))]

        normalized["blocks"] = blocks or [self._as_paragraph_block("")]
        return normalized

    def _flatten_nested_cell(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten incorrectly nested cell structure.

        LLM sometimes generates error structures like:
        { "cells": [ { "blocks": [...] }, { "cells": [...] } ] }

        Should return first valid cell content.
        """
        nested_cells = cell.get("cells")
        if not isinstance(nested_cells, list) or not nested_cells:
            # No valid nested content, return empty cell
            return {"blocks": [self._as_paragraph_block("")]}

        # Recursively find first valid cell containing blocks
        for nested in nested_cells:
            if isinstance(nested, dict):
                if "blocks" in nested:
                    # Found valid cell, normalize recursively
                    return self._normalize_table_cell(nested)
                elif "cells" in nested:
                    # Continue recursively flattening
                    result = self._flatten_nested_cell(nested)
                    if result:
                        return result

        # No valid content found, try to extract text from first nested element
        first_nested = nested_cells[0]
        if isinstance(first_nested, dict):
            text = self._extract_block_text(first_nested)
            return {"blocks": [self._as_paragraph_block(text or "")]}

        return {"blocks": [self._as_paragraph_block("")]}

    def _coerce_cell_blocks(
        self, blocks: Any, source: Dict[str, Any] | None
    ) -> List[Dict[str, Any]]:
        """Force cell.blocks field conversion to valid block array"""
        if isinstance(blocks, list):
            entries = blocks
        elif blocks is None:
            entries = []
        else:
            entries = [blocks]

        normalized_blocks: List[Dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, dict):
                normalized_blocks.append(entry)
            elif isinstance(entry, list):
                normalized_blocks.extend(self._coerce_cell_blocks(entry, None))
            elif isinstance(entry, (str, int, float)):
                normalized_blocks.append(self._as_paragraph_block(str(entry)))
            elif entry is None:
                continue
            else:
                normalized_blocks.append(self._as_paragraph_block(str(entry)))

        if normalized_blocks:
            return normalized_blocks

        text_hint = ""
        if isinstance(source, dict):
            text_hint = self._extract_block_text(source).strip()
        return [self._as_paragraph_block(text_hint or "--")]

    def _build_default_table_cell(self) -> Dict[str, Any]:
        """Generate a minimal renderable blank cell"""
        return {"blocks": [self._as_paragraph_block("--")]}

    def _normalize_paragraph_block(self, block: Dict[str, Any]):
        """Unify paragraph inlines, remove illegal marks"""
        inlines = block.get("inlines")
        normalized_runs: List[Dict[str, Any]] = []
        if isinstance(inlines, list) and inlines:
            for run in inlines:
                normalized_runs.extend(self._coerce_inline_run(run))
        else:
            normalized_runs = [self._as_inline_run(self._extract_block_text(block))]
        if not normalized_runs:
            normalized_runs = [self._as_inline_run("")]
        block["inlines"] = self._strip_inline_artifacts(normalized_runs)

    def _strip_inline_artifacts(self, inlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove JSON sentinel text mistakenly written by LLM, prevent rendering garbage like `{\"type\": \"\"}`"""
        cleaned: List[Dict[str, Any]] = []
        for run in inlines or []:
            if not isinstance(run, dict):
                continue
            text = run.get("text")
            if isinstance(text, str):
                stripped = text.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and set(payload.keys()).issubset({"type", "value"}):
                        continue
            cleaned.append(run)
        return cleaned or [self._as_inline_run("")]

    def _merge_fragment_sequences(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge sentence fragments split by LLM into multiple segments, avoid isolated <p> in HTML"""
        if not isinstance(blocks, list):
            return blocks

        merged: List[Dict[str, Any]] = []
        fragment_buffer: List[Dict[str, Any]] = []

        def flush_buffer():
            """Write current fragment buffer to merged list, merge to single paragraph when necessary"""
            nonlocal fragment_buffer
            if not fragment_buffer:
                return
            if len(fragment_buffer) == 1:
                merged.append(fragment_buffer[0])
            else:
                merged.append(self._combine_paragraph_fragments(fragment_buffer))
            fragment_buffer = []

        for block in blocks:
            # Type check: skip abnormal blocks that are not dict type, avoid AttributeError
            if not isinstance(block, dict):
                # Try to convert non-dict types to paragraph
                if isinstance(block, str) and block.strip():
                    converted = self._as_paragraph_block(block)
                    logger.warning(f"Detected non-dict block (string), converted to paragraph: {block[:50]}...")
                    merged.append(converted)
                elif isinstance(block, list):
                    # List-type block may be LLM output error, try to extract valid content
                    logger.warning(f"Detected list-type block, trying to extract valid content: {block}")
                    for item in block:
                        if isinstance(item, dict):
                            self._ensure_block_type(item)
                            merged.append(self._merge_nested_fragments(item))
                        elif isinstance(item, str) and item.strip():
                            merged.append(self._as_paragraph_block(item))
                else:
                    logger.warning(f"Skipped invalid block (type: {type(block).__name__}): {block}")
                continue
            if self._is_paragraph_fragment(block):
                fragment_buffer.append(block)
                continue
            flush_buffer()
            merged.append(self._merge_nested_fragments(block))

        flush_buffer()
        return merged

    def _merge_nested_fragments(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively process fragment merging for nested structures (callout/blockquote/engineQuote/list/table)"""
        # Type check: ensure block is dict type
        if not isinstance(block, dict):
            # Try to convert non-dict types to paragraph
            if isinstance(block, str) and block.strip():
                logger.warning("_merge_nested_fragments received string type, converted to paragraph")
                return self._as_paragraph_block(block)
            elif isinstance(block, list):
                # Try to extract first valid dict from list
                for item in block:
                    if isinstance(item, dict):
                        self._ensure_block_type(item)
                        return self._merge_nested_fragments(item)
                logger.warning("_merge_nested_fragments received invalid list, returning empty paragraph")
                return self._as_paragraph_block("")
            else:
                logger.warning(f"_merge_nested_fragments received invalid type ({type(block).__name__}), returning empty paragraph")
                return self._as_paragraph_block("")

        block_type = block.get("type")
        if block_type in {"callout", "blockquote", "engineQuote"}:
            nested = block.get("blocks")
            if isinstance(nested, list):
                block["blocks"] = self._merge_fragment_sequences(nested)
        elif block_type == "list":
            items = block.get("items")
            if isinstance(items, list):
                for entry in items:
                    if isinstance(entry, list):
                        merged_entry = self._merge_fragment_sequences(entry)
                        entry[:] = merged_entry
        elif block_type == "table":
            for row in block.get("rows", []):
                if not isinstance(row, dict):
                    continue
                cells = row.get("cells") or []
                for cell in cells:
                    if not isinstance(cell, dict):
                        continue
                    nested_blocks = cell.get("blocks")
                    if isinstance(nested_blocks, list):
                        cell["blocks"] = self._merge_fragment_sequences(nested_blocks)
        return block

    def _combine_paragraph_fragments(self, fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple sentence fragments into single paragraph block"""
        template = dict(fragments[0])
        combined_inlines: List[Dict[str, Any]] = []
        for fragment in fragments:
            runs = fragment.get("inlines")
            if isinstance(runs, list) and runs:
                combined_inlines.extend(runs)
            else:
                fallback_text = self._extract_block_text(fragment)
                combined_inlines.append(self._as_inline_run(fallback_text))
        if not combined_inlines:
            combined_inlines.append(self._as_inline_run(""))
        template["inlines"] = combined_inlines
        return template

    def _is_paragraph_fragment(self, block: Dict[str, Any]) -> bool:
        """Determine if paragraph is a short fragment incorrectly split"""
        if not isinstance(block, dict) or block.get("type") != "paragraph":
            return False
        inlines = block.get("inlines")
        text = ""
        has_marks = False
        if isinstance(inlines, list) and inlines:
            parts: List[str] = []
            for run in inlines:
                if not isinstance(run, dict):
                    continue
                parts.append(str(run.get("text") or ""))
                marks = run.get("marks")
                if isinstance(marks, list) and any(marks):
                    has_marks = True
            text = "".join(parts)
        else:
            text = self._extract_block_text(block)
        stripped = (text or "").strip()
        if not stripped:
            return True
        if has_marks:
            return False
        if "\n" in stripped:
            return False

        short_limit = self._PARAGRAPH_FRAGMENT_MAX_CHARS
        long_limit = getattr(
            self,
            "_PARAGRAPH_FRAGMENT_NO_TERMINATOR_MAX_CHARS",
            short_limit * 3,
        )

        if stripped[-1] in self._TERMINATION_PUNCTUATION:
            return len(stripped) <= short_limit

        if len(stripped) > long_limit:
            return False
        return True

    def _coerce_inline_run(self, run: Any) -> List[Dict[str, Any]]:
        """Normalize arbitrary inline writing to valid run"""
        if isinstance(run, dict):
            normalized_run = dict(run)
            text = normalized_run.get("text")
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            marks = normalized_run.get("marks")
            sanitized_marks, extra_text = self._sanitize_inline_marks(marks)
            normalized_run["marks"] = sanitized_marks
            normalized_run["text"] = (text or "") + extra_text
            return [normalized_run]
        if isinstance(run, str):
            return [self._as_inline_run(run)]
        if isinstance(run, (int, float)):
            return [self._as_inline_run(str(run))]
        if isinstance(run, list):
            normalized: List[Dict[str, Any]] = []
            for item in run:
                normalized.extend(self._coerce_inline_run(item))
            return normalized
        return [self._as_inline_run("" if run is None else str(run))]

    def _sanitize_inline_marks(self, marks: Any) -> Tuple[List[Dict[str, Any]], str]:
        """Filter illegal marks and convert break-type control characters to text"""
        text_suffix = ""
        if marks is None:
            return [], text_suffix
        mark_list = marks if isinstance(marks, list) else [marks]
        sanitized: List[Dict[str, Any]] = []
        for mark in mark_list:
            normalized_mark, extra_text = self._normalize_inline_mark(mark)
            if normalized_mark:
                sanitized.append(normalized_mark)
            if extra_text:
                text_suffix += extra_text
        return sanitized, text_suffix

    def _normalize_inline_mark(self, mark: Any) -> Tuple[Dict[str, Any] | None, str]:
        """Perform compatibility mapping for single mark, or convert to text when necessary"""
        if not isinstance(mark, dict):
            return None, ""
        canonical_type = self._canonical_inline_mark_type(mark.get("type"))
        if canonical_type == self._LINE_BREAK_SENTINEL:
            return None, "\n"
        if canonical_type in ALLOWED_INLINE_MARKS:
            normalized = dict(mark)
            normalized["type"] = canonical_type
            return normalized, ""
        return None, ""

    def _canonical_inline_mark_type(self, mark_type: Any) -> str | None:
        """Map mark type to Schema-supported values"""
        if not isinstance(mark_type, str):
            return None
        normalized = mark_type.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        if lowered in {"break", "linebreak", "br"}:
            return self._LINE_BREAK_SENTINEL
        return self._INLINE_MARK_ALIASES.get(lowered, lowered)

    def _extract_block_text(self, block: Dict[str, Any]) -> str:
        """Prioritize extracting fallback text from text/content and other fields"""
        for key in ("text", "content", "value", "title"):
            value = block.get(key)
            if isinstance(value, str):
                return value
            if value is not None:
                return str(value)
        return ""

    # Valid listType values
    _ALLOWED_LIST_TYPES = {"ordered", "bullet", "task"}
    # listType alias mappings
    _LIST_TYPE_ALIASES = {
        "unordered": "bullet",
        "ul": "bullet",
        "ol": "ordered",
        "numbered": "ordered",
        "checkbox": "task",
        "check": "task",
        "todo": "task",
    }

    def _normalize_list_type(self, block: Dict[str, Any]):
        """
        Ensure list block's listType is valid.

        If listType is missing or invalid, auto-fix to bullet.
        """
        list_type = block.get("listType")
        if list_type in self._ALLOWED_LIST_TYPES:
            return
        # Try alias mapping
        if isinstance(list_type, str):
            lowered = list_type.strip().lower()
            if lowered in self._LIST_TYPE_ALIASES:
                block["listType"] = self._LIST_TYPE_ALIASES[lowered]
                logger.warning(f"Mapped listType '{list_type}' to '{block['listType']}'")
                return
            if lowered in self._ALLOWED_LIST_TYPES:
                block["listType"] = lowered
                return
        # Unrecognized, default to bullet
        logger.warning(f"Detected invalid listType: {list_type}, fixed to bullet")
        block["listType"] = "bullet"

    def _normalize_list_items(self, items: Any) -> List[List[Dict[str, Any]]]:
        """Ensure list block's items follow [[block, block], ...] structure"""
        if not isinstance(items, list):
            return []
        normalized: List[List[Dict[str, Any]]] = []
        for item in items:
            normalized.extend(self._coerce_list_item(item))
        return [entry for entry in normalized if entry]

    def _coerce_list_item(self, item: Any) -> List[List[Dict[str, Any]]]:
        """Unify various nested writing styles to block arrays"""
        result: List[List[Dict[str, Any]]] = []
        if isinstance(item, dict):
            self._ensure_block_type(item)
            result.append([item])
            return result
        if isinstance(item, list):
            dicts = [elem for elem in item if isinstance(elem, dict)]
            if dicts:
                for elem in dicts:
                    self._ensure_block_type(elem)
                result.append(dicts)
            for elem in item:
                if isinstance(elem, list):
                    result.extend(self._coerce_list_item(elem))
                elif isinstance(elem, dict):
                    continue
                elif isinstance(elem, str):
                    result.append([self._as_paragraph_block(elem)])
                elif isinstance(elem, (int, float)):
                    result.append([self._as_paragraph_block(str(elem))])
        elif isinstance(item, str):
            result.append([self._as_paragraph_block(item)])
        elif isinstance(item, (int, float)):
            result.append([self._as_paragraph_block(str(item))])
        return result

    def _normalize_widget_block(self, block: Dict[str, Any]):
        """Ensure widget has top-level data or dataRef"""
        has_data = block.get("data") is not None or block.get("dataRef") is not None
        if has_data:
            return
        props = block.get("props")
        if isinstance(props, dict) and "data" in props:
            block["data"] = props.pop("data")
            return
        block["data"] = {"labels": [], "datasets": []}

    def _ensure_block_type(self, block: Dict[str, Any]):
        """If block lacks valid type, downgrade to paragraph"""
        block_type = block.get("type")
        if isinstance(block_type, str) and block_type in ALLOWED_BLOCK_TYPES:
            return
        text = ""
        for key in ("text", "content", "title"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            try:
                text = json.dumps(block, ensure_ascii=False)
            except Exception:
                text = str(block)
        block.clear()
        block["type"] = "paragraph"
        block["inlines"] = [self._as_inline_run(text)]

    @staticmethod
    def _as_paragraph_block(text: str) -> Dict[str, Any]:
        """Quickly wrap string into paragraph block for unified processing"""
        return {
            "type": "paragraph",
            "inlines": [ChapterGenerationNode._as_inline_run(text)],
        }

    @staticmethod
    def _as_inline_run(text: str) -> Dict[str, Any]:
        """Construct base inline run, ensure marks field exists"""
        return {"text": text or "", "marks": []}

    @staticmethod
    def _parse_with_candidates(payloads: List[str]) -> Dict[str, Any]:
        """Try multiple payloads in order until parsing succeeds"""
        last_exc: json.JSONDecodeError | None = None
        for payload in payloads:
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc


__all__ = [
    "ChapterGenerationNode",
    "ChapterJsonParseError",
    "ChapterContentError",
    "ChapterValidationError",
]
