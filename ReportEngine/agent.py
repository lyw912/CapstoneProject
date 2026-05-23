"""
Report Agent main class.

This module orchestrates template selection, layout design, chapter generation,
IR stitching and HTML rendering sub-processes, serving as the central dispatcher
for the Report Engine. Core responsibilities include:
1. Managing input data and state, coordinating Media/Query analysis engines, forum logs and templates;
2. Driving the node sequence: template selection → layout generation → word budgeting → chapter writing → stitching and rendering;
3. Handling error fallbacks, streaming event distribution, persistence and final deliverable saving.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple

from loguru import logger

from .core import (
    ChapterStorage,
    DocumentComposer,
    TemplateSection,
    parse_template_sections,
)
from .ir import IRValidator
from .llms import LLMClient
from .nodes import (
    TemplateSelectionNode,
    ChapterGenerationNode,
    ChapterJsonParseError,
    ChapterContentError,
    ChapterValidationError,
    DocumentLayoutNode,
    WordBudgetNode,
)
from .renderers import HTMLRenderer
from .state import ReportState
from .utils.config import settings, Settings
from .prompts.prompts import ENGLISH_REPORT_LANGUAGE_RULE


class StageOutputFormatError(ValueError):
    """Controlled exception raised when stage output structure does not match expectations."""


class FileCountBaseline:
    """
    File count baseline manager.

    This utility is used for:
    - Recording the number of Markdown files exported by Media/Query engines at task startup;
    - Quickly detecting new report arrivals during subsequent polling;
    - Providing the Flask layer with a basis for determining“whether inputs are ready”.
    """
    
    def __init__(self):
        """
        On initialization, prefer reading existing baseline snapshots.

        If `logs/report_baseline.json` does not exist, an empty snapshot will be created automatically,
        so that subsequent `initialize_baseline` can write the real baseline on first run.
        """
        self.baseline_file = 'logs/report_baseline.json'
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, int]:
        """
        Load baseline data.

        - Parse JSON directly when snapshot file exists;
        - Catch all loading exceptions and return empty dict to keep caller logic simple.
        """
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load baseline data: {e}")
        return {}
    
    def _save_baseline(self):
        """
        Write current baseline to disk.

        Uses `ensure_ascii=False` + indentation format for human readability;
        automatically creates target directory if missing.
        """
        try:
            os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(self.baseline_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Failed to save baseline data: {e}")
    
    def initialize_baseline(self, directories: Dict[str, str]) -> Dict[str, int]:
        """
        Initialize file count baseline.

        Traverse each engine directory and count `.md` files, persisting the result
        as the initial baseline. Subsequent `check_new_files` will compare against this.
        """
        current_counts = {}
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                current_counts[engine] = len(md_files)
            else:
                current_counts[engine] = 0
        
        # Save baseline data
        self.baseline_data = current_counts.copy()
        self._save_baseline()
        
        logger.info(f"File count baseline initialized: {current_counts}")
        return current_counts
    
    def check_new_files(self, directories: Dict[str, str]) -> Dict[str, Any]:
        """
        Check for new files.

        Compare current directory file counts with baseline:
        - Count new additions and determine if all engines are ready;
        - Return detailed counts, missing list for Web layer to display to users.
        """
        current_counts = {}
        new_files_found = {}
        all_have_new = True
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                current_counts[engine] = len(md_files)
                baseline_count = self.baseline_data.get(engine, 0)
                
                if current_counts[engine] > baseline_count:
                    new_files_found[engine] = current_counts[engine] - baseline_count
                else:
                    new_files_found[engine] = 0
                    all_have_new = False
            else:
                current_counts[engine] = 0
                new_files_found[engine] = 0
                all_have_new = False
        
        return {
            'ready': all_have_new,
            'baseline_counts': self.baseline_data,
            'current_counts': current_counts,
            'new_files_found': new_files_found,
            'missing_engines': [engine for engine, count in new_files_found.items() if count == 0]
        }
    
    def get_latest_files(self, directories: Dict[str, str]) -> Dict[str, str]:
        """
        Get the latest file in each directory.

        Find the most recently written Markdown via `os.path.getmtime`,
        ensuring the generation pipeline always uses the latest dual-engine reports.
        """
        latest_files = {}
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                if md_files:
                    latest_file = max(md_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
                    latest_files[engine] = os.path.join(directory, latest_file)
        
        return latest_files


class ReportAgent:
    """
    Report Agent main class.

    Responsible for integrating:
    - LLM client and its four upper-level inference nodes;
    - Chapter storage, IR stitching, renderer and other output pipelines;
    - State management, logging, input/output validation and persistence.
    """
    _CONTENT_SPARSE_MIN_ATTEMPTS = 3
    _CONTENT_SPARSE_WARNING_TEXT = "The LLM-generated content for this chapter may be too short; consider re-running the program if necessary."
    _STRUCTURAL_RETRY_ATTEMPTS = 2
    
    def __init__(self, config: Optional[Settings] = None):
        """
        Initialize Report Agent.

        Args:
            config: Configuration object, auto-loaded if not provided

        Steps overview:
            1. Parse configuration and connect core components like logging/LLM/renderer;
            2. Construct four inference nodes (template, layout, word budget, chapter);
            3. Initialize file baseline and chapter output directory;
            4. Build serializable state container for external service queries.
        """
        # Load configuration
        self.config = config or settings
        
        # Initialize file baseline manager
        self.file_baseline = FileCountBaseline()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        self.json_rescue_clients = self._initialize_rescue_llms()
        
        # Initialize chapter-level storage/validation/renderer components
        self.chapter_storage = ChapterStorage(self.config.CHAPTER_OUTPUT_DIR)
        self.document_composer = DocumentComposer()
        self.validator = IRValidator()
        self.renderer = HTMLRenderer()
        
        # Initialize nodes
        self._initialize_nodes()
        
        # Initialize file count baseline
        self._initialize_file_baseline()
        
        # State
        self.state = ReportState()
        
        # Ensure output directories exist
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.DOCUMENT_IR_OUTPUT_DIR, exist_ok=True)
        
        logger.info("Report Agent initialized")
        logger.info(f"Using LLM: {self.llm_client.get_model_info()}")
        
    def _setup_logging(self):
        """
        Setup logging.

        - Ensure log directory exists;
        - Use dedicated loguru sink to write Report Engine specific log files,
          avoiding confusion with other subsystems.
        - [FIX] Configure real-time log writing, disable buffering to ensure frontend sees logs immediately
        - [FIX] Prevent duplicate handler additions
        """
        # Ensure log directory exists
        log_dir = os.path.dirname(self.config.LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)

        def _exclude_other_engines(record):
            """
            Filter out logs from other engines (Media/Query/Forum), keep all other logs.

            Use path matching primarily, fall back to module name when path is unavailable.
            """
            excluded_keywords = ("MediaEngine", "QueryEngine", "ForumEngine")
            try:
                file_path = record["file"].path
                if any(keyword in file_path for keyword in excluded_keywords):
                    return False
            except Exception:
                pass

            try:
                module_name = record.get("module", "")
                if isinstance(module_name, str):
                    lowered = module_name.lower()
                    if any(keyword.lower() in lowered for keyword in excluded_keywords):
                        return False
            except Exception:
                pass

            return True

        # [FIX] Check if handler for this file already exists to avoid duplicates
        # loguru deduplicates automatically, but explicit check is safer
        log_file_path = str(Path(self.config.LOG_FILE).resolve())

        # Check existing handlers
        handler_exists = False
        for handler_id, handler_config in logger._core.handlers.items():
            if hasattr(handler_config, 'sink'):
                sink = handler_config.sink
                # Check if file sink with same path
                if hasattr(sink, '_name') and sink._name == log_file_path:
                    handler_exists = True
                    logger.debug(f"Log handler already exists, skipping: {log_file_path}")
                    break

        if not handler_exists:
            # [FIX] Create dedicated logger with real-time writing config
            # - enqueue=False: Disable async queue for immediate writes
            # - buffering=1: Line buffering, flush each log entry immediately to file
            # - level="DEBUG": Log all levels
            # - encoding="utf-8": Explicitly specify UTF-8 encoding
            # - mode="a": Append mode to preserve history
            handler_id = logger.add(
                self.config.LOG_FILE,
                level="DEBUG",
                enqueue=False,      # Disable async queue, synchronous write
                buffering=1,        # Line buffer, each line writes immediately
                serialize=False,    # Plain text format, do not serialize to JSON
                encoding="utf-8",   # Explicit UTF-8 encoding
                mode="a",           # Append mode
                filter=_exclude_other_engines # Filter logs from four Engines, keep other information
            )
            logger.debug(f"Added log handler (ID: {handler_id}): {self.config.LOG_FILE}")

        # [FIX] Verify log file is writable
        try:
            with open(self.config.LOG_FILE, 'a', encoding='utf-8') as f:
                f.write('')  # Try writing empty string to verify permissions
                f.flush()    # Flush immediately
        except Exception as e:
            logger.error(f"Cannot write to log file: {self.config.LOG_FILE}, error: {e}")
            raise
        
    def _initialize_file_baseline(self):
        """
        Initialize file count baseline.

        Pass Media/Query directories to `FileCountBaseline`,
        generate one-time reference values, then judge by increment whether engines produced new reports.
        """
        directories = {
            'media': 'media_engine_streamlit_reports',
            'query': 'query_engine_streamlit_reports'
        }
        self.file_baseline.initialize_baseline(directories)
    
    def _initialize_llm(self) -> LLMClient:
        """
        Initialize LLM client.

        Build unified `LLMClient` instance using API Key / model / Base URL from config,
        providing a reusable inference entry point for all nodes.
        """
        return LLMClient(
            api_key=self.config.REPORT_ENGINE_API_KEY,
            model_name=self.config.REPORT_ENGINE_MODEL_NAME,
            base_url=self.config.REPORT_ENGINE_BASE_URL,
        )

    def _initialize_rescue_llms(self) -> List[Tuple[str, LLMClient]]:
        """
        Initialize list of LLM clients needed for cross-engine chapter repair.

        Order follows“Report → Forum → Media”, missing configs are automatically skipped.
        """
        clients: List[Tuple[str, LLMClient]] = []
        if self.llm_client:
            clients.append(("report_engine", self.llm_client))
        fallback_specs = [
            (
                "forum_engine",
                self.config.FORUM_HOST_API_KEY,
                self.config.FORUM_HOST_MODEL_NAME,
                self.config.FORUM_HOST_BASE_URL,
            ),
            (
                "media_engine",
                self.config.MEDIA_ENGINE_API_KEY,
                self.config.MEDIA_ENGINE_MODEL_NAME,
                self.config.MEDIA_ENGINE_BASE_URL,
            ),
        ]
        for label, api_key, model_name, base_url in fallback_specs:
            if not api_key or not model_name:
                continue
            try:
                client = LLMClient(api_key=api_key, model_name=model_name, base_url=base_url)
            except Exception as exc:
                logger.warning(f"{label} LLM initialization failed, skipping this repair channel: {exc}")
                continue
            clients.append((label, client))
        return clients
    
    def _initialize_nodes(self):
        """
        Initialize processing nodes.

        Instantiate four nodes in sequence: template selection, document layout, word budget, chapter generation,
        where the chapter node additionally depends on IR validator and chapter storage.
        """
        self.template_selection_node = TemplateSelectionNode(
            self.llm_client,
            self.config.TEMPLATE_DIR
        )
        self.document_layout_node = DocumentLayoutNode(self.llm_client)
        self.word_budget_node = WordBudgetNode(self.llm_client)
        self.chapter_generation_node = ChapterGenerationNode(
            self.llm_client,
            self.validator,
            self.chapter_storage,
            fallback_llm_clients=self.json_rescue_clients,
            error_log_dir=self.config.JSON_ERROR_LOG_DIR,
        )
    
    def generate_report(self, query: str, reports: List[Any], forum_logs: str = "",
                        custom_template: str = "", save_report: bool = True,
                        stream_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> str:
        """
        Generate comprehensive report (Chapter JSON → IR → HTML).

        Main stages:
            1. Normalize dual-engine reports + forum logs, and emit streaming events;
            2. Template selection → template slicing → document layout → word budget planning;
            3. Invoke LLM chapter by chapter with word targets, auto-retry on parse errors;
            4. Stitch chapters into Document IR, then hand to HTML renderer for final product;
            5. Optionally persist HTML/IR/state to disk and return path info externally.

        Args:
            query: The final report topic or query statement to generate.
            reports: Raw outputs from Query/Media analysis engines, can be strings or more complex objects.
            forum_logs: Forum/collaboration records for LLM to understand multi-person discussion context.
            custom_template: User-specified Markdown template, if empty let template node auto-select.
            save_report: Whether to auto-write HTML, IR and state to disk after generation.
            stream_handler: Optional streaming event callback receiving stage labels and payloads for real-time UI display.

        Returns:
            dict: Contains `html_content` and HTML/IR/state file paths; if `save_report=False` returns HTML string only.

        Raises:
            Exception: Thrown when any sub-node or rendering stage fails, outer caller handles fallback.
        """
        start_time = datetime.now()
        report_id = f"report-{uuid4().hex[:8]}"
        self.state.task_id = report_id
        self.state.query = query
        self.state.metadata.query = query
        self.state.mark_processing()

        normalized_reports = self._normalize_reports(reports)

        def emit(event_type: str, payload: Dict[str, Any]):
            """Event dispatcher for Report Engine streaming channel, ensuring errors do not leak out."""
            if not stream_handler:
                return
            try:
                stream_handler(event_type, payload)
            except Exception as callback_error:  # pragma: no cover - logging only
                logger.warning(f"Streaming event callback failed: {callback_error}")

        logger.info(f"Starting report generation {report_id}: {query}")
        logger.info(f"Input data - report count: {len(reports)}, forum log length: {len(str(forum_logs))}")
        emit('stage', {'stage': 'agent_start', 'report_id': report_id, 'query': query})

        try:
            template_result = self._select_template(query, reports, forum_logs, custom_template)
            template_result = self._ensure_mapping(
                template_result,
                "template selection result",
                expected_keys=["template_name", "template_content"],
            )
            self.state.metadata.template_used = template_result.get('template_name', '')
            emit('stage', {
                'stage': 'template_selected',
                'template': template_result.get('template_name'),
                'reason': template_result.get('selection_reason')
            })
            emit('progress', {'progress': 10, 'message': 'Template selection complete'})
            sections = self._slice_template(template_result.get('template_content', ''))
            if not sections:
                raise ValueError("Template cannot be parsed into chapters, please check template content.")
            emit('stage', {'stage': 'template_sliced', 'section_count': len(sections)})

            template_text = template_result.get('template_content', '')
            template_overview = self._build_template_overview(template_text, sections)
            # Design global title, TOC and visual theme based on template skeleton + dual-engine content
            layout_design = self._run_stage_with_retry(
                "document design",
                lambda: self.document_layout_node.run(
                    sections,
                    template_text,
                    normalized_reports,
                    forum_logs,
                    query,
                    template_overview,
                ),
                # toc field replaced by tocPlan, select/validate per latest Schema here
                expected_keys=["title", "hero", "tocPlan", "tocTitle"],
            )
            emit('stage', {
                'stage': 'layout_designed',
                'title': layout_design.get('title'),
                'toc': layout_design.get('tocTitle')
            })
            emit('progress', {'progress': 15, 'message': 'Document title/TOC design complete'})
            # Use newly generated design draft for book-wide word budget planning, constraining each chapter's word count and focus
            word_plan = self._run_stage_with_retry(
                "chapter word budget planning",
                lambda: self.word_budget_node.run(
                    sections,
                    layout_design,
                    normalized_reports,
                    forum_logs,
                    query,
                    template_overview,
                ),
                expected_keys=["chapters", "totalWords", "globalGuidelines"],
                postprocess=self._normalize_word_plan,
            )
            emit('stage', {
                'stage': 'word_plan_ready',
                'chapter_targets': len(word_plan.get('chapters', []))
            })
            emit('progress', {'progress': 20, 'message': 'Chapter word budget generated'})
            # Record each chapter's target word count/emphasis points, pass to chapter LLM later
            chapter_targets = {
                entry.get("chapterId"): entry
                for entry in word_plan.get("chapters", [])
                if entry.get("chapterId")
            }

            generation_context = self._build_generation_context(
                query,
                normalized_reports,
                forum_logs,
                template_result,
                layout_design,
                chapter_targets,
                word_plan,
                template_overview,
            )
            # Global metadata needed for IR/rendering, carrying title/theme/TOC/word count info from design draft
            manifest_meta = {
                "query": query,
                "title": layout_design.get("title") or (f"{query} - Sentiment Insight Report" if query else template_result.get("template_name")),
                "subtitle": layout_design.get("subtitle"),
                "tagline": layout_design.get("tagline"),
                "templateName": template_result.get("template_name"),
                "selectionReason": template_result.get("selection_reason"),
                "themeTokens": generation_context.get("theme_tokens", {}),
                "toc": {
                    "depth": 3,
                    "autoNumbering": True,
                    "title": layout_design.get("tocTitle") or "Table of Contents",
                },
                "hero": layout_design.get("hero"),
                "layoutNotes": layout_design.get("layoutNotes"),
                "wordPlan": {
                    "totalWords": word_plan.get("totalWords"),
                    "globalGuidelines": word_plan.get("globalGuidelines"),
                },
                "templateOverview": template_overview,
            }
            if layout_design.get("themeTokens"):
                manifest_meta["themeTokens"] = layout_design["themeTokens"]
            if layout_design.get("tocPlan"):
                manifest_meta["toc"]["customEntries"] = layout_design["tocPlan"]
            # Initialize chapter output directory and write manifest for streaming storage
            run_dir = self.chapter_storage.start_session(report_id, manifest_meta)
            self._persist_planning_artifacts(run_dir, layout_design, word_plan, template_overview)
            emit('stage', {'stage': 'storage_ready', 'run_dir': str(run_dir)})

            chapters = []
            chapter_max_attempts = max(
                self._CONTENT_SPARSE_MIN_ATTEMPTS, self.config.CHAPTER_JSON_MAX_ATTEMPTS
            )
            total_chapters = len(sections)  # Total chapter count
            completed_chapters = 0  # Completed chapter count

            for section in sections:
                logger.info(f"Generating chapter: {section.title}")
                emit('chapter_status', {
                    'chapterId': section.chapter_id,
                    'title': section.title,
                    'status': 'running'
                })
                # Chapter streaming callback: pass LLM returned delta to SSE for real-time frontend rendering
                def chunk_callback(delta: str, meta: Dict[str, Any], section_ref: TemplateSection = section):
                    """
                    Chapter content streaming callback.

                    Args:
                        delta: The latest incremental text output from LLM.
                        meta: Chapter metadata returned by node, used for fallback.
                        section_ref: Defaults to current chapter, ensuring positioning when metadata is missing.
                    """
                    emit('chapter_chunk', {
                        'chapterId': meta.get('chapterId') or section_ref.chapter_id,
                        'title': meta.get('title') or section_ref.title,
                        'delta': delta
                    })

                chapter_payload: Dict[str, Any] | None = None
                attempt = 1
                best_sparse_candidate: Dict[str, Any] | None = None
                best_sparse_score = -1
                fallback_used = False
                while attempt <= chapter_max_attempts:
                    try:
                        chapter_payload = self.chapter_generation_node.run(
                            section,
                            generation_context,
                            run_dir,
                            stream_callback=chunk_callback
                        )
                        break
                    except (ChapterJsonParseError, ChapterContentError, ChapterValidationError) as structured_error:
                        if isinstance(structured_error, ChapterContentError):
                            error_kind = "content_sparse"
                            readable_label = "content density anomaly"
                        elif isinstance(structured_error, ChapterValidationError):
                            error_kind = "validation"
                            readable_label = "structure validation failed"
                        else:
                            error_kind = "json_parse"
                            readable_label = "JSON parse failed"
                        if isinstance(structured_error, ChapterContentError):
                            candidate = getattr(structured_error, "chapter_payload", None)
                            candidate_score = getattr(structured_error, "body_characters", 0) or 0
                            if isinstance(candidate, dict) and candidate_score >= 0:
                                if candidate_score > best_sparse_score:
                                    best_sparse_candidate = deepcopy(candidate)
                                    best_sparse_score = candidate_score
                        will_fallback = (
                            isinstance(structured_error, ChapterContentError)
                            and attempt >= chapter_max_attempts
                            and attempt >= self._CONTENT_SPARSE_MIN_ATTEMPTS
                            and best_sparse_candidate is not None
                        )
                        logger.warning(
                            "Chapter {title} {label} (attempt {attempt}/{total}): {error}",
                            title=section.title,
                            label=readable_label,
                            attempt=attempt,
                            total=chapter_max_attempts,
                            error=structured_error,
                        )
                        status_value = 'retrying' if attempt < chapter_max_attempts or will_fallback else 'error'
                        status_payload = {
                            'chapterId': section.chapter_id,
                            'title': section.title,
                            'status': status_value,
                            'attempt': attempt,
                            'error': str(structured_error),
                            'reason': error_kind,
                        }
                        if isinstance(structured_error, ChapterValidationError):
                            validation_errors = getattr(structured_error, "errors", None)
                            if validation_errors:
                                status_payload['errors'] = validation_errors
                        if will_fallback:
                            status_payload['warning'] = 'content_sparse_fallback_pending'
                        emit('chapter_status', status_payload)
                        if will_fallback:
                            logger.warning(
                                "Chapter {title} reached max attempts, keeping version with most characters (~{score} chars) as fallback",
                                title=section.title,
                                score=best_sparse_score,
                            )
                            chapter_payload = self._finalize_sparse_chapter(best_sparse_candidate)
                            fallback_used = True
                            break
                        if attempt >= chapter_max_attempts:
                            raise
                        attempt += 1
                        continue
                    except (AttributeError, TypeError, KeyError, IndexError, ValueError, json.JSONDecodeError) as structure_error:
                        # Catch runtime errors caused by JSON structure anomalies, wrap as retryable exceptions
                        # Includes:
                        # - AttributeError: e.g., list.get() call failed
                        # - TypeError: type mismatch
                        # - KeyError: missing dict key
                        # - IndexError: list index out of bounds
                        # - ValueError: value error (e.g., LLM returned empty content, missing required fields)
                        # - json.JSONDecodeError: JSON parse failure (cases not caught internally)
                        error_type = type(structure_error).__name__
                        logger.warning(
                            "Chapter {title} encountered {error_type} during generation (attempt {attempt}/{total}), will retry: {error}",
                            title=section.title,
                            error_type=error_type,
                            attempt=attempt,
                            total=chapter_max_attempts,
                            error=structure_error,
                        )
                        emit('chapter_status', {
                            'chapterId': section.chapter_id,
                            'title': section.title,
                            'status': 'retrying' if attempt < chapter_max_attempts else 'error',
                            'attempt': attempt,
                            'error': str(structure_error),
                            'reason': 'structure_error',
                            'error_type': error_type
                        })
                        if attempt >= chapter_max_attempts:
                            # Reached max retry attempts, wrap as ChapterJsonParseError and raise
                            raise ChapterJsonParseError(
                                f"Chapter {section.title} failed due to {error_type} after {chapter_max_attempts} attempts: {structure_error}"
                            ) from structure_error
                        attempt += 1
                        continue
                    except Exception as chapter_error:
                        if not self._should_retry_inappropriate_content_error(chapter_error):
                            raise
                        logger.warning(
                            "Chapter {title} triggered content safety restriction (attempt {attempt}/{total}), preparing to regenerate: {error}",
                            title=section.title,
                            attempt=attempt,
                            total=chapter_max_attempts,
                            error=chapter_error,
                        )
                        emit('chapter_status', {
                            'chapterId': section.chapter_id,
                            'title': section.title,
                            'status': 'retrying' if attempt < chapter_max_attempts else 'error',
                            'attempt': attempt,
                            'error': str(chapter_error),
                            'reason': 'content_filter'
                        })
                        if attempt >= chapter_max_attempts:
                            raise
                        attempt += 1
                        continue
                if chapter_payload is None:
                    raise ChapterJsonParseError(
                        f"Chapter {section.title} JSON could not be parsed after {chapter_max_attempts} attempts"
                    )
                chapters.append(chapter_payload)
                completed_chapters += 1  # Update completed chapter count
                # Calculate current progress: 20% + 80% * (completed / total), rounded
                chapter_progress = 20 + round(80 * completed_chapters / total_chapters)
                emit('progress', {
                    'progress': chapter_progress,
                    'message': f'Chapter {completed_chapters}/{total_chapters} completed'
                })
                completion_status = {
                    'chapterId': section.chapter_id,
                    'title': section.title,
                    'status': 'completed',
                    'attempt': attempt,
                }
                if fallback_used:  # using fallback
                    completion_status['warning'] = 'content_sparse_fallback'
                    completion_status['warningMessage'] = self._CONTENT_SPARSE_WARNING_TEXT
                emit('chapter_status', completion_status)

            document_ir = self.document_composer.build_document(
                report_id,
                manifest_meta,
                chapters
            )
            emit('stage', {'stage': 'chapters_compiled', 'chapter_count': len(chapters)})
            html_report = self.renderer.render(document_ir)
            emit('stage', {'stage': 'html_rendered', 'html_length': len(html_report)})

            self.state.html_content = html_report
            self.state.mark_completed()

            saved_files = {}
            if save_report:
                saved_files = self._save_report(html_report, document_ir, report_id)
                emit('stage', {'stage': 'report_saved', 'files': saved_files})

            generation_time = (datetime.now() - start_time).total_seconds()
            self.state.metadata.generation_time = generation_time
            logger.info(f"Report generation complete, elapsed: {generation_time:.2f} seconds")
            emit('metrics', {'generation_seconds': generation_time})
            return {
                'html_content': html_report,
                'report_id': report_id,
                **saved_files
            }

        except Exception as e:
            self.state.mark_failed(str(e))
            logger.exception(f"Error occurred during report generation: {str(e)}")
            emit('error', {'stage': 'agent_failed', 'message': str(e)})
            raise
    
    def _select_template(self, query: str, reports: List[Any], forum_logs: str, custom_template: str):
        """
        Select report template.

        Prefer user-specified template; otherwise pass query, dual-engine reports and forum logs
        as context to TemplateSelectionNode, letting LLM return the most suitable
        template name, content and reason, automatically recorded in state.

        Args:
            query: Report topic for prompt focusing on industry/event.
            reports: Multi-source report originals to help LLM judge structure complexity.
            forum_logs: Forum or collaboration discussion text for background supplementation.
            custom_template: Custom Markdown template from CLI/frontend, used directly if non-empty.

        Returns:
            dict: Structured result containing `template_name`, `template_content` and `selection_reason` for downstream nodes.
        """
        logger.info("Selecting report template...")
        
        # If user provided custom template, use it directly
        if custom_template:
            logger.info("Using user custom template")
            return {
                'template_name': 'custom',
                'template_content': custom_template,
                'selection_reason': 'User-specified custom template'
            }
        
        template_input = {
            'query': query,
            'reports': reports,
            'forum_logs': forum_logs
        }
        
        try:
            template_result = self.template_selection_node.run(template_input)
            
            # Update state
            self.state.metadata.template_used = template_result['template_name']
            
            logger.info(f"Selected template: {template_result['template_name']}")
            logger.info(f"Selection reason: {template_result['selection_reason']}")
            
            return template_result
        except Exception as e:
            logger.error(f"Template selection failed, using default template: {str(e)}")
            # Use fallback template directly
            fallback_template = {
                'template_name': 'Social Public Hot Event Analysis Report Template',
                'template_content': self._get_fallback_template_content(),
                'selection_reason': 'Template selection failed, using default social hot event analysis template'
            }
            self.state.metadata.template_used = fallback_template['template_name']
            return fallback_template
    
    def _slice_template(self, template_markdown: str) -> List[TemplateSection]:
        """
        Slice template into chapter list, providing fallback if empty.

        Delegate to `parse_template_sections` to parse Markdown headings/numbers into
        `TemplateSection` list, ensuring stable chapter IDs for subsequent chapter generation.
        When template format is abnormal, falls back to built-in simple skeleton to avoid crash.

        Args:
            template_markdown: Complete template Markdown text.

        Returns:
            list[TemplateSection]: Parsed chapter sequence; returns single-chapter fallback structure if parsing fails.
        """
        sections = parse_template_sections(template_markdown)
        if sections:
            return sections
        logger.warning("Template did not parse into chapters, using default chapter skeleton")
        fallback = TemplateSection(
            title="1.0 Comprehensive Analysis",
            slug="section-1-0",
            order=10,
            depth=1,
            raw_title="1.0 Comprehensive Analysis",
            number="1.0",
            chapter_id="S1",
            outline=["1.1 Abstract", "1.2 Data Highlights", "1.3 Risk Warnings"],
        )
        return [fallback]

    def _build_generation_context(
        self,
        query: str,
        reports: Dict[str, str],
        forum_logs: str,
        template_result: Dict[str, Any],
        layout_design: Dict[str, Any],
        chapter_directives: Dict[str, Any],
        word_plan: Dict[str, Any],
        template_overview: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build shared context needed for chapter generation.

        Integrate template name, layout design, theme colors, word budget, forum logs etc.
        into `generation_context` in one go, reused directly when calling LLM for each chapter
        later, ensuring all chapters share consistent tone and visual constraints.

        Args:
            query: User query keywords.
            reports: Normalized query/media report mapping.
            forum_logs: Engine discussion records.
            template_result: Template metadata returned by template node.
            layout_design: Title/TOC/theme design produced by document layout node.
            chapter_directives: Chapter directive mapping returned by word budget node.
            word_plan: Raw word budget result containing global word count constraints.
            template_overview: Chapter skeleton summary extracted from template slicing.

        Returns:
            dict: Complete context needed for LLM chapter generation, containing theme colors, layout, constraints etc.
        """
        # Prefer theme colors customized in design draft, otherwise fall back to default theme
        theme_tokens = (
            layout_design.get("themeTokens")
            if layout_design else None
        ) or self._default_theme_tokens()

        return {
            "query": query,
            "template_name": template_result.get("template_name"),
            "reports": reports,
            "forum_logs": self._stringify(forum_logs),
            "theme_tokens": theme_tokens,
            "style_directives": {
                "tone": "analytical",
                "audience": "executive",
                "language": "en-US",
                "language_rule": ENGLISH_REPORT_LANGUAGE_RULE,
            },
            "data_bundles": [],
            "max_tokens": min(self.config.MAX_CONTENT_LENGTH, 6000),
            "layout": layout_design or {},
            "template_overview": template_overview or {},
            "chapter_directives": chapter_directives or {},
            "word_plan": word_plan or {},
        }

    def _normalize_reports(self, reports: List[Any]) -> Dict[str, str]:
        """
        Normalize reports from different sources into strings.

        Conventional order is Query/Media, engine-provided objects may be
        dicts or custom types, so uniformly go through `_stringify` for fault tolerance.

        Args:
            reports: Report list of any type, allowing missing or out-of-order items.

        Returns:
            dict: Mapping containing `query_engine` and `media_engine` string fields.
        """
        keys = ["query_engine", "media_engine"]
        normalized: Dict[str, str] = {}
        for idx, key in enumerate(keys):
            value = reports[idx] if idx < len(reports) else ""
            normalized[key] = self._stringify(value)
        return normalized

    def _should_retry_inappropriate_content_error(self, error: Exception) -> bool:
        """
        Determine if LLM exception was caused by content safety/inappropriate content.

        When specific keywords are detected in vendor-returned errors, allow chapter generation
        to retry to bypass sporadic content moderation triggers.

        Args:
            error: Exception object thrown by LLM client.

        Returns:
            bool: True if content moderation keywords matched, False otherwise.
        """
        message = str(error) if error else ""
        if not message:
            return False
        normalized = message.lower()
        keywords = [
            "inappropriate content",
            "content violation",
            "content moderation",
            "model-studio/error-code",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _run_stage_with_retry(
        self,
        stage_name: str,
        fn: Callable[[], Any],
        expected_keys: Optional[List[str]] = None,
        postprocess: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run single LLM stage with limited retries on structure anomalies.

        This method only does local fix/retry for structure errors, avoiding full Agent restart.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self._STRUCTURAL_RETRY_ATTEMPTS + 1):
            try:
                raw_result = fn()
                result = self._ensure_mapping(raw_result, stage_name, expected_keys)
                if postprocess:
                    result = postprocess(result, stage_name)
                return result
            except StageOutputFormatError as exc:
                last_error = exc
                logger.warning(
                    "{stage} output structure anomaly (attempt {attempt}/{total}), will try fix or retry: {error}",
                    stage=stage_name,
                    attempt=attempt,
                    total=self._STRUCTURAL_RETRY_ATTEMPTS,
                    error=exc,
                )
                if attempt >= self._STRUCTURAL_RETRY_ATTEMPTS:
                    break
        raise last_error  # type: ignore[misc]

    def _ensure_mapping(
        self,
        value: Any,
        context: str,
        expected_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ensure stage output is dict; if list returned, try to extract best matching element.
        """
        if isinstance(value, dict):
            return value

        if isinstance(value, list):
            candidates = [item for item in value if isinstance(item, dict)]
            if candidates:
                best = candidates[0]
                if expected_keys:
                    candidates.sort(
                        key=lambda item: sum(1 for key in expected_keys if key in item),
                        reverse=True,
                    )
                    best = candidates[0]
                logger.warning(
                    "{context} returned list, automatically extracted element with most expected keys to continue",
                    context=context,
                )
                return best
            raise StageOutputFormatError(f"{context} returned list but lacks available object elements")

        if value is None:
            raise StageOutputFormatError(f"{context} returned empty result")

        raise StageOutputFormatError(
            f"{context} returned type {type(value).__name__}, expected dict"
        )

    def _normalize_word_plan(self, word_plan: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        """
        Clean word budget result, ensuring chapters/globalGuidelines/totalWords type safety.
        """
        raw_chapters = word_plan.get("chapters", [])
        if isinstance(raw_chapters, dict):
            chapters_iterable = raw_chapters.values()
        elif isinstance(raw_chapters, list):
            chapters_iterable = raw_chapters
        else:
            chapters_iterable = []

        normalized: List[Dict[str, Any]] = []
        for idx, entry in enumerate(chapters_iterable):
            if isinstance(entry, dict):
                normalized.append(entry)
                continue
            if isinstance(entry, list):
                dict_candidate = next((item for item in entry if isinstance(item, dict)), None)
                if dict_candidate:
                    logger.warning(
                        "{stage} chapter entry #{idx} is list, extracted first object for downstream",
                        stage=stage_name,
                        idx=idx + 1,
                    )
                    normalized.append(dict_candidate)
                    continue
            logger.warning(
                "{stage} skipping unparsable chapter entry #{idx} (type: {type_name})",
                stage=stage_name,
                idx=idx + 1,
                type_name=type(entry).__name__,
            )

        if not normalized:
            raise StageOutputFormatError(f"{stage_name} lacks valid chapter planning, cannot continue")

        word_plan["chapters"] = normalized

        guidelines = word_plan.get("globalGuidelines")
        if not isinstance(guidelines, list):
            if guidelines is None or guidelines == "":
                word_plan["globalGuidelines"] = []
            else:
                logger.warning(
                    "{stage} globalGuidelines type anomaly, converted to list wrapper",
                    stage=stage_name,
                )
                word_plan["globalGuidelines"] = [guidelines]

        if not isinstance(word_plan.get("totalWords"), (int, float)):
            logger.warning(
                "{stage} totalWords type anomaly, using default value 10000",
                stage=stage_name,
            )
            word_plan["totalWords"] = 10000

        return word_plan

    def _finalize_sparse_chapter(self, chapter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Construct content-sparse fallback chapter: copy original payload and insert warning paragraph.
        """
        safe_chapter = deepcopy(chapter or {})
        if not isinstance(safe_chapter, dict):
            safe_chapter = {}
        self._ensure_sparse_warning_block(safe_chapter)
        return safe_chapter

    def _ensure_sparse_warning_block(self, chapter: Dict[str, Any]) -> None:
        """
        Insert warning paragraph after chapter heading to alert readers that chapter is short.
        """
        warning_block = {
            "type": "paragraph",
            "inlines": [
                {
                    "text": self._CONTENT_SPARSE_WARNING_TEXT,
                    "marks": [{"type": "italic"}],
                }
            ],
            "meta": {"role": "content-sparse-warning"},
        }
        blocks = chapter.get("blocks")
        if isinstance(blocks, list) and blocks:
            inserted = False
            for idx, block in enumerate(blocks):
                if isinstance(block, dict) and block.get("type") == "heading":
                    blocks.insert(idx + 1, warning_block)
                    inserted = True
                    break
            if not inserted:
                blocks.insert(0, warning_block)
        else:
            chapter["blocks"] = [warning_block]
        meta = chapter.get("meta")
        if isinstance(meta, dict):
            meta["contentSparseWarning"] = True
        else:
            chapter["meta"] = {"contentSparseWarning": True}

    def _stringify(self, value: Any) -> str:
        """
        Safely convert object to string.

        - dict/list uniformly serialized to formatted JSON for prompt consumption;
        - Other types go through `str()`, None returns empty string to avoid None propagation.

        Args:
            value: Any Python object.

        Returns:
            str: String representation suitable for prompts/logs.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:
                return str(value)
        return str(value)

    def _default_theme_tokens(self) -> Dict[str, Any]:
        """
        Construct default theme variables for renderer/LLM shared use.

        Use this color palette when layout node does not return dedicated colors, keeping report style consistent.

        Returns:
            dict: Theme dictionary containing colors, fonts, spacing, boolean switches and other rendering parameters.
        """
        return {
            "colors": {
                "bg": "#f8f9fa",
                "text": "#212529",
                "primary": "#007bff",
                "secondary": "#6c757d",
                "card": "#ffffff",
                "border": "#dee2e6",
                "accent1": "#17a2b8",
                "accent2": "#28a745",
                "accent3": "#ffc107",
                "accent4": "#dc3545",
            },
            "fonts": {
                "body": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif",
                "heading": "'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif",  # Chinese fonts
            },
            "spacing": {"container": "1200px", "gutter": "24px"},
            "vars": {
                "header_sticky": True,
                "toc_depth": 3,
                "enable_dark_mode": True,
            },
        }

    def _build_template_overview(
        self,
        template_markdown: str,
        sections: List[TemplateSection],
    ) -> Dict[str, Any]:
        """
        Extract template title and chapter skeleton for design/word budget unified reference.

        Also record auxiliary fields like chapter ID/slug/order to ensure multi-node alignment.

        Args:
            template_markdown: Original template text for parsing global title.
            sections: `TemplateSection` list as chapter skeleton.

        Returns:
            dict: Overview structure containing template title and chapter metadata.
        """
        fallback_title = sections[0].title if sections else ""
        overview = {
            "title": self._extract_template_title(template_markdown, fallback_title),
            "chapters": [],
        }
        for section in sections:
            overview["chapters"].append(
                {
                    "chapterId": section.chapter_id,
                    "title": section.title,
                    "rawTitle": section.raw_title,
                    "number": section.number,
                    "slug": section.slug,
                    "order": section.order,
                    "depth": section.depth,
                    "outline": section.outline,
                }
            )
        return overview

    @staticmethod
    def _extract_template_title(template_markdown: str, fallback: str = "") -> str:
        """
        Try to extract first heading from Markdown.

        Prefer returning first `#` syntax heading; if template first line is body text,
        fall back to first non-empty line or caller-provided fallback.

        Args:
            template_markdown: Original template text.
            fallback: Fallback title when document lacks explicit heading.

        Returns:
            str: Parsed heading text.
        """
        for line in template_markdown.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped:
                fallback = fallback or stripped
        return fallback or "Intelligent Sentiment Analysis Report"
    
    def _get_fallback_template_content(self) -> str:
        """
        Get fallback template content.

        Use this Markdown template when template directory is unavailable or LLM selection fails,
        ensuring subsequent workflow can still provide structured chapters.
        """
        return """# Social Public Hot Event Analysis Report

## Executive Summary
This report comprehensively analyzes current social hot events, integrating perspectives and data from multiple information sources.

## Event Overview
### Basic Information
- Event Nature: {event_nature}
- Event Time: {event_time}
- Event Scope: {event_scope}

## Sentiment Situation Analysis
### Overall Trend
{sentiment_analysis}

### Main Opinion Distribution
{opinion_distribution}

## Media Reporting Analysis
### Mainstream Media Attitude
{media_analysis}

### Reporting Focus
{report_focus}

## Social Impact Assessment
### Direct Impact
{direct_impact}

### Potential Impact
{potential_impact}

## Response Recommendations
### Immediate Measures
{immediate_actions}

### Long-term Strategy
{long_term_strategy}

## Conclusion and Outlook
{conclusion}

---
*Report Type: Social Public Hot Event Analysis*
*Generation Time: {generation_time}*
"""
    
    def _save_report(self, html_content: str, document_ir: Dict[str, Any], report_id: str) -> Dict[str, Any]:
        """
        Save HTML and IR to files and return path information.

        Generate human-readable filenames based on query and timestamp, also write
        runtime `ReportState` to JSON for downstream troubleshooting or resumable execution.

        Args:
            html_content: Rendered HTML content.
            document_ir: Document IR structured data.
            report_id: Current task ID for creating unique filenames.

        Returns:
            dict: Records absolute and relative paths for HTML/IR/State files.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(
            c for c in self.state.metadata.query if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        query_safe = query_safe.replace(" ", "_")[:30] or "report"

        html_filename = f"final_report_{query_safe}_{timestamp}.html"
        html_path = Path(self.config.OUTPUT_DIR) / html_filename
        html_path.write_text(html_content, encoding="utf-8")
        html_abs = str(html_path.resolve())
        html_rel = os.path.relpath(html_abs, os.getcwd())

        ir_path = self._save_document_ir(document_ir, query_safe, timestamp)
        ir_abs = str(ir_path.resolve())
        ir_rel = os.path.relpath(ir_abs, os.getcwd())

        state_filename = f"report_state_{query_safe}_{timestamp}.json"
        state_path = Path(self.config.OUTPUT_DIR) / state_filename
        self.state.save_to_file(str(state_path))
        state_abs = str(state_path.resolve())
        state_rel = os.path.relpath(state_abs, os.getcwd())

        logger.info(f"HTML report saved: {html_path}")
        logger.info(f"Document IR saved: {ir_path}")
        logger.info(f"State saved to: {state_path}")
        
        return {
            'report_filename': html_filename,
            'report_filepath': html_abs,
            'report_relative_path': html_rel,
            'ir_filename': ir_path.name,
            'ir_filepath': ir_abs,
            'ir_relative_path': ir_rel,
            'state_filename': state_filename,
            'state_filepath': state_abs,
            'state_relative_path': state_rel,
        }

    def _save_document_ir(self, document_ir: Dict[str, Any], query_safe: str, timestamp: str) -> Path:
        """
        Write entire IR to separate directory.

        Decouple `Document IR` from HTML storage for debugging rendering differences and
        re-rendering or exporting other formats without re-running LLM.

        Args:
            document_ir: Entire report IR structure.
            query_safe: Cleaned query phrase for file naming.
            timestamp: Runtime timestamp ensuring filename uniqueness.

        Returns:
            Path: Path to saved IR file.
        """
        filename = f"report_ir_{query_safe}_{timestamp}.json"
        ir_path = Path(self.config.DOCUMENT_IR_OUTPUT_DIR) / filename
        ir_path.write_text(
            json.dumps(document_ir, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ir_path
    
    def _persist_planning_artifacts(
        self,
        run_dir: Path,
        layout_design: Dict[str, Any],
        word_plan: Dict[str, Any],
        template_overview: Dict[str, Any],
    ):
        """
        Save document design draft, word budget and template overview as JSON.

        These intermediate files (document_layout/word_plan/template_overview)
        facilitate quick positioning during debugging or post-analysis: how title/TOC/theme were determined,
        word allocation requirements, for subsequent manual correction.

        Args:
            run_dir: Chapter output root directory.
            layout_design: Raw output from document layout node.
            word_plan: Word budget node output.
            template_overview: Template overview JSON.
        """
        artifacts = {
            "document_layout": layout_design,
            "word_plan": word_plan,
            "template_overview": template_overview,
        }
        for name, payload in artifacts.items():
            if not payload:
                continue
            path = run_dir / f"{name}.json"
            try:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.warning(f"Failed to write {name}: {exc}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary, directly returning serializable state dict for API layer queries."""
        return self.state.to_dict()
    
    def load_state(self, filepath: str):
        """Load state from file and override current state for breakpoint resumption."""
        self.state = ReportState.load_from_file(filepath)
        logger.info(f"State loaded from {filepath}")
    
    def save_state(self, filepath: str):
        """Save state to file, typically for post-task analysis and backup."""
        self.state.save_to_file(filepath)
        logger.info(f"State saved to {filepath}")
    
    def check_input_files(self, media_dir: str, query_dir: str, forum_log_path: str) -> Dict[str, Any]:
        """
        Check if input files are ready (based on file count increase).

        Args:
            media_dir: MediaEngine report directory
            query_dir: QueryEngine report directory
            forum_log_path: Forum log file path

        Returns:
            Check result dict containing file counts, missing list, latest file paths etc.
        """
        # Check file count changes in each report directory
        directories = {
            'media': media_dir,
            'query': query_dir
        }
        
        # Use file baseline manager to check for new files
        check_result = self.file_baseline.check_new_files(directories)
        
        # Check forum logs
        forum_ready = os.path.exists(forum_log_path)
        
        # Build return result
        result = {
            'ready': check_result['ready'] and forum_ready,
            'baseline_counts': check_result['baseline_counts'],
            'current_counts': check_result['current_counts'],
            'new_files_found': check_result['new_files_found'],
            'missing_files': [],
            'files_found': [],
            'latest_files': {}
        }
        
        # Build detailed information
        for engine, new_count in check_result['new_files_found'].items():
            current_count = check_result['current_counts'][engine]
            baseline_count = check_result['baseline_counts'].get(engine, 0)
            
            if new_count > 0:
                result['files_found'].append(f"{engine}: {current_count} files ({new_count} new)")
            else:
                result['missing_files'].append(f"{engine}: {current_count} files (baseline {baseline_count}, no new)")
        
        # Check forum logs
        if forum_ready:
            result['files_found'].append(f"forum: {os.path.basename(forum_log_path)}")
        else:
            result['missing_files'].append("forum: log file does not exist")
        
        # Get latest file paths (for actual report generation)
        if result['ready']:
            result['latest_files'] = self.file_baseline.get_latest_files(directories)
            if forum_ready:
                result['latest_files']['forum'] = forum_log_path
        
        return result
    
    def load_input_files(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Load input file contents

        Args:
            file_paths: Dictionary of file paths

        Returns:
            Loaded content dict containing `reports` list and `forum_logs` string
        """
        content = {
            'reports': [],
            'forum_logs': ''
        }
        
        # Load report files
        engines = ['query', 'media']
        for engine in engines:
            if engine in file_paths:
                try:
                    with open(file_paths[engine], 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    content['reports'].append(report_content)
                    logger.info(f"Loaded {engine} report: {len(report_content)} characters")
                except Exception as e:
                    logger.exception(f"Failed to load {engine} report: {str(e)}")
                    content['reports'].append("")
        
        # Load forum logs
        if 'forum' in file_paths:
            try:
                with open(file_paths['forum'], 'r', encoding='utf-8') as f:
                    content['forum_logs'] = f.read()
                logger.info(f"Loaded forum logs: {len(content['forum_logs'])} characters")
            except Exception as e:
                logger.exception(f"Failed to load forum logs: {str(e)}")
        
        return content


def create_agent(config_file: Optional[str] = None) -> ReportAgent:
    """
    Convenience function to create Report Agent instance.

    Args:
        config_file: Configuration file path

    Returns:
        ReportAgent instance

    Currently driven by environment variables for `Settings`, keeping `config_file` parameter for future extension.
    """
    
    config = Settings()  # Initialize with empty config, populated from environment variables
    return ReportAgent(config)
