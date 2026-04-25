"""
Report Engine Flask Interface.

This module provides a unified HTTP/SSE entry point for frontend/CLI, responsible for:
1. Initializing ReportAgent and coordinating background threads;
2. Managing task queuing, progress querying, streaming push, and log downloads;
3. Providing auxiliary capabilities such as template listing and input file checking.
"""

import os
import json
import threading
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from flask import Blueprint, request, jsonify, Response, send_file, stream_with_context
from typing import Dict, Any, List, Optional
from loguru import logger
from .agent import ReportAgent, create_agent
from .nodes import ChapterJsonParseError
from .utils.config import settings


# Create Flask Blueprint
report_bp = Blueprint('report_engine', __name__)

# Global variables
report_agent = None
current_task = None
task_lock = threading.Lock()

# ====== Streaming Push and Task History Management ======
# Use bounded deque to cache recent events, enabling quick replay after SSE disconnects
MAX_TASK_HISTORY = 5
STREAM_HEARTBEAT_INTERVAL = 15  # Heartbeat interval in seconds
STREAM_IDLE_TIMEOUT = 120  # Max keep-alive time after terminal state to prevent orphaned SSE blocking
STREAM_TERMINAL_STATUSES = {"completed", "error", "cancelled"}
stream_lock = threading.Lock()
stream_subscribers = defaultdict(list)
tasks_registry: Dict[str, 'ReportTask'] = {}
LOG_STREAM_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_stream_handler_id: Optional[int] = None

EXCLUDED_ENGINE_PATH_KEYWORDS = ("ForumEngine", "MediaEngine", "QueryEngine")

def _is_excluded_engine_log(record: Dict[str, Any]) -> bool:
    """
    Determine if the log originates from other engines (Media/Query/Forum), used for filtering mixed logs.

    Returns:
        bool: True indicates the log should be filtered (i.e., not written/forwarded).
    """
    try:
        file_path = record["file"].path
        if any(keyword in file_path for keyword in EXCLUDED_ENGINE_PATH_KEYWORDS):
            return True
    except Exception:
        pass

    # Fallback: try filtering by module name to prevent accidental mixing when file info is missing
    try:
        module_name = record.get("module", "")
        if isinstance(module_name, str):
            lowered = module_name.lower()
            return any(keyword.lower() in lowered for keyword in EXCLUDED_ENGINE_PATH_KEYWORDS)
    except Exception:
        pass

    return False


def _stream_log_to_task(message):
    """
    Synchronize loguru logs to the current task's SSE events, ensuring real-time visibility on the frontend.

    Only pushes when there is a running task to avoid irrelevant log flooding.
    """
    try:
        record = message.record
        level_name = record["level"].name
        if level_name not in LOG_STREAM_LEVELS:
            return
        if _is_excluded_engine_log(record):
            return

        with task_lock:
            task = current_task

        if not task or task.status not in ("running", "pending"):
            return

        timestamp = record["time"].strftime("%H:%M:%S.%f")[:-3]
        formatted_line = f"[{timestamp}] [{level_name}] {record['message']}"
        task.publish_event(
            "log",
            {
                "line": formatted_line,
                "level": level_name.lower(),
                "timestamp": timestamp,
                "message": record["message"],
                "module": record.get("module", ""),
                "function": record.get("function", ""),
            },
        )
    except Exception:
        # Avoid log recursion within the log hook
        pass


def _setup_log_stream_forwarder():
    """Mount a one-time loguru hook for the current process to enable real-time SSE forwarding."""
    global log_stream_handler_id
    if log_stream_handler_id is not None:
        return
    log_stream_handler_id = logger.add(
        _stream_log_to_task,
        level="DEBUG",
        enqueue=False,
        catch=True,
    )


def _register_stream(task_id: str) -> Queue:
    """
    Register an event queue for the specified task for SSE listener consumption.

    The returned Queue will be stored in `stream_subscribers`, and the SSE generator will continuously read from it.

    Args:
        task_id: The task ID to listen to.

    Returns:
        Queue: A thread-safe event queue.
    """
    queue = Queue()
    with stream_lock:
        stream_subscribers[task_id].append(queue)
    return queue


def _unregister_stream(task_id: str, queue: Queue):
    """
    Safely remove the event queue to avoid memory leaks.

    Should be called in a finally block to ensure resources are released even in exceptional cases.

    Args:
        task_id: The task ID.
        queue: The previously registered event queue.
    """
    with stream_lock:
        listeners = stream_subscribers.get(task_id, [])
        if queue in listeners:
            listeners.remove(queue)
        if not listeners and task_id in stream_subscribers:
            stream_subscribers.pop(task_id, None)


def _broadcast_event(task_id: str, event: Dict[str, Any]):
    """
    Push the event to all listeners, with proper exception handling on failure.

    Uses a shallow copy of the listener list to prevent iteration exceptions caused by concurrent removals.

    Args:
        task_id: The task ID to push to.
        event: The structured event payload.
    """
    with stream_lock:
        listeners = list(stream_subscribers.get(task_id, []))
    for queue in listeners:
        try:
            queue.put(event, timeout=0.1)
        except Exception:
            logger.exception("Failed to push streaming event, skipping current listener queue")


def _prune_task_history_locked():
    """
    Called while holding task_lock to clean up excessive historical tasks.

    Only retains the most recent `MAX_TASK_HISTORY` tasks to avoid excessive memory consumption during long runs.

    Note:
        This function assumes the caller has already acquired `task_lock`; otherwise, race condition risks exist.
    """
    if len(tasks_registry) <= MAX_TASK_HISTORY:
        return
    # Sort by creation time and remove oldest tasks
    sorted_tasks = sorted(tasks_registry.values(), key=lambda t: t.created_at)
    for task in sorted_tasks[:-MAX_TASK_HISTORY]:
        tasks_registry.pop(task.task_id, None)


def _get_task(task_id: str) -> Optional['ReportTask']:
    """
    Unified task lookup method, prioritizing the current task.

    Avoids redundant lock logic, making it convenient for multiple APIs to share.

    Args:
        task_id: The task ID.

    Returns:
        ReportTask | None: Returns the task instance when found, otherwise None.
    """
    with task_lock:
        if current_task and current_task.task_id == task_id:
            return current_task
        return tasks_registry.get(task_id)


def _format_sse(event: Dict[str, Any]) -> str:
    """
    Format message according to SSE protocol.

    Outputs a three-segment text in the format `id:/event:/data:` for direct browser consumption.

    Args:
        event: The event payload, must at least contain id/type.

    Returns:
        str: String as required by the SSE protocol.
    """
    payload = json.dumps(event, ensure_ascii=False)
    event_id = event.get('id', 0)
    event_type = event.get('type', 'message')
    return f"id: {event_id}\nevent: {event_type}\ndata: {payload}\n\n"


def _safe_filename_segment(value: str, fallback: str = "report") -> str:
    """
    Generate a safe segment for use in filenames, preserving alphanumeric characters and common separators.

    Args:
        value: The original string.
        fallback: Fallback text to use when value is empty or sanitized to empty.
    """
    sanitized = "".join(c for c in str(value) if c.isalnum() or c in (" ", "-", "_")).strip()
    sanitized = sanitized.replace(" ", "_")
    return sanitized or fallback


def initialize_report_engine():
    """
    Initialize the Report Engine.

    Creates a singleton ReportAgent for immediate task acceptance after API startup.

    Returns:
        bool: True if initialization succeeded, False on exception.
    """
    global report_agent
    try:
        report_agent = create_agent()
        logger.info("Report Engine initialized successfully")
        _setup_log_stream_forwarder()

        # Check PDF generation dependency (Pango)
        try:
            from .utils.dependency_check import log_dependency_status
            log_dependency_status()
        except Exception as dep_err:
            logger.warning(f"Dependency check failed: {dep_err}")

        return True
    except Exception as e:
        logger.exception(f"Report Engine initialization failed: {str(e)}")
        return False


class ReportTask:
    """
    Report generation task.

    This object tracks running status, progress, event history, and final file paths,
    serving both background thread updates and HTTP interface reads.
    """

    def __init__(self, query: str, task_id: str, custom_template: str = ""):
        """
        Initialize the task object, recording the query, custom template, and runtime metadata.

        Args:
            query: The report topic to be generated
            task_id: Unique task ID, typically constructed from timestamp
            custom_template: Optional custom Markdown template
        """
        self.task_id = task_id
        self.query = query
        self.custom_template = custom_template
        self.status = "pending"  # Four statuses (pending/running/completed/error)
        self.progress = 0
        self.result = None
        self.error_message = ""
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.html_content = ""
        self.report_file_path = ""
        self.report_file_relative_path = ""
        self.report_file_name = ""
        self.state_file_path = ""
        self.state_file_relative_path = ""
        self.ir_file_path = ""
        self.ir_file_relative_path = ""
        self.markdown_file_path = ""
        self.markdown_file_relative_path = ""
        self.markdown_file_name = ""
        # ====== Streaming Event Cache and Concurrency Protection ======
        # Use deque to store recent events, combined with locks to ensure thread-safe access
        self.event_history: deque = deque(maxlen=1000)
        self._event_lock = threading.Lock()
        self.last_event_id = 0

    def update_status(self, status: str, progress: int = None, error_message: str = ""):
        """
        Update task status and broadcast events.

        Automatically refreshes `updated_at`, error messages, and triggers `status` type SSE.

        Args:
            status: Task phase (pending/running/completed/error/cancelled).
            progress: Optional progress percentage.
            error_message: Human-readable description when an error occurs.
        """
        self.status = status
        if progress is not None:
            self.progress = progress
        if error_message:
            self.error_message = error_message
        self.updated_at = datetime.now()
        # Push status change event for real-time frontend refresh
        self.publish_event(
            'status',
            {
                'status': self.status,
                'progress': self.progress,
                'error_message': self.error_message,
                'hint': error_message or '',
                'task': self.to_dict(),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for direct return to JSON API."""
        return {
            'task_id': self.task_id,
            'query': self.query,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'has_result': bool(self.html_content),
            'report_file_ready': bool(self.report_file_path),
            'report_file_name': self.report_file_name,
            'report_file_path': self.report_file_relative_path or self.report_file_path,
            'state_file_ready': bool(self.state_file_path),
            'state_file_path': self.state_file_relative_path or self.state_file_path,
            'ir_file_ready': bool(self.ir_file_path),
            'ir_file_path': self.ir_file_relative_path or self.ir_file_path,
            'markdown_file_ready': bool(self.markdown_file_path),
            'markdown_file_name': self.markdown_file_name,
            'markdown_file_path': self.markdown_file_relative_path or self.markdown_file_path
        }

    def publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Cache and broadcast arbitrary events.

        Args:
            event_type: The event name in SSE.
            payload: Actual business data.
        """
        timestamp = datetime.utcnow().isoformat() + 'Z'
        event: Dict[str, Any] = {
            'id': 0,
            'type': event_type,
            'task_id': self.task_id,
            'timestamp': timestamp,
            'payload': payload,
        }
        with self._event_lock:
            self.last_event_id += 1
            event['id'] = self.last_event_id
            self.event_history.append(event)
        _broadcast_event(self.task_id, event)

    def history_since(self, last_event_id: Optional[int]) -> List[Dict[str, Any]]:
        """
        Replay historical events based on Last-Event-ID to ensure no events are missed after reconnect.

        Args:
            last_event_id: The last event ID recorded by the SSE client.

        Returns:
            list[dict]: List of events after last_event_id.
        """
        with self._event_lock:
            if last_event_id is None:
                return list(self.event_history)
            return [evt for evt in self.event_history if evt['id'] > last_event_id]


def check_engines_ready() -> Dict[str, Any]:
    """
    Check if both Media/Query sub-engines have new files.

    Calls ReportAgent's baseline detection logic along with forum log existence check,
    serving as a prerequisite validation for /status and /generate endpoints.
    """
    directories = {
        'media': 'media_engine_streamlit_reports',
        'query': 'query_engine_streamlit_reports'
    }

    forum_log_path = 'logs/forum.log'

    if not report_agent:
        return {
            'ready': False,
            'error': 'Report Engine not initialized'
        }

    return report_agent.check_input_files(
        directories['media'],
        directories['query'],
        forum_log_path
    )


def run_report_generation(task: ReportTask, query: str, custom_template: str = ""):
    """
    Run report generation in a background thread.

    Includes: input check → document loading → ReportAgent invocation → output persistence →
    pushing stage events. Errors are automatically pushed and recorded in status.

    Args:
        task: The task object for this run, holding the internal event queue.
        query: The report topic.
        custom_template: Optional custom template string.
    """
    global current_task

    try:
        # Encapsulate push logic within local closure for passing to ReportAgent
        def stream_handler(event_type: str, payload: Dict[str, Any]):
            """All stage events are dispatched through a unified interface to ensure log consistency."""
            task.publish_event(event_type, payload)
            # Sync update task progress if event contains progress info
            if event_type == 'progress' and 'progress' in payload:
                task.update_status("running", payload['progress'])

        task.update_status("running", 5)
        task.publish_event('stage', {'message': 'Task started, checking input files', 'stage': 'prepare'})

        # Check input files
        check_result = check_engines_ready()
        if not check_result['ready']:
            task.update_status("error", 0, f"Input files not ready: {check_result.get('missing_files', [])}")
            return

        task.publish_event('stage', {
            'message': 'Input files verified, preparing to load content',
            'stage': 'io_ready',
            'files': check_result.get('latest_files', {})
        })

        # Load input files
        content = report_agent.load_input_files(check_result['latest_files'])
        task.publish_event('stage', {'message': 'Source data loaded, starting generation process', 'stage': 'data_loaded'})

        # Generate report (with fallback retry to mitigate transient network jitter)
        for attempt in range(1, 3):
            try:
                task.publish_event('stage', {
                    'message': f'Invoking ReportAgent to generate report (attempt {attempt})',
                    'stage': 'agent_running',
                    'attempt': attempt
                })
                generation_result = report_agent.generate_report(
                    query=query,
                    reports=content['reports'],
                    forum_logs=content['forum_logs'],
                    custom_template=custom_template,
                    save_report=True,
                    stream_handler=stream_handler
                )
                break
            except ChapterJsonParseError as err:
                hint_message = "Try switching Report Engine's API to a more powerful LLM with longer context"
                task.publish_event('warning', {
                    'message': hint_message,
                    'stage': 'agent_running',
                    'attempt': attempt,
                    'reason': 'chapter_json_parse',
                    'error': str(err),
                    'task': task.to_dict(),
                })
                # Legacy logic: restart Report Engine after JSON parse failure
                # backoff = min(5 * attempt, 15)
                # task.publish_event('stage', {
                #     'message': f'Retrying generation task in {backoff} seconds',
                #     'stage': 'retry_wait',
                #     'wait_seconds': backoff
                # })
                # time.sleep(backoff)
                raise ChapterJsonParseError(hint_message) from err
            except Exception as err:
                # Push errors to frontend immediately for retry strategy observation
                task.publish_event('warning', {
                    'message': f'ReportAgent execution failed: {str(err)}',
                    'stage': 'agent_running',
                    'attempt': attempt
                })
                if attempt == 2:
                    raise
                # Simple exponential backoff to prevent frequent rate limiting (in seconds)
                backoff = min(5 * attempt, 15)
                task.publish_event('stage', {
                    'message': f'Retrying generation task in {backoff} seconds',
                    'stage': 'retry_wait',
                    'wait_seconds': backoff
                })
                time.sleep(backoff)

        if isinstance(generation_result, dict):
            html_report = generation_result.get('html_content', '')
        else:
            html_report = generation_result

        task.publish_event('stage', {'message': 'Report generation complete, preparing persistence', 'stage': 'persist'})

        # Save results
        task.html_content = html_report
        if isinstance(generation_result, dict):
            task.report_file_path = generation_result.get('report_filepath', '')
            task.report_file_relative_path = generation_result.get('report_relative_path', '')
            task.report_file_name = generation_result.get('report_filename', '')
            task.state_file_path = generation_result.get('state_filepath', '')
            task.state_file_relative_path = generation_result.get('state_relative_path', '')
            task.ir_file_path = generation_result.get('ir_filepath', '')
            task.ir_file_relative_path = generation_result.get('ir_relative_path', '')
        task.publish_event('html_ready', {
            'message': 'HTML rendering complete, refresh to preview',
            'report_file': task.report_file_relative_path or task.report_file_path,
            'state_file': task.state_file_relative_path or task.state_file_path,
            'task': task.to_dict(),
        })
        task.update_status("completed", 100)
        task.publish_event('completed', {
            'message': 'Task completed',
            'duration_seconds': (task.updated_at - task.created_at).total_seconds(),
            'report_file': task.report_file_relative_path or task.report_file_path,
            'task': task.to_dict(),
        })

    except Exception as e:
        logger.exception(f"Error occurred during report generation: {str(e)}")
        task.update_status("error", 0, str(e))
        task.publish_event('error', {
            'message': str(e),
            'stage': 'failed',
            'task': task.to_dict(),
        })
        # Only clean up task on error
        with task_lock:
            if current_task and current_task.task_id == task.task_id:
                current_task = None


@report_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get Report Engine status, including engine readiness and current task information.

    Returns:
        Response: JSON structure containing initialized/engines_ready/current_task etc.
    """
    try:
        engines_status = check_engines_ready()

        return jsonify({
            'success': True,
            'initialized': report_agent is not None,
            'engines_ready': engines_status['ready'],
            'files_found': engines_status.get('files_found', []),
            'missing_files': engines_status.get('missing_files', []),
            'current_task': current_task.to_dict() if current_task else None
        })
    except Exception as e:
        logger.exception(f"Failed to get Report Engine status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """
    Start report generation.

    Responsible for queuing, creating background threads, clearing logs, and returning SSE endpoint.

    Request Body:
        query: Report topic (optional).
        custom_template: Custom template string (optional).

    Returns:
        Response: JSON containing task_id and SSE stream url.
    """
    global current_task

    try:
        # Check if a task is already running
        with task_lock:
            if current_task and current_task.status == "running":
                return jsonify({
                    'success': False,
                    'error': 'A report generation task is already running',
                    'current_task': current_task.to_dict()
                }), 400

            # Clean up completed task if exists
            if current_task and current_task.status in ["completed", "error"]:
                current_task = None

        # Get request parameters
        data = request.get_json() or {}
        if not isinstance(data, dict):
            logger.warning("generate_report received non-object JSON payload, ignoring original content")
            data = {}
        query = data.get('query', 'Intelligent Sentiment Analysis Report')
        custom_template = data.get('custom_template', '')

        # Clear log file
        clear_report_log()

        # Check if Report Engine is initialized
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Engine not initialized'
            }), 500

        # Check if input files are ready
        engines_status = check_engines_ready()
        if not engines_status['ready']:
            return jsonify({
                'success': False,
                'error': 'Input files not ready',
                'missing_files': engines_status.get('missing_files', [])
            }), 400

        # Create new task
        task_id = f"report_{int(time.time())}"
        task = ReportTask(query, task_id, custom_template)

        with task_lock:
            current_task = task
            tasks_registry[task_id] = task
            _prune_task_history_locked()

        # Proactively push pending event to inform frontend that task is queued
        task.publish_event(
            'status',
            {
                'status': task.status,
                'progress': task.progress,
                'message': 'Task queued, waiting for resources',
                'task': task.to_dict(),
            }
        )

        # Run report generation in background thread
        thread = threading.Thread(
            target=run_report_generation,
            args=(task, query, custom_template),
            daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Report generation started',
            'task': task.to_dict(),
            'stream_url': f"/api/report/stream/{task_id}"
        })

    except Exception as e:
        logger.exception(f"Failed to start report generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id: str):
    """
    Get report generation progress, returns a completed fallback if task was cleaned up.

    Args:
        task_id: Unique task identifier.

    Returns:
        Response: JSON containing current task status.
    """
    try:
        task = _get_task(task_id)
        if not task:
            # If task doesn't exist, history may have been cleaned up, return a completed fallback
            return jsonify({
                'success': True,
                'task': {
                    'task_id': task_id,
                    'status': 'completed',
                    'progress': 100,
                    'error_message': '',
                    'has_result': True,
                    'report_file_ready': False,
                    'report_file_name': '',
                    'report_file_path': '',
                    'state_file_ready': False,
                    'state_file_path': ''
                }
            })

        return jsonify({
            'success': True,
            'task': task.to_dict()
        })

    except Exception as e:
        logger.exception(f"Failed to get report generation progress: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/stream/<task_id>', methods=['GET'])
def stream_task(task_id: str):
    """
    SSE-based real-time push interface.

    - Automatically replay historical events after Last-Event-ID;
    - Send periodic heartbeats to prevent proxy timeouts;
    - Automatically unregister listener after task completion.

    Args:
        task_id: Unique task identifier.

    Returns:
        Response: Response of type `text/event-stream`.
    """
    task = _get_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': 'Task does not exist'}), 404

    last_event_header = request.headers.get('Last-Event-ID')
    try:
        last_event_id = int(last_event_header) if last_event_header else None
    except ValueError:
        last_event_id = None

    def client_disconnected() -> bool:
        """
        Detect early if client has disconnected to avoid triggering BrokenPipe on continued writes.

        Eventlet on Windows throws ConnectionAbortedError when closing connections,
        early exit from generator can reduce meaningless logs.
        """
        try:
            env_input = request.environ.get('wsgi.input')
            return bool(getattr(env_input, 'closed', False))
        except Exception:
            return False

    def event_generator():
        """
        SSE event generator.

        - Responsible for registering and consuming the event queue for the corresponding task;
        - Replay historical events first, then continuously listen for real-time events;
        - Send periodic heartbeats and automatically unregister listener after task completion.
        """
        queue = _register_stream(task_id)
        last_data_ts = time.time()
        try:
            # In reconnect scenario, replay historical events first to ensure UI state consistency
            history = task.history_since(last_event_id)
            for event in history:
                yield _format_sse(event)
                if event.get('type') != 'heartbeat':
                    last_data_ts = time.time()

            finished = task.status in STREAM_TERMINAL_STATUSES
            while True:
                if finished:
                    break
                if client_disconnected():
                    logger.info(f"SSE client disconnected, stopping push: {task_id}")
                    break
                event = None
                try:
                    event = queue.get(timeout=STREAM_HEARTBEAT_INTERVAL)
                except Empty:
                    if task.status in STREAM_TERMINAL_STATUSES:
                        logger.info(f"Task {task_id} ended with no new events, SSE auto-closing")
                        break
                    heartbeat = {
                        'id': f"hb-{int(time.time() * 1000)}",
                        'type': 'heartbeat',
                        'task_id': task_id,
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'payload': {'status': task.status}
                    }
                    event = heartbeat
                if event is None:
                    logger.warning(f"SSE push failed to get event (task {task_id}), ending early")
                    break

                try:
                    yield _format_sse(event)
                    if event.get('type') != 'heartbeat':
                        last_data_ts = time.time()
                except GeneratorExit:
                    logger.info(f"SSE generator closed, stopping push for task {task_id}")
                    break
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
                    logger.warning(f"SSE connection interrupted by client (task {task_id}): {exc}")
                    break
                except Exception as exc:
                    event_type = event.get('type') if isinstance(event, dict) else 'unknown'
                    logger.exception(f"SSE push failed (task {task_id}, event {event_type}): {exc}")
                    break

                if event.get('type') in ("completed", "error", "cancelled"):
                    finished = True
                else:
                    finished = finished or task.status in STREAM_TERMINAL_STATUSES

                # Keep alive for a limited time in terminal state to prevent backend loop from running after frontend ended
                if task.status in STREAM_TERMINAL_STATUSES:
                    idle_for = time.time() - last_data_ts
                    if idle_for > STREAM_IDLE_TIMEOUT:
                        logger.info(f"Task {task_id} in terminal state and idle for {int(idle_for)}s, actively closing SSE")
                        break
        finally:
            _unregister_stream(task_id, queue)

    response = Response(
        stream_with_context(event_generator()),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@report_bp.route('/result/<task_id>', methods=['GET'])
def get_result(task_id: str):
    """
    Get report generation result.

    Args:
        task_id: Task ID.

    Returns:
        Response: JSON containing HTML preview and file path.
    """
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != "completed":
            return jsonify({
                'success': False,
                'error': 'Report not yet completed',
                'task': task.to_dict()
            }), 400

        return Response(
            task.html_content,
            mimetype='text/html'
        )

    except Exception as e:
        logger.exception(f"Failed to get report generation result: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/result/<task_id>/json', methods=['GET'])
def get_result_json(task_id: str):
    """Get report generation result (JSON format)"""
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != "completed":
            return jsonify({
                'success': False,
                'error': 'Report not yet completed',
                'task': task.to_dict()
            }), 400

        return jsonify({
            'success': True,
            'task': task.to_dict(),
            'html_content': task.html_content
        })

    except Exception as e:
        logger.exception(f"Failed to get report generation result: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/download/<task_id>', methods=['GET'])
def download_report(task_id: str):
    """
    Download the generated report HTML file.

    Args:
        task_id: Task ID.

    Returns:
        Response: Attachment download response for the HTML file.
    """
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != "completed" or not task.report_file_path:
            return jsonify({
                'success': False,
                'error': 'Report not yet completed or not saved'
            }), 400

        if not os.path.exists(task.report_file_path):
            return jsonify({
                'success': False,
                'error': 'Report file does not exist or has been deleted'
            }), 404

        download_name = task.report_file_name or os.path.basename(task.report_file_path)
        return send_file(
            task.report_file_path,
            mimetype='text/html',
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        logger.exception(f"Failed to download report: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id: str):
    """
    Cancel report generation task.

    Args:
        task_id: The task ID to be cancelled.

    Returns:
        Response: JSON containing cancellation result or error message.
    """
    global current_task

    try:
        with task_lock:
            if current_task and current_task.task_id == task_id:
                if current_task.status == "running":
                    current_task.update_status("cancelled", 0, "User cancelled task")
                    current_task.publish_event('cancelled', {
                        'message': 'Task terminated by user',
                        'task': current_task.to_dict(),
                    })
                current_task = None
            task = tasks_registry.get(task_id)
            if task and task.status == 'running':
                task.update_status("cancelled", task.progress, "User cancelled task")
                task.publish_event('cancelled', {
                    'message': 'Task terminated by user',
                    'task': task.to_dict(),
                })

                return jsonify({
                    'success': True,
                    'message': 'Task cancelled'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Task does not exist or cannot be cancelled'
                }), 404

    except Exception as e:
        logger.exception(f"Failed to cancel report generation task: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/templates', methods=['GET'])
def get_templates():
    """
    Get available template list for frontend to display optional Markdown skeletons.

    Returns:
        Response: JSON listing template name/description/size.
    """
    try:
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Engine not initialized'
            }), 500

        template_dir = settings.TEMPLATE_DIR
        templates = []

        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.md'):
                    template_path = os.path.join(template_dir, filename)
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        templates.append({
                            'name': filename.replace('.md', ''),
                            'filename': filename,
                            'description': content.split('\n')[0] if content else 'No description',
                            'size': len(content)
                        })
                    except Exception as e:
                        logger.exception(f"Failed to read template {filename}: {str(e)}")

        return jsonify({
            'success': True,
            'templates': templates,
            'template_dir': template_dir
        })

    except Exception as e:
        logger.exception(f"Failed to get available template list: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handling
@report_bp.errorhandler(404)
def not_found(error):
    """404 fallback handler: ensure API returns unified JSON structure"""
    logger.exception(f"API endpoint does not exist: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'API endpoint does not exist'
    }), 404


@report_bp.errorhandler(500)
def internal_error(error):
    """500 fallback handler: catch exceptions not actively caught"""
    logger.exception(f"Server internal error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Server internal error'
    }), 500


def clear_report_log():
    """
    Clear report.log file so new tasks only see logs from the current run.

    Returns:
        None
    """
    try:
        log_file = settings.LOG_FILE

        # [Fix] Use truncate instead of reopening to avoid conflicts with logger's file handle
        # Open in append mode, then truncate, keeping the file handle valid
        with open(log_file, 'r+', encoding='utf-8') as f:
            f.truncate(0)  # Clear file content without closing the file
            f.flush()      # Flush immediately

        logger.info(f"Cleared log file: {log_file}")
    except FileNotFoundError:
        # File doesn't exist, create empty file
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')
            logger.info(f"Created log file: {log_file}")
        except Exception as e:
            logger.exception(f"Failed to create log file: {str(e)}")
    except Exception as e:
        logger.exception(f"Failed to clear log file: {str(e)}")


@report_bp.route('/log', methods=['GET'])
def get_report_log():
    """
    Get report.log content, stripping whitespace from each line.

    [Fix] Optimized large file reading, added error handling and file locking

    Returns:
        Response: JSON containing array of latest log lines.
    """
    try:
        log_file = settings.LOG_FILE

        if not os.path.exists(log_file):
            return jsonify({
                'success': True,
                'log_lines': []
            })

        # [Fix] Check file size to avoid memory issues when reading large files
        file_size = os.path.getsize(log_file)
        max_size = 10 * 1024 * 1024  # 10MB limit

        if file_size > max_size:
            # File too large, only read last 10MB
            with open(log_file, 'rb') as f:
                f.seek(-max_size, 2)  # Seek 10MB from end of file
                # Skip potentially incomplete first line
                f.readline()
                content = f.read().decode('utf-8', errors='replace')
            lines = content.splitlines()
            logger.warning(f"Log file too large ({file_size} bytes), returning only last {max_size} bytes")
        else:
            # Normal size, read completely
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

        # Strip trailing newlines and empty lines
        log_lines = [line.rstrip('\n\r') for line in lines if line.strip()]

        return jsonify({
            'success': True,
            'log_lines': log_lines
        })

    except PermissionError as e:
        logger.error(f"Insufficient permission to read log: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Insufficient permission to read log'
        }), 403
    except UnicodeDecodeError as e:
        logger.error(f"Log file encoding error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Log file encoding error'
        }), 500
    except Exception as e:
        logger.exception(f"Failed to read log: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to read log: {str(e)}'
        }), 500


@report_bp.route('/log/clear', methods=['POST'])
def clear_log():
    """
    Manually clear logs, providing REST endpoint for frontend one-click reset.

    Returns:
        Response: JSON indicating whether cleanup succeeded.
    """
    try:
        clear_report_log()
        return jsonify({
            'success': True,
            'message': 'Log cleared'
        })
    except Exception as e:
        logger.exception(f"Failed to clear log: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to clear log: {str(e)}'
        }), 500


@report_bp.route('/export/md/<task_id>', methods=['GET'])
def export_markdown(task_id: str):
    """
    Export report in Markdown format.

    Calls MarkdownRenderer based on saved Document IR to generate file and return download.
    """
    try:
        task = tasks_registry.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != 'completed':
            return jsonify({
                'success': False,
                'error': f'Task not completed, current status: {task.status}'
            }), 400

        if not task.ir_file_path or not os.path.exists(task.ir_file_path):
            return jsonify({
                'success': False,
                'error': 'IR file does not exist, cannot generate Markdown'
            }), 404

        with open(task.ir_file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        from .renderers import MarkdownRenderer
        renderer = MarkdownRenderer()
        # Pass ir_file_path so fixed charts are automatically saved to IR file
        markdown_text = renderer.render(document_ir, ir_file_path=task.ir_file_path)

        metadata = document_ir.get('metadata') if isinstance(document_ir, dict) else {}
        topic = (metadata or {}).get('topic') or (metadata or {}).get('title') or (metadata or {}).get('query') or task.query
        safe_topic = _safe_filename_segment(topic or 'report')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{safe_topic}_{timestamp}.md"

        output_dir = Path(settings.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / filename
        md_path.write_text(markdown_text, encoding='utf-8')

        task.markdown_file_path = str(md_path.resolve())
        task.markdown_file_relative_path = os.path.relpath(task.markdown_file_path, os.getcwd())
        task.markdown_file_name = filename

        logger.info(f"Markdown export completed: {md_path}")

        return send_file(
            task.markdown_file_path,
            mimetype='text/markdown',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.exception(f"Markdown export failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Markdown export failed: {str(e)}'
        }), 500


@report_bp.route('/export/pdf/<task_id>', methods=['GET'])
def export_pdf(task_id: str):
    """
    Export report in PDF format.

    Generates optimized PDF from IR JSON file, supporting automatic layout adjustment.

    Args:
        task_id: Task ID

    Query Parameters:
        optimize: Whether to enable layout optimization (default true)

    Returns:
        Response: PDF file stream or error message
    """
    try:
        # Check Pango dependency
        from .utils.dependency_check import check_pango_available
        pango_available, pango_message = check_pango_available()
        if not pango_available:
            return jsonify({
                'success': False,
                'error': 'PDF export unavailable: missing system dependencies',
                'details': 'Please refer to README.md in the root directory, step 2 of "Source Startup" (PDF export dependencies) for installation instructions',
                'system_message': pango_message
            }), 503

        # Get task information
        task = tasks_registry.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        # Check if task is completed
        if task.status != 'completed':
            return jsonify({
                'success': False,
                'error': f'Task not completed, current status: {task.status}'
            }), 400

        # Get IR file path
        if not task.ir_file_path or not os.path.exists(task.ir_file_path):
            return jsonify({
                'success': False,
                'error': 'IR file does not exist'
            }), 404

        # Read IR data
        with open(task.ir_file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        # Check if layout optimization is enabled
        optimize = request.args.get('optimize', 'true').lower() == 'true'

        # Create PDF renderer and generate PDF
        from .renderers import PDFRenderer
        renderer = PDFRenderer()

        logger.info(f"Starting PDF export, task ID: {task_id}, layout optimization: {optimize}")

        # Generate PDF byte stream
        pdf_bytes = renderer.render_to_bytes(document_ir, optimize_layout=optimize)

        # Determine download filename
        topic = document_ir.get('metadata', {}).get('topic', 'report')
        pdf_filename = f"report_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Return PDF file
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{pdf_filename}"',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        logger.exception(f"PDF export failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'PDF export failed: {str(e)}'
        }), 500


@report_bp.route('/export/pdf-from-ir', methods=['POST'])
def export_pdf_from_ir():
    """
    Export PDF directly from IR JSON (no task ID required).

    Suitable for scenarios where frontend directly passes IR data.

    Request Body:
        {
            "document_ir": {...},  // Document IR JSON
            "optimize": true       // Whether to enable layout optimization (optional)
        }

    Returns:
        Response: PDF file stream or error message
    """
    try:
        # Check Pango dependency
        from .utils.dependency_check import check_pango_available
        pango_available, pango_message = check_pango_available()
        if not pango_available:
            return jsonify({
                'success': False,
                'error': 'PDF export unavailable: missing system dependencies',
                'details': 'Please refer to README.md in the root directory, step 2 of "Source Startup" (PDF export dependencies) for installation instructions',
                'system_message': pango_message
            }), 503

        data = request.get_json() or {}
        if not isinstance(data, dict):
            logger.warning("export_pdf_from_ir request body is not a JSON object")
            return jsonify({
                'success': False,
                'error': 'Request body must be a JSON object'
            }), 400

        if not data or 'document_ir' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing document_ir parameter'
            }), 400

        document_ir = data['document_ir']
        optimize = data.get('optimize', True)

        # Create PDF renderer and generate PDF
        from .renderers import PDFRenderer
        renderer = PDFRenderer()

        logger.info(f"Exporting PDF directly from IR, layout optimization: {optimize}")

        # Generate PDF byte stream
        pdf_bytes = renderer.render_to_bytes(document_ir, optimize_layout=optimize)

        # Determine download filename
        topic = document_ir.get('metadata', {}).get('topic', 'report')
        pdf_filename = f"report_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Return PDF file
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{pdf_filename}"',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        logger.exception(f"PDF export from IR failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'PDF export failed: {str(e)}'
        }), 500
