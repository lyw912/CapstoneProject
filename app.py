"""
Flask Main Application - Unified management of three Streamlit applications
"""

import os
import sys
import json
import uuid

# [FIX] Set environment variables early to ensure all modules use unbuffered mode
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering to ensure real-time log output

import subprocess
import socket
import time
import threading
from datetime import datetime, timedelta
from queue import Queue
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import atexit
import requests
from loguru import logger
import importlib
from pathlib import Path

from config import settings
from utils.sensitive_input_filter import reject_if_sensitive

# Import ReportEngine
try:
    from ReportEngine.flask_interface import report_bp, initialize_report_engine
    REPORT_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"ReportEngine import failed: {e}")
    REPORT_ENGINE_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Dedicated-to-creating-a-concise-and-versatile-public-opinion-analysis-platform'
socketio = SocketIO(app, cors_allowed_origins="*")

# eventlet occasionally throws ConnectionAbortedError when client disconnects actively,
# wrap it defensively here to avoid meaningless stack trace pollution in logs
# (only enabled when eventlet is available).
def _patch_eventlet_disconnect_logging():
    try:
        import eventlet.wsgi  # type: ignore
    except Exception as exc:  # pragma: no cover - only effective in production
        logger.debug(f"eventlet not available, skipping disconnect patch: {exc}")
        return

    try:
        original_finish = eventlet.wsgi.HttpProtocol.finish  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        logger.debug(f"eventlet missing HttpProtocol.finish, skipping disconnect patch: {exc}")
        return

    def _safe_finish(self, *args, **kwargs):  # pragma: no cover - triggered at runtime
        try:
            return original_finish(self, *args, **kwargs)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            try:
                environ = getattr(self, 'environ', {}) or {}
                method = environ.get('REQUEST_METHOD', '')
                path = environ.get('PATH_INFO', '')
                logger.info(f"Client actively disconnected, ignoring exception: {method} {path} ({exc})")
            except Exception:
                logger.info(f"Client actively disconnected, ignoring exception: {exc}")
            return

    eventlet.wsgi.HttpProtocol.finish = _safe_finish  # type: ignore[attr-defined]
    logger.info("Applied defensive protection for eventlet connection interruptions")

_patch_eventlet_disconnect_logging()

# Register ReportEngine Blueprint
if REPORT_ENGINE_AVAILABLE:
    app.register_blueprint(report_bp, url_prefix='/api/report')
    logger.info("ReportEngine interface registered")
else:
    logger.info("ReportEngine unavailable, skipping interface registration")

# Create log directory
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

CONFIG_MODULE_NAME = 'config'
CONFIG_FILE_PATH = Path(__file__).resolve().parent / 'config.py'
CONFIG_KEYS = [
    'HOST',
    'PORT',
    'MEDIA_ENGINE_API_KEY',
    'MEDIA_ENGINE_BASE_URL',
    'MEDIA_ENGINE_MODEL_NAME',
    'QUERY_ENGINE_API_KEY',
    'QUERY_ENGINE_BASE_URL',
    'QUERY_ENGINE_MODEL_NAME',
    'REPORT_ENGINE_API_KEY',
    'REPORT_ENGINE_BASE_URL',
    'REPORT_ENGINE_MODEL_NAME',
    'FORUM_HOST_API_KEY',
    'FORUM_HOST_BASE_URL',
    'FORUM_HOST_MODEL_NAME',
    'KEYWORD_OPTIMIZER_API_KEY',
    'KEYWORD_OPTIMIZER_BASE_URL',
    'KEYWORD_OPTIMIZER_MODEL_NAME',
    'LANGSMITH_TRACING',
    'LANGSMITH_API_KEY',
    'LANGSMITH_ENDPOINT',
    'LANGSMITH_PROJECT',
    'LANGCHAIN_TRACING_V2',
    'LANGCHAIN_PROJECT',
    'TAVILY_API_KEY',
    'SEARCH_TOOL_TYPE',
    'BOCHA_WEB_SEARCH_API_KEY',
    'ANSPIRE_API_KEY',
    'COORDINATOR_MEDIA_AGENT_TIMEOUT',
    'COORDINATOR_QUERY_AGENT_TIMEOUT',
]


def _load_config_module():
    """Load or reload the config module to ensure latest values are available."""
    importlib.invalidate_caches()
    module = sys.modules.get(CONFIG_MODULE_NAME)
    try:
        if module is None:
            module = importlib.import_module(CONFIG_MODULE_NAME)
        else:
            module = importlib.reload(module)
    except ModuleNotFoundError:
        return None
    return module


def read_config_values():
    """Return the current configuration values that are exposed to the frontend."""
    try:
        # Reload config to get the latest Settings instance
        from config import reload_settings, settings
        reload_settings()
        
        values = {}
        for key in CONFIG_KEYS:
        # Read value from Pydantic Settings instance
            value = getattr(settings, key, None)
            # Convert to string for uniform handling on the frontend.
            if value is None:
                values[key] = ''
            else:
                values[key] = str(value)
        return values
    except Exception as exc:
        logger.exception(f"Failed to read config: {exc}")
        return {}


def _serialize_config_value(value):
    """Serialize Python values back to a config.py assignment-friendly string."""
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return 'None'

    value_str = str(value)
    escaped = value_str.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def write_config_values(updates):
    """Persist configuration updates to .env file (Pydantic Settings source)."""
    from pathlib import Path
    
    # Determine .env file path (consistent with logic in config.py)
    project_root = Path(__file__).resolve().parent
    cwd_env = Path.cwd() / ".env"
    env_file_path = cwd_env if cwd_env.exists() else (project_root / ".env")
    
    # Read existing .env file content
    env_lines = []
    env_key_indices = {}  # Track index position of each key in the file
    if env_file_path.exists():
        env_lines = env_file_path.read_text(encoding='utf-8').splitlines()
        # Extract existing keys and their indices
        for i, line in enumerate(env_lines):
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                if '=' in line_stripped:
                    key = line_stripped.split('=')[0].strip()
                    env_key_indices[key] = i
    
    # Update or add configuration items
    for key, raw_value in updates.items():
        # Format value for .env file (no quotes needed unless string contains spaces)
        if raw_value is None or raw_value == '':
            env_value = ''
        elif isinstance(raw_value, (int, float)):
            env_value = str(raw_value)
        elif isinstance(raw_value, bool):
            env_value = 'True' if raw_value else 'False'
        else:
            value_str = str(raw_value)
            # Quotes needed if contains spaces or special characters
            if ' ' in value_str or '\n' in value_str or '#' in value_str:
                escaped = value_str.replace('\\', '\\\\').replace('"', '\\"')
                env_value = f'"{escaped}"'
            else:
                env_value = value_str
        
        # Update or add configuration item
        if key in env_key_indices:
            # Update existing line
            env_lines[env_key_indices[key]] = f'{key}={env_value}'
        else:
            # Add new line to end of file
            env_lines.append(f'{key}={env_value}')
    
    # Write to .env file
    env_file_path.parent.mkdir(parents=True, exist_ok=True)
    env_file_path.write_text('\n'.join(env_lines) + '\n', encoding='utf-8')
    
    # Reload config module (this re-reads the .env file and creates a new Settings instance)
    _load_config_module()


system_state_lock = threading.Lock()
system_state = {
    'started': False,
    'starting': False,
    'shutdown_in_progress': False
}

COORDINATOR_CACHE_DIR = Path('AgentCoordinator/cache')
COORDINATOR_LATEST_OUTPUT = COORDINATOR_CACHE_DIR / 'coordinator_output_latest.json'
COORDINATOR_FEEDBACK_LOG = COORDINATOR_CACHE_DIR / 'frontend_feedback.jsonl'
coordinator_task_lock = threading.Lock()
coordinator_tasks = {}


def _set_system_state(*, started=None, starting=None):
    """Safely update the cached system state flags."""
    with system_state_lock:
        if started is not None:
            system_state['started'] = started
        if starting is not None:
            system_state['starting'] = starting


def _get_system_state():
    """Return a shallow copy of the system state flags."""
    with system_state_lock:
        return system_state.copy()


def _prepare_system_start():
    """Mark the system as starting if it is not already running or starting."""
    with system_state_lock:
        if system_state['started']:
            return False, 'System already started'
        if system_state['starting']:
            return False, 'System is starting'
        system_state['starting'] = True
        return True, None

def _mark_shutdown_requested():
    """Mark shutdown as requested; returns False if shutdown is already in progress."""
    with system_state_lock:
        if system_state.get('shutdown_in_progress'):
            return False
        system_state['shutdown_in_progress'] = True
        return True


def initialize_system_components():
    """Start the final demo runtime: Flask APIs, React static UI, Coordinator, and ReportEngine.

    Streamlit Media/Query apps are legacy operator surfaces. The final React
    frontend calls QueryEngine/MediaEngine through AgentCoordinator and passes
    the Coordinator artifact to ReportEngine, so the Streamlit processes are
    deliberately kept stopped for the final demo.
    """
    logs = []
    errors = []

    for app_name in STREAMLIT_SCRIPTS:
        try:
            success, message = stop_streamlit_app(app_name)
            processes[app_name]['status'] = 'stopped'
            logs.append(f"{app_name} Streamlit disabled for final demo: {message if success else 'not running'}")
        except Exception as exc:  # pragma: no cover - safe catch
            logs.append(f"{app_name} Streamlit stop skipped: {exc}")
            logger.exception(f"Failed to stop legacy Streamlit app: {app_name}")

    try:
        stop_forum_engine()
        processes['forum']['status'] = 'stopped'
        logs.append("ForumEngine monitor disabled for final demo")
    except Exception as exc:  # pragma: no cover - safe catch
        logs.append(f"ForumEngine stop skipped: {exc}")
        logger.exception("Failed to stop ForumEngine monitor")

    if REPORT_ENGINE_AVAILABLE:
        try:
            if initialize_report_engine():
                logs.append("ReportEngine initialization successful")
            else:
                msg = "ReportEngine initialization failed"
                logs.append(msg)
                errors.append(msg)
        except Exception as exc:  # pragma: no cover
            msg = f"ReportEngine initialization exception: {exc}"
            logs.append(msg)
            errors.append(msg)
    else:
        errors.append("ReportEngine is not available")

    if errors:
        return False, logs, errors

    return True, logs, []

# Initialize ForumEngine's forum.log file
def init_forum_log():
    """Initialize forum.log file"""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        # Create file if not exists and write start marker, clear and rewrite if exists
        if not forum_log_file.exists():
            with open(forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEngine System Initialization - {start_time} ===\n")
            logger.info(f"ForumEngine: forum.log initialized")
        else:
            with open(forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEngine System Initialization - {start_time} ===\n")
            logger.info(f"ForumEngine: forum.log initialized")
    except Exception as e:
        logger.exception(f"ForumEngine: Failed to initialize forum.log: {e}")

# Initialize forum.log
init_forum_log()

# Start ForumEngine intelligent monitoring
def start_forum_engine():
    """Start ForumEngine forum"""
    try:
        from ForumEngine.monitor import start_forum_monitoring
        logger.info("ForumEngine: Starting forum...")
        success = start_forum_monitoring()
        if not success:
            logger.info("ForumEngine: Forum startup failed")
    except Exception as e:
        logger.exception(f"ForumEngine: Failed to start forum: {e}")

# Stop ForumEngine intelligent monitoring
def stop_forum_engine():
    """Stop ForumEngine forum"""
    try:
        from ForumEngine.monitor import stop_forum_monitoring
        logger.info("ForumEngine: Stopping forum...")
        stop_forum_monitoring()
        logger.info("ForumEngine: Forum stopped")
    except Exception as e:
        logger.exception(f"ForumEngine: Failed to stop forum: {e}")

def parse_forum_log_line(line):
    """Parse forum.log line content and extract conversation information"""
    import re
    
    # Match format: [timestamp] [source] content (source allows case variations and spaces)
    pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*\[([^\]]+)\]\s*(.*)'
    match = re.match(pattern, line)
    
    if not match:
        return None

    timestamp, raw_source, content = match.groups()
    source = raw_source.strip().upper()

    # Filter out system messages and empty content
    if source == 'SYSTEM' or not content.strip():
        return None
    
    # Support two analysis Agents and coordinator (log labels remain MEDIA / QUERY / HOST)
    if source not in ['QUERY', 'MEDIA', 'HOST']:
        return None
    
    # Decode escaped newlines in log, preserve multi-line format
    cleaned_content = content.replace('\\n', '\n').replace('\\r', '').strip()
    
    # Determine message type and sender based on source (consistent with Proposal naming)
    if source == 'HOST':
        message_type = 'host'
        sender = 'Agent Coordinator'
    elif source == 'MEDIA':
        message_type = 'agent'
        sender = 'Multimodal Agent'
    elif source == 'QUERY':
        message_type = 'agent'
        sender = 'Query Agent'
    else:
        message_type = 'agent'
        sender = source.title()
    
    return {
        'type': message_type,
        'sender': sender,
        'content': cleaned_content,
        'timestamp': timestamp,
        'source': source
    }

# Forum log listener
# Store historical log send position for each client
forum_log_positions = {}

def monitor_forum_log():
    """Monitor forum.log file changes and push to frontend"""
    import time
    from pathlib import Path

    forum_log_file = LOG_DIR / "forum.log"
    last_position = 0
    processed_lines = set()  # Track processed lines to avoid duplicates

    # If file exists, get initial position but don't skip content
    if forum_log_file.exists():
        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Record file size but don't add to processed_lines
            # This allows users to get history when opening forum tab
            f.seek(0, 2)  # Move to end of file
            last_position = f.tell()

    while True:
        try:
            if forum_log_file.exists():
                with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()

                    if new_lines:
                        for line in new_lines:
                            line = line.rstrip('\n\r')
                            if line.strip():
                                line_hash = hash(line.strip())

                                # Avoid processing the same line repeatedly
                                if line_hash in processed_lines:
                                    continue

                                processed_lines.add(line_hash)

                                # Parse log line and send forum message
                                parsed_message = parse_forum_log_line(line)
                                if parsed_message:
                                    socketio.emit('forum_message', parsed_message)

                                # Only send console messages when forum is displayed in console
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                formatted_line = f"[{timestamp}] {line}"
                                socketio.emit('console_output', {
                                    'app': 'forum',
                                    'line': formatted_line
                                })

                        last_position = f.tell()

                        # Clean up processed_lines set to avoid memory leak (keep last 1000 line hashes)
                        if len(processed_lines) > 1000:
                            # Keep last 500 line hashes
                            recent_hashes = list(processed_lines)[-500:]
                            processed_lines = set(recent_hashes)

            time.sleep(1)  # Check every second
        except Exception as e:
            logger.error(f"Forum log monitoring error: {e}")
            time.sleep(5)

# Start Forum log monitoring thread
forum_monitor_thread = threading.Thread(target=monitor_forum_log, daemon=True)
forum_monitor_thread.start()

# Global variables to store process information
processes = {
    'media': {'process': None, 'port': 8502, 'status': 'stopped', 'output': [], 'log_file': None, 'healthcheck_started_at': None},
    'query': {'process': None, 'port': 8503, 'status': 'stopped', 'output': [], 'log_file': None, 'healthcheck_started_at': None},
    'forum': {'process': None, 'port': None, 'status': 'stopped', 'output': [], 'log_file': None}  # Mark as running after startup
}

STREAMLIT_SCRIPTS = {
    'media': 'SingleEngineApp/media_engine_streamlit_app.py',
    'query': 'SingleEngineApp/query_engine_streamlit_app.py'
}

def _log_shutdown_step(message: str):
    """Unified logging of shutdown steps for troubleshooting."""
    logger.info(f"[Shutdown] {message}")


def _describe_running_children():
    """List currently alive child processes."""
    running = []
    for name, info in processes.items():
        proc = info.get('process')
        if proc is not None and proc.poll() is None:
            port_desc = f", port={info.get('port')}" if info.get('port') else ""
            running.append(f"{name}(pid={proc.pid}{port_desc})")
    return running

# Output queues
output_queues = {
    'media': Queue(),
    'query': Queue(),
    'forum': Queue()
}

def write_log_to_file(app_name, line):
    """Write log to file"""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()
    except Exception as e:
        logger.error(f"Error writing log for {app_name}: {e}")

def read_log_from_file(app_name, tail_lines=None):
    """Read log from file"""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        if not log_file_path.exists():
            return []
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
            
            if tail_lines:
                return lines[-tail_lines:]
            return lines
    except Exception as e:
        logger.exception(f"Error reading log for {app_name}: {e}")
        return []

def read_process_output(process, app_name):
    """Read process output and write to file"""
    import select
    import sys
    
    while True:
        try:
            if process.poll() is not None:
                # Process ended, read remaining output
                remaining_output = process.stdout.read()
                if remaining_output:
                    lines = remaining_output.decode('utf-8', errors='replace').split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            write_log_to_file(app_name, formatted_line)
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                break
            
            # Use non-blocking read
            if sys.platform == 'win32':
                # Use different method on Windows
                output = process.stdout.readline()
                if output:
                    line = output.decode('utf-8', errors='replace').strip()
                    if line:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        formatted_line = f"[{timestamp}] {line}"
                        
                        # Write to log file
                        write_log_to_file(app_name, formatted_line)
                        
                        # Send to frontend
                        socketio.emit('console_output', {
                            'app': app_name,
                            'line': formatted_line
                        })
                else:
                    # Brief sleep when no output
                    time.sleep(0.1)
            else:
                # Unix systems use select
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    output = process.stdout.readline()
                    if output:
                        line = output.decode('utf-8', errors='replace').strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            
                            # Write to log file
                            write_log_to_file(app_name, formatted_line)
                            
                            # Send to frontend
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                            
        except Exception as e:
            error_msg = f"Error reading output for {app_name}: {e}"
            logger.exception(error_msg)
            write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
            break


def _local_port_is_free(port: int) -> bool:
    """Check if local port is bindable (not being listened by other processes)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _windows_pids_listening_on_port(port: int) -> list[int]:
    """Parse netstat, return PIDs listening on specified port (deduplicated)."""
    suffix = f":{port}"
    pids: set[int] = set()
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            timeout=30,
        )
    except Exception:
        return []
    for line in result.stdout.splitlines():
        line = line.strip()
        parts = line.split()
        if len(parts) < 5 or parts[0] != "TCP":
            continue
        local = parts[1]
        if not local.endswith(suffix):
            continue
        if parts[3] != "LISTENING":
            continue
        try:
            pids.add(int(parts[4]))
        except ValueError:
            continue
    return list(pids)


def _try_release_port(port: int, app_name: str) -> tuple[bool, str]:
    """
    If port is occupied, try to terminate the occupying process (common when Streamlit didn't exit properly).
    Windows: use netstat to find PID then taskkill; other systems only prompt manual handling.
    """
    if _local_port_is_free(port):
        return True, ""

    if sys.platform == "win32":
        pids = _windows_pids_listening_on_port(port)
        my_pid = os.getpid()
        killed = []
        for pid in pids:
            if pid == my_pid:
                continue
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=30,
                )
                killed.append(pid)
            except Exception as exc:
                logger.warning(f"Failed to terminate process {pid} occupying port {port}: {exc}")
        if killed:
            msg = f"Port {port} was occupied, attempted to terminate process PIDs: {killed}"
            logger.info(msg)
            write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            time.sleep(0.5)

        if _local_port_is_free(port):
            return True, ""
        return False, (
            f"Port {port} is still occupied. Please close the program occupying this port, or end the corresponding process in Task Manager and retry."
        )

    return False, (
        f"Port {port} is occupied. Please terminate the process occupying this port and retry (e.g., lsof -i :{port} or ss -tlnp)."
    )


def start_streamlit_app(app_name, script_path, port):
    """Start Streamlit application"""
    try:
        if processes[app_name]['process'] is not None:
            return False, "Application already running"
        
        # Check if file exists
        if not os.path.exists(script_path):
            return False, f"File does not exist: {script_path}"

        # Clear log first, then release port (release process may write to log)
        log_file_path = LOG_DIR / f"{app_name}.log"
        if log_file_path.exists():
            log_file_path.unlink()

        ok, port_msg = _try_release_port(port, app_name)
        if not ok:
            write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {port_msg}")
            return False, port_msg
        
        # Create startup log
        start_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Starting {app_name} application..."
        write_log_to_file(app_name, start_msg)
        
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            script_path,
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            # '--logger.level', 'debug',  # Increase log verbosity
            '--logger.level', 'info',
            '--server.enableCORS', 'false'
        ]
        
        # Set environment variables to ensure UTF-8 encoding and reduce buffering
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
            'PYTHONUNBUFFERED': '1',  # Disable Python buffering
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        # Use current working directory instead of script directory
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,  # No buffering
            universal_newlines=False,
            cwd=os.getcwd(),
            env=env,
            encoding=None,  # Let us handle encoding manually
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        processes[app_name]['process'] = process
        processes[app_name]['status'] = 'starting'
        processes[app_name]['output'] = []
        processes[app_name]['healthcheck_started_at'] = time.time()
        
        # Start output reading thread
        output_thread = threading.Thread(
            target=read_process_output,
            args=(process, app_name),
            daemon=True
        )
        output_thread.start()
        
        return True, f"{app_name} application starting..."
        
    except Exception as e:
        error_msg = f"Startup failed: {str(e)}"
        write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return False, error_msg

def stop_streamlit_app(app_name):
    """Stop Streamlit application"""
    try:
        process = processes[app_name]['process']
        if process is None:
            _log_shutdown_step(f"{app_name} not running, skipping stop")
            return False, "Application not running"
        
        try:
            pid = process.pid
        except Exception:
            pid = 'unknown'

        _log_shutdown_step(f"Stopping {app_name} (pid={pid})")
        process.terminate()
        
        # Wait for process to end
        try:
            process.wait(timeout=5)
            _log_shutdown_step(f"{app_name} exit completed, returncode={process.returncode}")
        except subprocess.TimeoutExpired:
            _log_shutdown_step(f"{app_name} termination timeout, attempting forced kill (pid={pid})")
            process.kill()
            process.wait()
            _log_shutdown_step(f"{app_name} forced termination completed, returncode={process.returncode}")
        
        processes[app_name]['process'] = None
        processes[app_name]['status'] = 'stopped'
        processes[app_name]['healthcheck_started_at'] = None
        
        return True, f"{app_name} application stopped"
        
    except Exception as e:
        _log_shutdown_step(f"{app_name} stop failed: {e}")
        return False, f"Stop failed: {str(e)}"

HEALTHCHECK_PATH = "/_stcore/health"
HEALTHCHECK_PROXIES = {'http': None, 'https': None}
HEALTHCHECK_GRACE_SECONDS = 15


def _build_healthcheck_url(port):
    return f"http://127.0.0.1:{port}{HEALTHCHECK_PATH}"


def _healthcheck_grace_active(app_name: str) -> bool:
    """Check if healthcheck grace period is active for the given app."""
    started_at = processes.get(app_name, {}).get('healthcheck_started_at')
    if not started_at:
        return False
    return (time.time() - started_at) < HEALTHCHECK_GRACE_SECONDS


def _log_healthcheck_failure(app_name: str, exc: Exception):
    """Log healthcheck failure, respecting grace period."""
    if _healthcheck_grace_active(app_name):
        logger.debug(f"Starting {app_name}, please wait")
        return
    logger.warning(f"{app_name} healthcheck failed: {exc}")


def check_app_status():
    """Check application status"""
    for app_name, info in processes.items():
        if info['process'] is not None:
            if info['process'].poll() is None:
                # Process still running, check if port is accessible
                try:
                    response = requests.get(
                        _build_healthcheck_url(info['port']),
                        timeout=2,
                        proxies=HEALTHCHECK_PROXIES
                    )
                    if response.status_code == 200:
                        info['status'] = 'running'
                    else:
                        info['status'] = 'starting'
                except Exception as exc:
                    _log_healthcheck_failure(app_name, exc)
                    info['status'] = 'starting'
            else:
                # Process has ended
                info['process'] = None
                info['status'] = 'stopped'
                info['healthcheck_started_at'] = None

def wait_for_app_startup(app_name, max_wait_time=90):
    """Wait for application startup to complete"""
    import time
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        info = processes[app_name]
        if info['process'] is None:
            return False, "Process stopped"
        
        if info['process'].poll() is not None:
            return False, "Process startup failed"
        
        try:
            response = requests.get(
                _build_healthcheck_url(info['port']),
                timeout=2,
                proxies=HEALTHCHECK_PROXIES
            )
            if response.status_code == 200:
                info['status'] = 'running'
                return True, "Startup successful"
        except Exception as exc:
            _log_healthcheck_failure(app_name, exc)

        time.sleep(1)

    return False, "Startup timeout"

def cleanup_processes():
    """Clean up all processes"""
    _log_shutdown_step("Starting sequential cleanup of child processes")
    for app_name in STREAMLIT_SCRIPTS:
        stop_streamlit_app(app_name)

    processes['forum']['status'] = 'stopped'
    try:
        stop_forum_engine()
    except Exception:  # pragma: no cover
        logger.exception("Failed to stop ForumEngine")
    _log_shutdown_step("Child process cleanup completed")
    _set_system_state(started=False, starting=False)

def cleanup_processes_concurrent(timeout: float = 6.0):
    """Concurrently clean up all child processes, force kill remaining processes after timeout."""
    _log_shutdown_step(f"Starting concurrent cleanup of child processes (timeout {timeout}s)")
    _log_shutdown_step("Only terminating child processes started and tracked by current console, no port scan")
    running_before = _describe_running_children()
    if running_before:
        _log_shutdown_step("Currently alive child processes: " + ", ".join(running_before))
    else:
        _log_shutdown_step("No alive child processes detected, will still send shutdown commands")

    threads = []

    # Concurrently stop Streamlit child processes
    for app_name in STREAMLIT_SCRIPTS:
        t = threading.Thread(target=stop_streamlit_app, args=(app_name,), daemon=True)
        threads.append(t)
        t.start()

    # Concurrently stop ForumEngine
    forum_thread = threading.Thread(target=stop_forum_engine, daemon=True)
    threads.append(forum_thread)
    forum_thread.start()

    # Wait for all threads to complete, up to timeout seconds
    end_time = time.time() + timeout
    for t in threads:
        remaining = end_time - time.time()
        if remaining <= 0:
            break
        t.join(timeout=remaining)

    # Second check: force kill still-alive child processes
    for app_name in STREAMLIT_SCRIPTS:
        proc = processes[app_name]['process']
        if proc is not None and proc.poll() is None:
            try:
                _log_shutdown_step(f"{app_name} process still alive, triggering second termination (pid={proc.pid})")
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                try:
                    _log_shutdown_step(f"{app_name} second termination failed, attempting kill (pid={proc.pid})")
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    logger.warning(f"{app_name} process forced exit failed, continuing shutdown")
            finally:
                processes[app_name]['process'] = None
                processes[app_name]['status'] = 'stopped'

    processes['forum']['status'] = 'stopped'
    _log_shutdown_step("Concurrent cleanup ended, marking system as not started")
    _set_system_state(started=False, starting=False)

def _schedule_server_shutdown(delay_seconds: float = 0.1):
    """Exit as soon as possible after cleanup to avoid blocking current request."""
    def _shutdown():
        time.sleep(delay_seconds)
        try:
            socketio.stop()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"SocketIO stop exception, continuing exit: {exc}")
        _log_shutdown_step("SocketIO stop command sent, main process will exit soon")
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()

def _start_async_shutdown(cleanup_timeout: float = 3.0):
    """Asynchronously trigger cleanup and force exit to avoid blocking HTTP requests."""
    _log_shutdown_step(f"Shutdown command received, starting async cleanup (timeout {cleanup_timeout}s)")

    def _force_exit():
        _log_shutdown_step("Shutdown timeout, triggering forced exit")
        os._exit(0)

    # Hard timeout protection to ensure exit even if cleanup thread fails
    hard_timeout = cleanup_timeout + 2.0
    force_timer = threading.Timer(hard_timeout, _force_exit)
    force_timer.daemon = True
    force_timer.start()

    def _cleanup_and_exit():
        try:
            cleanup_processes_concurrent(timeout=cleanup_timeout)
        except Exception as exc:  # pragma: no cover
            logger.exception(f"Shutdown cleanup exception: {exc}")
        finally:
            _log_shutdown_step("Cleanup thread ended, scheduling main process exit")
            _schedule_server_shutdown(0.05)

    threading.Thread(target=_cleanup_and_exit, daemon=True).start()

# Register cleanup function
atexit.register(cleanup_processes)


def _latest_coordinator_archive():
    """Return the newest timestamped coordinator output when the fixed latest file is absent."""
    try:
        archives = sorted(
            COORDINATOR_CACHE_DIR.glob('coordinator_output_*.json'),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return archives[0] if archives else None
    except Exception:
        logger.exception("Failed to scan coordinator cache")
        return None


def _load_latest_coordinator_output():
    path = COORDINATOR_LATEST_OUTPUT if COORDINATOR_LATEST_OUTPUT.exists() else _latest_coordinator_archive()
    if not path or not path.exists():
        return None, None
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle), path


def _load_feedback_records(limit=20):
    if not COORDINATOR_FEEDBACK_LOG.exists():
        return []
    records = []
    try:
        with open(COORDINATOR_FEEDBACK_LOG, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed coordinator feedback record")
        return records[-limit:]
    except Exception:
        logger.exception("Failed to read coordinator feedback log")
        return []


def _feedback_summary(records):
    by_target = {}
    for record in records:
        target = str(record.get('target') or 'overall')
        by_target[target] = by_target.get(target, 0) + 1
    return {
        'count': len(records),
        'by_target': by_target,
        'latest': records[-1] if records else None,
    }


def _apply_observability_env():
    """Expose LangSmith/LangChain tracing settings to libraries that read os.environ."""
    try:
        from config import reload_settings, settings
        reload_settings()
        tracing_enabled = bool(getattr(settings, 'LANGSMITH_TRACING', False))
        endpoint = getattr(settings, 'LANGSMITH_ENDPOINT', None) or 'https://api.smith.langchain.com'
        project = getattr(settings, 'LANGSMITH_PROJECT', None) or 'public-opinion-analysis'
        legacy_tracing = getattr(settings, 'LANGCHAIN_TRACING_V2', None)
        legacy_project = getattr(settings, 'LANGCHAIN_PROJECT', None) or project
        api_key = getattr(settings, 'LANGSMITH_API_KEY', None)

        os.environ['LANGSMITH_TRACING'] = 'true' if tracing_enabled else 'false'
        os.environ['LANGCHAIN_TRACING_V2'] = 'true' if (legacy_tracing if legacy_tracing is not None else tracing_enabled) else 'false'
        os.environ['LANGSMITH_ENDPOINT'] = str(endpoint)
        os.environ['LANGSMITH_PROJECT'] = str(project)
        os.environ['LANGCHAIN_PROJECT'] = str(legacy_project)
        if api_key:
            os.environ['LANGSMITH_API_KEY'] = str(api_key)
        elif 'LANGSMITH_API_KEY' in os.environ and not tracing_enabled:
            os.environ.pop('LANGSMITH_API_KEY', None)

        return {
            'enabled': tracing_enabled,
            'project': project,
            'endpoint': endpoint,
            'api_key_configured': bool(api_key),
        }
    except Exception as exc:
        logger.exception("Failed to apply observability environment")
        return {
            'enabled': False,
            'project': '',
            'endpoint': '',
            'api_key_configured': False,
            'error': str(exc),
        }



def _iso_datetime(value):
    if not value:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _safe_decimal_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _duration_ms(start_time, end_time):
    if not start_time or not end_time:
        return None
    try:
        return max(0, round((end_time - start_time).total_seconds() * 1000))
    except Exception:
        return None


def _public_run_url(client, run, project):
    try:
        return client.get_run_url(run=run, project_name=project)
    except Exception:
        app_path = getattr(run, 'app_path', None)
        if app_path:
            return app_path
    return None


def _summarize_langsmith_run(run, client=None, project=''):
    duration = _duration_ms(getattr(run, 'start_time', None), getattr(run, 'end_time', None))
    error = getattr(run, 'error', None)
    run_id = str(getattr(run, 'id', '') or '')
    trace_id = str(getattr(run, 'trace_id', '') or '')
    return {
        'id': run_id,
        'trace_id': trace_id,
        'name': str(getattr(run, 'name', '') or 'Run'),
        'type': str(getattr(run, 'run_type', '') or 'unknown'),
        'status': 'error' if error else str(getattr(run, 'status', '') or 'success'),
        'error': str(error)[:500] if error else '',
        'start_time': _iso_datetime(getattr(run, 'start_time', None)),
        'end_time': _iso_datetime(getattr(run, 'end_time', None)),
        'duration_ms': duration,
        'total_tokens': int(getattr(run, 'total_tokens', 0) or 0),
        'prompt_tokens': int(getattr(run, 'prompt_tokens', 0) or 0),
        'completion_tokens': int(getattr(run, 'completion_tokens', 0) or 0),
        'total_cost': _safe_decimal_float(getattr(run, 'total_cost', None)),
        'child_count': len(getattr(run, 'child_run_ids', None) or []),
        'feedback_stats': getattr(run, 'feedback_stats', None) or {},
        'url': _public_run_url(client, run, project) if client is not None else (getattr(run, 'app_path', None) or None),
    }


def _langsmith_fallback_from_artifact(observability):
    output, path = _load_latest_coordinator_output()
    trace = list((output or {}).get('coordinator_trace') or []) if output else []
    duration = float((output or {}).get('pipeline_duration_seconds') or 0) if output else 0
    return {
        'success': True,
        'enabled': bool(observability.get('enabled')),
        'configured': bool(observability.get('api_key_configured')),
        'project': observability.get('project') or '',
        'endpoint': observability.get('endpoint') or '',
        'source': 'local_artifact',
        'message': 'LangSmith traces were unavailable; showing the latest local run artifact.',
        'summary': {
            'trace_count': 1 if output else 0,
            'run_count': len(trace),
            'error_count': len((output or {}).get('agent_errors') or []) if output else 0,
            'avg_duration_ms': round(duration * 1000) if duration else None,
            'total_tokens': 0,
            'total_cost': None,
            'slowest_ms': round(duration * 1000) if duration else None,
        },
        'type_breakdown': [{'name': 'local step', 'value': len(trace)}] if trace else [],
        'timeline': [
            {
                'id': f'local-{index}',
                'trace_id': f'local-{index}',
                'name': entry.split(']')[0].strip('[') if isinstance(entry, str) and ']' in entry else f'Step {index + 1}',
                'type': 'local step',
                'status': 'success',
                'error': '',
                'start_time': None,
                'end_time': None,
                'duration_ms': None,
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_cost': None,
                'child_count': 0,
                'feedback_stats': {},
                'url': None,
                'summary': str(entry),
                'children': [],
            }
            for index, entry in enumerate(trace[:20])
        ],
        'project_url': langsmith_project_web_url(observability),
    }


def langsmith_project_web_url(observability):
    endpoint = str(observability.get('endpoint') or 'https://smith.langchain.com').rstrip('/')
    project = observability.get('project') or ''
    base = endpoint.replace('api.smith.langchain.com', 'smith.langchain.com')
    if not project:
        return 'https://smith.langchain.com/'
    return f"{base}/o/default/projects/p/{project}"


def _set_coordinator_task(task_id, **updates):
    with coordinator_task_lock:
        task = coordinator_tasks.setdefault(task_id, {})
        task.update(updates)
        task['updated_at'] = datetime.utcnow().isoformat() + 'Z'
        return task.copy()

COORDINATOR_NODE_PROGRESS = {
    'query_agent': (32, 'collect', 'Search'),
    'media_agent': (36, 'collect', 'Search'),
    'data_bridge': (46, 'collect', 'Trust'),
    'divergence_compute': (54, 'map', 'Divergence'),
    'perspective_gen': (60, 'reason', 'Debate'),
    'deliberation': (68, 'reason', 'Consensus'),
    'targeted_search': (72, 'collect', 'Search'),
    'echo_chamber': (76, 'verify', 'Bias'),
    'fact_opinion': (82, 'verify', 'Facts'),
    'platform_interpret': (88, 'map', 'Sentiment'),
    'synthesis': (94, 'write', 'Outline'),
    'report_agent': (98, 'write', 'Draft'),
}


def _pct_dict_text(values):
    if not isinstance(values, dict) or not values:
        return ''
    parts = []
    for key, value in sorted(values.items(), key=lambda item: -float(item[1] or 0))[:4]:
        try:
            parts.append(f"{key}: {float(value):.0%}")
        except Exception:
            parts.append(f"{key}: {value}")
    return ', '.join(parts)


def _coordinator_progress_detail(node_name, update, state):
    if node_name == 'query_agent':
        run = update.get('query_run') or {}
        output = run.get('output') or {}
        total = output.get('total_sources_kept') or output.get('total_sources') or len(output.get('sources') or [])
        stance = _pct_dict_text(output.get('stance_distribution') or {})
        top_sources = sorted(output.get('sources') or [], key=lambda item: item.get('trust_score', 0), reverse=True)[:3]
        titles = [str(item.get('title') or item.get('url') or 'source')[:90] for item in top_sources]
        message = f"Collected {total} sources" + (f"; stance mix {stance}" if stance else '')
        return {'message': message, 'evidence': titles}
    if node_name == 'media_agent':
        run = update.get('media_run') or {}
        text = run.get('text_output') or ''
        if text:
            return {'message': f"Media evidence package captured ({len(text)} chars)", 'evidence': [text[:140]]}
        return {'message': 'Media engine skipped or unavailable', 'evidence': []}
    if node_name == 'data_bridge':
        props = update.get('bridged_propositions') or []
        sample = [str(item.get('content') or '')[:120] for item in props[:3]]
        return {'message': f"Bridged {len(props)} evidence propositions", 'evidence': sample}
    if node_name == 'divergence_compute':
        matrix = update.get('divergence_matrix') or {}
        hotspots = update.get('divergence_hotspots') or []
        if matrix:
            max_pair, max_value = max(matrix.items(), key=lambda item: item[1])
            return {'message': f"Computed {len(matrix)} divergence pairs; max {max_pair} = {float(max_value):.2f}", 'evidence': hotspots[:3]}
        return {'message': 'No cross-source divergence pairs available', 'evidence': []}
    if node_name == 'perspective_gen':
        perspectives = update.get('perspectives') or []
        return {'message': f"Selected {len(perspectives)} review perspectives", 'evidence': perspectives[:4]}
    if node_name == 'deliberation':
        consensus = update.get('deliberation_consensus') or []
        dissents = update.get('deliberation_dissents') or []
        return {'message': f"Deliberation produced {len(consensus)} consensus points and {len(dissents)} open dissents", 'evidence': (consensus[:2] + dissents[:2])}
    if node_name == 'echo_chamber':
        warnings = update.get('echo_warnings') or []
        return {'message': f"Bias scan found {len(warnings)} watch item(s)", 'evidence': warnings[:3]}
    if node_name == 'fact_opinion':
        facts = update.get('verified_facts') or []
        opinions = update.get('opinions_sentiments') or []
        frameworks = update.get('analytical_frameworks') or []
        evidence = [str(item.get('fact') or '')[:140] for item in facts[:3] if isinstance(item, dict)]
        return {'message': f"Separated {len(facts)} facts, {len(opinions)} opinions, {len(frameworks)} frameworks", 'evidence': evidence}
    if node_name == 'platform_interpret':
        interps = update.get('platform_interpretations') or {}
        return {'message': f"Generated {len(interps)} platform reading(s)", 'evidence': [f"{k}: {str(v)[:120]}" for k, v in list(interps.items())[:3]]}
    if node_name == 'synthesis':
        context = update.get('synthesis_context') or {}
        insights = context.get('top_insights') or []
        confidence = update.get('synthesis_confidence', 0)
        evidence = [str(item.get('insight') or '')[:140] for item in insights[:3] if isinstance(item, dict)]
        return {'message': f"Synthesized {len(insights)} insights at {float(confidence or 0):.0%} confidence", 'evidence': evidence}
    if node_name == 'report_agent':
        report = update.get('report_output') or ''
        return {'message': f"Coordinator report draft prepared ({len(report)} chars)", 'evidence': []}
    return {'message': f"{node_name.replace('_', ' ').title()} completed", 'evidence': []}


def _update_coordinator_progress_from_node(task_id, node_name, update, state, elapsed):
    progress, stage, micro_stage = COORDINATOR_NODE_PROGRESS.get(node_name, (50, 'collect', 'Rank'))
    detail = _coordinator_progress_detail(node_name, update, state)
    entry = {
        'node': node_name,
        'stage': stage,
        'micro_stage': micro_stage,
        'message': detail.get('message', ''),
        'evidence': detail.get('evidence', []),
        'elapsed_seconds': round(elapsed, 2),
        'created_at': datetime.utcnow().isoformat() + 'Z',
    }
    with coordinator_task_lock:
        task = coordinator_tasks.setdefault(task_id, {})
        timeline = list(task.get('timeline') or [])
        timeline.append(entry)
        task.update({
            'status': 'running',
            'progress': max(progress, int(task.get('progress') or 0)),
            'stage': stage,
            'micro_stage': micro_stage,
            'message': detail.get('message', ''),
            'details': detail,
            'timeline': timeline[-30:],
            'updated_at': datetime.utcnow().isoformat() + 'Z',
        })
        return task.copy()


def _run_coordinator_task(task_id, query, feedback=''):
    _set_coordinator_task(
        task_id,
        status='running',
        progress=6,
        stage='brief',
        micro_stage='Intent',
        message='Brief received',
        details={'message': 'Brief received', 'evidence': []},
        timeline=[]
    )
    started = time.time()
    try:
        _apply_observability_env()
        _set_coordinator_task(task_id, progress=12, stage='brief', micro_stage='Context', message='Tracing and runtime configured')
        run_query = query
        if feedback:
            run_query = f"{query}\n\nOperator refinement request:\n{feedback}"
            _set_coordinator_task(task_id, progress=16, stage='brief', micro_stage='Scope', message='Revision request attached')
        _set_coordinator_task(task_id, progress=22, stage='brief', micro_stage='Scope', message='Compiling analysis graph')
        from AgentCoordinator.coordinator import AgentCoordinator
        coordinator = AgentCoordinator(use_checkpointing=True)
        _set_coordinator_task(task_id, progress=28, stage='collect', micro_stage='Search', message='Starting evidence collection')

        def progress_callback(node_name, update, state, elapsed):
            _update_coordinator_progress_from_node(task_id, node_name, update, state, elapsed)

        result = coordinator.run_sync(run_query, progress_callback=progress_callback)
        _set_coordinator_task(task_id, progress=99, stage='write', micro_stage='Export', message='Coordinator artifact written')
        _set_coordinator_task(
            task_id,
            status='completed',
            progress=100,
            stage='write',
            micro_stage='Export',
            message='Coordinator pipeline completed',
            duration_seconds=round(time.time() - started, 2),
            coordinator_output_path=result.get('coordinator_output_path'),
            thread_id=result.get('thread_id'),
            synthesis_confidence=result.get('synthesis_confidence'),
        )
    except Exception as exc:
        logger.exception(f"Coordinator task failed: {task_id}")
        _set_coordinator_task(
            task_id,
            status='error',
            progress=100,
            message='Coordinator pipeline failed',
            error=str(exc),
            duration_seconds=round(time.time() - started, 2),
        )



@app.route('/api/observability/langsmith', methods=['GET'])
def get_langsmith_observability():
    """Return a browser-safe summary of recent LangSmith traces for Monitor."""
    observability = _apply_observability_env()
    if not observability.get('api_key_configured'):
        return jsonify({
            'success': True,
            'enabled': bool(observability.get('enabled')),
            'configured': False,
            'project': observability.get('project') or '',
            'endpoint': observability.get('endpoint') or '',
            'source': 'not_configured',
            'message': 'Add a LangSmith API key to see remote traces here.',
            'summary': {
                'trace_count': 0,
                'run_count': 0,
                'error_count': 0,
                'avg_duration_ms': None,
                'total_tokens': 0,
                'total_cost': None,
                'slowest_ms': None,
            },
            'type_breakdown': [],
            'timeline': [],
            'project_url': langsmith_project_web_url(observability),
        })

    try:
        from langsmith import Client
        project = observability.get('project') or 'public-opinion-analysis'
        endpoint = observability.get('endpoint') or 'https://api.smith.langchain.com'
        client = Client(
            api_url=endpoint,
            api_key=os.environ.get('LANGSMITH_API_KEY'),
            timeout_ms=7000,
        )
        since = datetime.utcnow() - timedelta(days=14)
        root_runs = list(client.list_runs(
            project_name=project,
            is_root=True,
            start_time=since,
            limit=8,
            select=[
                'id', 'name', 'run_type', 'start_time', 'end_time', 'error', 'status',
                'trace_id', 'child_run_ids', 'total_tokens', 'prompt_tokens',
                'completion_tokens', 'total_cost', 'feedback_stats', 'app_path'
            ],
        ))

        timeline = []
        all_durations = []
        type_counts = {}
        error_count = 0
        total_tokens = 0
        total_cost = 0.0
        cost_available = False
        run_count = 0

        for run in root_runs:
            try:
                full_run = client.read_run(getattr(run, 'id'), load_child_runs=True)
            except Exception:
                full_run = run
            root_summary = _summarize_langsmith_run(full_run, client, project)
            children = getattr(full_run, 'child_runs', None) or []
            child_summaries = [_summarize_langsmith_run(child, client, project) for child in children[:40]]
            child_summaries.sort(key=lambda item: item.get('start_time') or '')
            root_summary['children'] = child_summaries
            root_summary['summary'] = f"{len(child_summaries)} child step(s), {root_summary.get('duration_ms') or 0} ms"
            timeline.append(root_summary)

            for item in [root_summary] + child_summaries:
                run_count += 1
                type_counts[item['type']] = type_counts.get(item['type'], 0) + 1
                if item.get('error'):
                    error_count += 1
                if item.get('duration_ms') is not None:
                    all_durations.append(item['duration_ms'])
                total_tokens += int(item.get('total_tokens') or 0)
                if item.get('total_cost') is not None:
                    cost_available = True
                    total_cost += float(item.get('total_cost') or 0)

        avg_duration = round(sum(all_durations) / len(all_durations)) if all_durations else None
        return jsonify({
            'success': True,
            'enabled': bool(observability.get('enabled')),
            'configured': True,
            'project': project,
            'endpoint': endpoint,
            'source': 'langsmith',
            'message': 'Recent LangSmith traces loaded.',
            'summary': {
                'trace_count': len(root_runs),
                'run_count': run_count,
                'error_count': error_count,
                'avg_duration_ms': avg_duration,
                'total_tokens': total_tokens,
                'total_cost': round(total_cost, 6) if cost_available else None,
                'slowest_ms': max(all_durations) if all_durations else None,
            },
            'type_breakdown': [{'name': name, 'value': value} for name, value in sorted(type_counts.items(), key=lambda item: -item[1])],
            'timeline': timeline,
            'project_url': langsmith_project_web_url(observability),
        })
    except Exception as exc:
        logger.exception('Failed to load LangSmith traces')
        fallback = _langsmith_fallback_from_artifact(observability)
        fallback['error'] = str(exc)
        return jsonify(fallback)

@app.route('/api/coordinator/latest', methods=['GET'])
def get_latest_coordinator_output():
    """Return the newest structured AgentCoordinator artifact for the final frontend."""
    try:
        output, path = _load_latest_coordinator_output()
        feedback_records = _load_feedback_records(limit=20)
        observability = _apply_observability_env()
        archive_count = len(list(COORDINATOR_CACHE_DIR.glob('coordinator_output_*.json'))) if COORDINATOR_CACHE_DIR.exists() else 0
        if not output or not path:
            return jsonify({
                'success': False,
                'message': 'No coordinator output has been generated yet',
                'metadata': {
                    'cache_dir': str(COORDINATOR_CACHE_DIR),
                    'archive_count': archive_count,
                },
                'feedback': {
                    'records': feedback_records,
                    'summary': _feedback_summary(feedback_records),
                },
                'observability': observability,
            }), 404

        modified_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        return jsonify({
            'success': True,
            'output': output,
            'metadata': {
                'path': str(path),
                'modified_at': modified_at,
                'archive_count': archive_count,
                'schema_version': output.get('schema_version'),
            },
            'feedback': {
                'records': feedback_records,
                'summary': _feedback_summary(feedback_records),
            },
            'observability': observability,
        })
    except Exception as exc:
        logger.exception("Failed to load latest coordinator output")
        return jsonify({'success': False, 'message': f'Failed to load coordinator output: {exc}'}), 500


@app.route('/api/coordinator/run', methods=['POST'])
def run_coordinator_api():
    """Start a background AgentCoordinator run from the final frontend."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({'success': False, 'message': 'Request body must be a JSON object'}), 400

    query = str(payload.get('query') or '').strip()
    feedback = str(payload.get('feedback') or '').strip()
    if not query:
        try:
            latest, _ = _load_latest_coordinator_output()
            query = str((latest or {}).get('query') or '').strip()
        except Exception:
            query = ''
    if not query:
        return jsonify({'success': False, 'message': 'Analysis query is required'}), 400

    blocked = reject_if_sensitive({'query': query, 'feedback': feedback}, settings)
    if blocked:
        return jsonify(blocked), 400

    task_id = f"coord_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now = datetime.utcnow().isoformat() + 'Z'
    with coordinator_task_lock:
        coordinator_tasks[task_id] = {
            'task_id': task_id,
            'query': query,
            'has_feedback': bool(feedback),
            'status': 'queued',
            'progress': 0,
            'message': 'Coordinator task queued',
            'created_at': now,
            'updated_at': now,
        }

    thread = threading.Thread(target=_run_coordinator_task, args=(task_id, query, feedback), daemon=True)
    thread.start()
    return jsonify({'success': True, 'task': coordinator_tasks[task_id]})


@app.route('/api/coordinator/task/<task_id>', methods=['GET'])
def get_coordinator_task(task_id):
    with coordinator_task_lock:
        task = coordinator_tasks.get(task_id)
        if not task:
            return jsonify({'success': False, 'message': 'Coordinator task does not exist'}), 404
        return jsonify({'success': True, 'task': task})


@app.route('/api/coordinator/feedback', methods=['GET', 'POST'])
def coordinator_feedback_api():
    """Record operator feedback for traceability and optional follow-up refinement."""
    if request.method == 'GET':
        records = _load_feedback_records(limit=100)
        return jsonify({'success': True, 'records': records, 'summary': _feedback_summary(records)})

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({'success': False, 'message': 'Request body must be a JSON object'}), 400

    feedback_text = str(payload.get('feedback') or '').strip()
    if not feedback_text:
        return jsonify({'success': False, 'message': 'Feedback text is required'}), 400

    record = {
        'id': f"fb_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'query': str(payload.get('query') or '').strip(),
        'target': str(payload.get('target') or 'overall').strip() or 'overall',
        'action': str(payload.get('action') or 'review').strip() or 'review',
        'priority': str(payload.get('priority') or 'normal').strip() or 'normal',
        'feedback': feedback_text,
        'thread_id': str(payload.get('thread_id') or '').strip(),
        'source': 'final_frontend',
    }
    try:
        COORDINATOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(COORDINATOR_FEEDBACK_LOG, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        return jsonify({'success': True, 'record': record})
    except Exception as exc:
        logger.exception("Failed to record coordinator feedback")
        return jsonify({'success': False, 'message': f'Failed to record feedback: {exc}'}), 500

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get all application statuses"""
    check_app_status()
    return jsonify({
        app_name: {
            'status': info['status'],
            'port': info['port'],
            'output_lines': len(info['output'])
        }
        for app_name, info in processes.items()
    })

@app.route('/api/start/<app_name>')
def start_app(app_name):
    """Start specified application"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})

    if app_name == 'forum':
        try:
            start_forum_engine()
            processes['forum']['status'] = 'running'
            return jsonify({'success': True, 'message': 'ForumEngine started'})
        except Exception as exc:  # pragma: no cover
            logger.exception("Manual ForumEngine startup failed")
            return jsonify({'success': False, 'message': f'ForumEngine startup failed: {exc}'})

    script_path = STREAMLIT_SCRIPTS.get(app_name)
    if not script_path:
        return jsonify({'success': False, 'message': 'This application does not support startup operation'})

    success, message = start_streamlit_app(
        app_name,
        script_path,
        processes[app_name]['port']
    )

    if success:
        # Wait for application startup
        startup_success, startup_message = wait_for_app_startup(app_name, 15)
        if not startup_success:
            message += f" but startup check failed: {startup_message}"
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop/<app_name>')
def stop_app(app_name):
    """Stop specified application"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})

    if app_name == 'forum':
        try:
            stop_forum_engine()
            processes['forum']['status'] = 'stopped'
            return jsonify({'success': True, 'message': 'ForumEngine stopped'})
        except Exception as exc:  # pragma: no cover
            logger.exception("Manual ForumEngine stop failed")
            return jsonify({'success': False, 'message': f'ForumEngine stop failed: {exc}'})

    success, message = stop_streamlit_app(app_name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/output/<app_name>')
def get_output(app_name):
    """Get application output"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})
    
    # Special handling for Agent Coordinator (forum log)
    if app_name == 'forum':
        try:
            forum_log_content = read_log_from_file('forum')
            return jsonify({
                'success': True,
                'output': forum_log_content,
                'total_lines': len(forum_log_content)
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Failed to read forum log: {str(e)}'})
    
    # Read full log from file
    output_lines = read_log_from_file(app_name)
    
    return jsonify({
        'success': True,
        'output': output_lines
    })

@app.route('/api/test_log/<app_name>')
def test_log(app_name):
    """Test log writing functionality"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})
    
    # Write test message
    test_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Test log message - {datetime.now()}"
    write_log_to_file(app_name, test_msg)
    
    # Send via Socket.IO
    socketio.emit('console_output', {
        'app': app_name,
        'line': test_msg
    })
    
    return jsonify({
        'success': True,
        'message': f'Test message written to {app_name} log'
    })

@app.route('/api/forum/start')
def start_forum_monitoring_api():
    """Manually start ForumEngine forum"""
    try:
        from ForumEngine.monitor import start_forum_monitoring
        success = start_forum_monitoring()
        if success:
            return jsonify({'success': True, 'message': 'ForumEngine forum started'})
        else:
            return jsonify({'success': False, 'message': 'ForumEngine forum startup failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to start forum: {str(e)}'})

@app.route('/api/forum/stop')
def stop_forum_monitoring_api():
    """Manually stop ForumEngine forum"""
    try:
        from ForumEngine.monitor import stop_forum_monitoring
        stop_forum_monitoring()
        return jsonify({'success': True, 'message': 'ForumEngine forum stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to stop forum: {str(e)}'})

@app.route('/api/forum/log')
def get_forum_log():
    """Get ForumEngine's forum.log content"""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        if not forum_log_file.exists():
            return jsonify({
                'success': True,
                'log_lines': [],
                'parsed_messages': [],
                'total_lines': 0
            })
        
        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
        
        # Parse each log line and extract conversation information
        parsed_messages = []
        for line in lines:
            parsed_message = parse_forum_log_line(line)
            if parsed_message:
                parsed_messages.append(parsed_message)
        
        return jsonify({
            'success': True,
            'log_lines': lines,
            'parsed_messages': parsed_messages,
            'total_lines': len(lines)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to read forum.log: {str(e)}'})

@app.route('/api/forum/log/history', methods=['POST'])
def get_forum_log_history():
    """Get Forum history logs (supports starting from specified position)"""
    try:
        data = request.get_json()
        start_position = data.get('position', 0)  # Client's last received position
        max_lines = data.get('max_lines', 1000)   # Maximum lines to return

        forum_log_file = LOG_DIR / "forum.log"
        if not forum_log_file.exists():
            return jsonify({
                'success': True,
                'log_lines': [],
                'position': 0,
                'has_more': False
            })

        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Read from specified position
            f.seek(start_position)
            lines = []
            line_count = 0

            for line in f:
                if line_count >= max_lines:
                    break
                line = line.rstrip('\n\r')
                if line.strip():
                    # Add timestamp
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    formatted_line = f"[{timestamp}] {line}"
                    lines.append(formatted_line)
                    line_count += 1

            # Record current position
            current_position = f.tell()

            # Check if there is more content
            f.seek(0, 2)  # Move to end of file
            end_position = f.tell()
            has_more = current_position < end_position

        return jsonify({
            'success': True,
            'log_lines': lines,
            'position': current_position,
            'has_more': has_more
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to read forum history: {str(e)}'})

@app.route('/api/search', methods=['POST'])
def search():
    """Unified search interface"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'success': False, 'message': 'Search query cannot be empty'})
    
    blocked = reject_if_sensitive({'query': query}, settings)
    if blocked:
        return jsonify(blocked), 400
    
    # ForumEngine forum is already running in background, will automatically detect search activity
    # logger.info("ForumEngine: Search request received, forum will automatically detect log changes")
    
    # Check which applications are running
    check_app_status()
    running_apps = [name for name, info in processes.items() if info['status'] == 'running']
    
    if not running_apps:
        return jsonify({'success': False, 'message': 'No running applications'})
    
    # Send search request to running applications
    results = {}
    api_ports = {'media': 8502, 'query': 8503}
    
    for app_name in running_apps:
        try:
            api_port = api_ports[app_name]
            # Call Streamlit application's API endpoint
            response = requests.post(
                f"http://localhost:{api_port}/api/search",
                json={'query': query},
                timeout=10
            )
            if response.status_code == 200:
                results[app_name] = response.json()
            else:
                results[app_name] = {'success': False, 'message': 'API call failed'}
        except Exception as e:
            results[app_name] = {'success': False, 'message': str(e)}
    
    # After search, can choose to stop monitoring or let it continue to capture subsequent processing logs
    # Here we let monitoring continue, users can manually stop via other interfaces
    
    return jsonify({
        'success': True,
        'query': query,
        'results': results
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Expose selected configuration values to the frontend."""
    try:
        config_values = read_config_values()
        return jsonify({'success': True, 'config': config_values})
    except Exception as exc:
        logger.exception("Failed to read config")
        return jsonify({'success': False, 'message': f'Failed to read config: {exc}'}), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration values and persist them to config.py."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict) or not payload:
        return jsonify({'success': False, 'message': 'Request body cannot be empty'}), 400

    updates = {}
    for key, value in payload.items():
        if key in CONFIG_KEYS:
            updates[key] = value if value is not None else ''

    if not updates:
        return jsonify({'success': False, 'message': 'No configuration items to update'}), 400

    try:
        write_config_values(updates)
        updated_config = read_config_values()
        return jsonify({'success': True, 'config': updated_config})
    except Exception as exc:
        logger.exception("Failed to update config")
        return jsonify({'success': False, 'message': f'Failed to update config: {exc}'}), 500


@app.route('/api/system/status')
def get_system_status():
    """Return final demo runtime status."""
    state = _get_system_state()
    return jsonify({
        'success': True,
        'started': state['started'],
        'starting': state['starting'],
        'mode': 'final_react_demo',
        'streamlit_required': False
    })


@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the complete system after receiving request."""
    allowed, message = _prepare_system_start()
    if not allowed:
        return jsonify({'success': False, 'message': message}), 400

    try:
        success, logs, errors = initialize_system_components()
        if success:
            _set_system_state(started=True)
            return jsonify({'success': True, 'message': 'System startup successful', 'logs': logs})

        _set_system_state(started=False)
        return jsonify({
            'success': False,
            'message': 'System startup failed',
            'logs': logs,
            'errors': errors
        }), 500
    except Exception as exc:  # pragma: no cover - fallback catch
        logger.exception("Exception occurred during system startup")
        _set_system_state(started=False)
        return jsonify({'success': False, 'message': f'System startup exception: {exc}'}), 500
    finally:
        _set_system_state(starting=False)

@app.route('/api/system/shutdown', methods=['POST'])
def shutdown_system():
    """Gracefully stop all components and shut down the current service process."""
    state = _get_system_state()
    if state['starting']:
        return jsonify({'success': False, 'message': 'System is starting/restarting, please wait'}), 400

    target_ports = [
        f"{name}:{info['port']}"
        for name, info in processes.items()
        if info.get('port')
    ]

    # When shutdown request is already in progress, return currently alive child processes for frontend progress tracking
    if not _mark_shutdown_requested():
        running = _describe_running_children()
        detail = 'Shutdown command issued, please wait...'
        if running:
            detail = f"Shutdown command issued, waiting for processes to exit: {', '.join(running)}"
        if target_ports:
            detail = f"{detail} (ports: {', '.join(target_ports)})"
        return jsonify({'success': True, 'message': detail, 'ports': target_ports})

    running = _describe_running_children()
    if running:
        _log_shutdown_step("Starting system shutdown, waiting for child processes to exit: " + ", ".join(running))
    else:
        _log_shutdown_step("Starting system shutdown, no alive child processes detected")

    try:
        _set_system_state(started=False, starting=False)
        _start_async_shutdown(cleanup_timeout=6.0)
        message = 'Shutdown command issued, stopping processes'
        if running:
            message = f"{message}: {', '.join(running)}"
        if target_ports:
            message = f"{message} (ports: {', '.join(target_ports)})"
        return jsonify({'success': True, 'message': message, 'ports': target_ports})
    except Exception as exc:  # pragma: no cover - fallback catch
        logger.exception("Exception occurred during system shutdown")
        return jsonify({'success': False, 'message': f'System shutdown exception: {exc}'}), 500

@socketio.on('connect')
def handle_connect():
    """Client connection"""
    emit('status', 'Connected to Flask server')

@socketio.on('request_status')
def handle_status_request():
    """Request status update"""
    check_app_status()
    emit('status_update', {
        app_name: {
            'status': info['status'],
            'port': info['port']
        }
        for app_name, info in processes.items()
    })

if __name__ == '__main__':
    # Read HOST and PORT from config file
    from config import settings
    HOST = settings.HOST
    PORT = settings.PORT
    
    logger.info("Waiting for configuration confirmation, system will start components after frontend command...")
    logger.info(f"Flask server started, access at: http://{HOST}:{PORT}")
    
    try:
        socketio.run(app, host=HOST, port=PORT, debug=False)
    except KeyboardInterrupt:
        logger.info("\nClosing application...")
        cleanup_processes()
        
    
