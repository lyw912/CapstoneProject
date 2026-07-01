"""
Log Monitor - Real-time monitoring of SummaryNode outputs from Media/Query log files
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import re
import json
from typing import Dict, Optional, List
from threading import Lock
from loguru import logger

# Import forum host module
try:
    from .llm_host import generate_host_speech
    HOST_AVAILABLE = True
except ImportError:
    logger.exception("ForumEngine: Forum host module not found, running in pure monitor mode")
    HOST_AVAILABLE = False

class LogMonitor:
    """Intelligent log monitor based on file changes"""
   
    def __init__(self, log_dir: str = "logs"):
        """Initialize log monitor"""
        self.log_dir = Path(log_dir)
        self.forum_log_file = self.log_dir / "forum.log"
       
        # Log files to monitor
        self.monitored_logs = {
            'media': self.log_dir / 'media.log',
            'query': self.log_dir / 'query.log'
        }
       
        # Monitor status
        self.is_monitoring = False
        self.monitor_thread = None
        self.file_positions = {}  # Track read position for each file
        self.file_line_counts = {}  # Track line count for each file
        self.is_searching = False  # Whether searching is active
        self.search_inactive_count = 0  # Search inactivity counter
        self.write_lock = Lock()  # Write lock to prevent concurrent write conflicts
        
        # Host-related status
        self.agent_speeches_buffer = []  # Agent speech buffer
        self.host_speech_threshold = 5  # Trigger host speech every 5 agent speeches
        self.is_host_generating = False  # Whether host is generating speech
       
        # Target node identification patterns
        # 1. Class name (may be in old format)
        # 2. Full module path (actual log format, includes engine prefix)
        # 3. Partial module path (compatibility)
        # 4. Key identifier text
        self.target_node_patterns = [
            'FirstSummaryNode',  # Class name
            'ReflectionSummaryNode',  # Class name
            'MediaEngine.nodes.summary_node',  # MediaEngine full path
            'QueryEngine.nodes.summary_node',  # QueryEngine full path
            'nodes.summary_node',  # Module path (compatibility, for partial matching)
            'Generating first paragraph summary',  # FirstSummaryNode identifier
            'Generating reflection summary',  # ReflectionSummaryNode identifier
        ]
        
        # Multi-line content capture status
        self.capturing_json = {}  # JSON capture status for each app
        self.json_buffer = {}     # JSON buffer for each app
        self.json_start_line = {} # JSON start line for each app
        self.in_error_block = {}  # Whether each app is currently inside an ERROR block
       
        # Ensure logs directory exists
        self.log_dir.mkdir(exist_ok=True)
   
    def clear_forum_log(self):
        """Clear forum.log file"""
        try:
            if self.forum_log_file.exists():
                self.forum_log_file.unlink()
           
            # Create new forum.log file and write start marker
            start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Use write_to_forum_log to write start marker for consistent formatting
            with open(self.forum_log_file, 'w', encoding='utf-8') as f:
                pass  # Create empty file first
            self.write_to_forum_log(f"=== ForumEngine Monitor Started - {start_time} ===", "SYSTEM")
               
            logger.info(f"ForumEngine: forum.log cleared and initialized")
            
            # Reset JSON capture status
            self.capturing_json = {}
            self.json_buffer = {}
            self.json_start_line = {}
            self.in_error_block = {}
            
            # Reset host-related status
            self.agent_speeches_buffer = []
            self.is_host_generating = False
           
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to clear forum.log: {e}")
   
    def write_to_forum_log(self, content: str, source: str = None):
        """Write content to forum.log (thread-safe)"""
        try:
            with self.write_lock:  # Use lock to ensure thread safety
                with open(self.forum_log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    # Convert actual newlines to \\n strings to ensure entire record is on one line
                    content_one_line = content.replace('\n', '\\n').replace('\r', '\\r')
                    # Add source label after timestamp if provided
                    if source:
                        f.write(f"[{timestamp}] [{source}] {content_one_line}\n")
                    else:
                        f.write(f"[{timestamp}] {content_one_line}\n")
                    f.flush()
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to write to forum.log: {e}")
    
    def get_log_level(self, line: str) -> Optional[str]:
        """Detect the log level of a log line (INFO/ERROR/WARNING/DEBUG, etc.)

        Supports loguru format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...

        Returns:
            'INFO', 'ERROR', 'WARNING', 'DEBUG', or None (if unrecognized)
        """
        # Check loguru format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...
        # Match pattern: | LEVEL | or | LEVEL     |
        match = re.search(r'\|\s*(INFO|ERROR|WARNING|DEBUG|TRACE|CRITICAL)\s*\|', line)
        if match:
            return match.group(1)
        return None
    
    def is_target_log_line(self, line: str) -> bool:
        """Check if the log line is a target log line (SummaryNode)

        Supports multiple identification methods:
        1. Class name: FirstSummaryNode, ReflectionSummaryNode
        2. Full module path: MediaEngine.nodes.summary_node, QueryEngine.nodes.summary_node
        3. Partial module path: nodes.summary_node (compatibility)
        4. Key identifier text: Generating first paragraph summary, Generating reflection summary

        Exclusion conditions:
        - ERROR level logs (error logs should not be recognized as target nodes)
        - Logs containing error keywords (JSON parse failure, JSON fix failure, etc.)
        """
        # Exclude ERROR level logs
        log_level = self.get_log_level(line)
        if log_level == 'ERROR':
            return False

        # Backward compatibility check
        if "| ERROR" in line or "| ERROR    |" in line:
            return False

        # Exclude logs containing error keywords
        error_keywords = ["JSON parsing failed", "JSON repair failed", "Traceback", "File \""]
        for keyword in error_keywords:
            if keyword in line:
                return False

        # Check if the line contains any target node pattern
        for pattern in self.target_node_patterns:
            if pattern in line:
                return True
        return False
    
    def is_valuable_content(self, line: str) -> bool:
        """Determine if the content is valuable (excluding short prompts and error messages)"""
        # If the line contains "Cleaned output", it is considered valuable
        if self._cleaned_output_marker(line):
            return True

        # Exclude common short prompt messages and error messages
        exclude_patterns = [
            "JSON parsing failed",
            "JSON repair failed",
            "Using cleaned text directly",
            "JSON parsing successful",
            "Successfully generated",
            "Paragraph updated",
            "Generating",
            "Starting processing",
            "Processing completed",
            "HOST speech read",
            "Failed to read HOST speech",
            "HOST speech not found",
            "Debug output",
            "Info log"
        ]

        for pattern in exclude_patterns:
            if pattern in line:
                return False

        # If the line is too short, it is also not considered valuable
        # Remove timestamps: supports both old and new formats
        clean_line = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', line)
        clean_line = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', clean_line)
        clean_line = clean_line.strip()
        if len(clean_line) < 30:  # Threshold can be adjusted
            return False

        return True
    
    def is_json_start_line(self, line: str) -> bool:
        """Determine if this is a JSON start line"""
        marker = self._cleaned_output_marker(line)
        if not marker:
            return False
        return "{" in line[line.find(marker) + len(marker):].lstrip()

    def _cleaned_output_marker(self, line: str) -> Optional[str]:
        """Return the cleaned-output marker used by English or Chinese logs."""
        for marker in ("Cleaned output:", "清理后的输出:"):
            if marker in line:
                return marker
        return None

    def _strip_log_prefix(self, line: str) -> str:
        """Remove old timestamp and loguru prefixes from a log line."""
        clean_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line.strip())
        clean_line = re.sub(
            r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*.*?\s*-\s*',
            '',
            clean_line,
        )
        return clean_line.strip()
    
    def is_json_end_line(self, line: str) -> bool:
        """Determine if this is a JSON end line

        Only checks for pure end marker lines without any log format information (timestamps, etc.).
        If the line contains a timestamp, it should be cleaned before checking, but here we return False
        to indicate that further processing is needed.
        """
        stripped = line.strip()

        # If the line contains a timestamp (old or new format), it's not a pure end line
        # Old format: [HH:MM:SS]
        if re.match(r'^\[\d{2}:\d{2}:\d{2}\]', stripped):
            return False
        # New format: YYYY-MM-DD HH:mm:ss.SSS
        if re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}', stripped):
            return False

        # For lines without timestamps, check if they are pure end markers
        if stripped == "}" or stripped == "] }":
            return True
        return False
    
    def extract_json_content(self, json_lines: List[str]) -> Optional[str]:
        """Extract and parse JSON content from multiple lines"""
        try:
            # Find the starting position of JSON
            json_start_idx = -1
            marker = None
            for i, line in enumerate(json_lines):
                marker = self._cleaned_output_marker(line)
                if marker and "{" in line[line.find(marker) + len(marker):]:
                    json_start_idx = i
                    break

            if json_start_idx == -1 or marker is None:
                return None

            # Extract the JSON part
            first_line = json_lines[json_start_idx]
            json_start_pos = first_line.find(marker)
            if json_start_pos == -1:
                return None

            json_part = first_line[json_start_pos + len(marker):].strip()

            # If the first line contains a complete JSON, process it directly
            if json_part.strip().endswith("}") and json_part.count("{") == json_part.count("}"):
                try:
                    json_obj = json.loads(json_part.strip())
                    return self.format_json_content(json_obj)
                except json.JSONDecodeError:
                    # Single-line JSON parsing failed, try to fix
                    fixed_json = self.fix_json_string(json_part.strip())
                    if fixed_json:
                        try:
                            json_obj = json.loads(fixed_json)
                            return self.format_json_content(json_obj)
                        except json.JSONDecodeError:
                            pass
                    return None

            # Process multi-line JSON
            json_text = json_part
            for line in json_lines[json_start_idx + 1:]:
                clean_line = self._strip_log_prefix(line)
                json_text += clean_line

            # Try to parse JSON
            try:
                json_obj = json.loads(json_text.strip())
                return self.format_json_content(json_obj)
            except json.JSONDecodeError:
                # Multi-line JSON parsing failed, try to fix
                fixed_json = self.fix_json_string(json_text.strip())
                if fixed_json:
                    try:
                        json_obj = json.loads(fixed_json)
                        return self.format_json_content(json_obj)
                    except json.JSONDecodeError:
                        pass
                return None

        except Exception as e:
            # Other exceptions also return None without printing error messages
            return None

            # If the first line contains a complete JSON, process it directly
            if json_part.strip().endswith("}") and json_part.count("{") == json_part.count("}"):
                try:
                    json_obj = json.loads(json_part.strip())
                    return self.format_json_content(json_obj)
                except json.JSONDecodeError:
                    # Single-line JSON parsing failed, try to fix
                    fixed_json = self.fix_json_string(json_part.strip())
                    if fixed_json:
                        try:
                            json_obj = json.loads(fixed_json)
                            return self.format_json_content(json_obj)
                        except json.JSONDecodeError:
                            pass
                    return None

            # Process multi-line JSON
            json_text = json_part
            for line in json_lines[json_start_idx + 1:]:
                # Remove timestamps: supports old format [HH:MM:SS] and new format loguru (YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...)
                # Old format: [HH:MM:SS]
                clean_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)
                # New format: Remove loguru format timestamp and level information
                # Format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
                clean_line = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', clean_line)
                json_text += clean_line

            # Try to parse JSON
            try:
                json_obj = json.loads(json_text.strip())
                return self.format_json_content(json_obj)
            except json.JSONDecodeError:
                # Multi-line JSON parsing failed, try to fix
                fixed_json = self.fix_json_string(json_text.strip())
                if fixed_json:
                    try:
                        json_obj = json.loads(fixed_json)
                        return self.format_json_content(json_obj)
                    except json.JSONDecodeError:
                        pass
                return None

        except Exception as e:
            # Other exceptions also return None without printing error messages
            return None
    
    def format_json_content(self, json_obj: dict) -> str:
        """Format JSON content into a readable form"""
        try:
            # Extract main content, prioritize reflection summary, then first summary
            content = None

            if "updated_paragraph_latest_state" in json_obj:
                content = json_obj["updated_paragraph_latest_state"]
            elif "paragraph_latest_state" in json_obj:
                content = json_obj["paragraph_latest_state"]

            # If content is found, return it directly (keeping newlines as \n)
            if content:
                return content

            # If expected fields are not found, return the entire JSON as a string
            return f"Cleaned output: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"

        except Exception as e:
            logger.exception(f"ForumEngine: Error formatting JSON: {e}")
            return f"Cleaned output: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"

    def extract_node_content(self, line: str) -> Optional[str]:
        """Extract node content, removing timestamps, node names, and other prefixes"""
        content = line

        # Remove timestamp part: supports both old and new formats
        # Old format: [HH:MM:SS]
        match_old = re.search(r'\[\d{2}:\d{2}:\d{2}\]\s*(.+)', content)
        if match_old:
            content = match_old.group(1).strip()
        else:
            # New format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
            match_new = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*(.+)', content)
            if match_new:
                content = match_new.group(1).strip()

        if not content:
            return line.strip()

        # Remove all square bracket tags (including node names and app names)
        content = re.sub(r'^\[.*?\]\s*', '', content)

        # Continue removing possible multiple consecutive tags
        while re.match(r'^\[.*?\]\s*', content):
            content = re.sub(r'^\[.*?\]\s*', '', content)

        # Remove common prefixes (e.g., "First summary: ", "Reflection summary: ", etc.)
        prefixes_to_remove = [
            "First summary: ",
            "Reflection summary: ",
            "Cleaned output: ",
            "清理后的输出: ",
        ]

        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):]
                break

        # Remove possible app name labels (not inside square brackets)
        app_names = ['MEDIA', 'QUERY']
        for app_name in app_names:
            # Remove standalone APP_NAME (at the beginning of the line)
            content = re.sub(rf'^{app_name}\s+', '', content, flags=re.IGNORECASE)

        # Clean up extra spaces
        content = re.sub(r'\s+', ' ', content)

        return content.strip()
   
    def get_file_size(self, file_path: Path) -> int:
        """Get file size"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except:
            return 0

    def get_file_line_count(self, file_path: Path) -> int:
        """Get file line count"""
        try:
            if not file_path.exists():
                return 0
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
   
    def read_new_lines(self, file_path: Path, app_name: str) -> List[str]:
        """Read new lines from the file"""
        new_lines = []

        try:
            if not file_path.exists():
                return new_lines

            current_size = self.get_file_size(file_path)
            last_position = self.file_positions.get(app_name, 0)

            # If the file got smaller, it was cleared, restart from the beginning
            if current_size < last_position:
                last_position = 0
                # Reset JSON capture state
                self.capturing_json[app_name] = False
                self.json_buffer[app_name] = []
                self.in_error_block[app_name] = False

            if current_size > last_position:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    new_lines = new_content.split('\n')

                    # Update position
                    self.file_positions[app_name] = f.tell()

                    # Filter empty lines
                    new_lines = [line.strip() for line in new_lines if line.strip()]

        except Exception as e:
            logger.exception(f"ForumEngine: Failed to read {app_name} log: {e}")

        return new_lines
   
    def process_lines_for_json(self, lines: List[str], app_name: str) -> List[str]:
        """Process lines to capture multi-line JSON content

        Implements ERROR block filtering: if an ERROR level log is encountered,
        reject processing until the next INFO level log is encountered.
        """
        captured_contents = []

        # Initialize state
        if app_name not in self.capturing_json:
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
        if app_name not in self.in_error_block:
            self.in_error_block[app_name] = False

        for line in lines:
            if not line.strip():
                continue

            # First, check log level and update ERROR block state
            log_level = self.get_log_level(line)
            if log_level == 'ERROR':
                # Encountered ERROR, enter ERROR block state
                self.in_error_block[app_name] = True
                # If currently capturing JSON, stop immediately and clear buffer
                if self.capturing_json[app_name]:
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
                # Skip current line, do not process
                continue
            elif log_level == 'INFO':
                # Encountered INFO, exit ERROR block state
                self.in_error_block[app_name] = False
            # Other levels (WARNING, DEBUG, etc.) maintain current state

            # If in ERROR block, reject processing all content
            if self.in_error_block[app_name]:
                # If currently capturing JSON, stop immediately and clear buffer
                if self.capturing_json[app_name]:
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
                # Skip current line, do not process
                continue

            # Check if this is a target node line and JSON start marker
            is_target = self.is_target_log_line(line)
            is_json_start = self.is_json_start_line(line)

            # Only JSON output from target nodes (SummaryNode) should be captured
            # Filter out output from other nodes like SearchNode (they are not target nodes, even if they have JSON)
            if is_target and is_json_start:
                # Start capturing JSON (must be target node and contain "Cleaned output: {")
                self.capturing_json[app_name] = True
                self.json_buffer[app_name] = [line]
                self.json_start_line[app_name] = line

                # Check if this is a single-line JSON
                if line.strip().endswith("}"):
                    # Single-line JSON, process immediately
                    content = self.extract_json_content([line])
                    if content:  # Only successfully parsed content is recorded
                        # Remove duplicate tags and format
                        clean_content = self._clean_content_tags(content, app_name)
                        captured_contents.append(f"{clean_content}")
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []

            elif is_target and self.is_valuable_content(line):
                # Other valuable SummaryNode content (must be target node and valuable)
                clean_content = self._clean_content_tags(self.extract_node_content(line), app_name)
                captured_contents.append(f"{clean_content}")

            elif self.capturing_json[app_name]:
                # Subsequent lines while capturing JSON
                self.json_buffer[app_name].append(line)

                # Check if this is the end of JSON
                # First clean the timestamp, then determine if the cleaned line is an end marker
                cleaned_line = line.strip()
                cleaned_line = self._strip_log_prefix(cleaned_line)

                # After cleaning, determine if this is an end marker
                if cleaned_line == "}" or cleaned_line == "] }":
                    # JSON ended, process the complete JSON
                    content = self.extract_json_content(self.json_buffer[app_name])
                    if content:  # Only successfully parsed content is recorded
                        # Remove duplicate tags and format
                        clean_content = self._clean_content_tags(content, app_name)
                        captured_contents.append(f"{clean_content}")

                    # Reset state
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []

        return captured_contents
    
    def _trigger_host_speech(self):
        """Trigger host speech (synchronous execution)"""
        if not HOST_AVAILABLE or self.is_host_generating:
            return

        try:
            # Set generation flag
            self.is_host_generating = True

            # Get 5 speeches from buffer
            recent_speeches = self.agent_speeches_buffer[:5]
            if len(recent_speeches) < 5:
                self.is_host_generating = False
                return

            logger.info("ForumEngine: Generating host speech...")

            # Call host speech generation (pass recent 5 speeches)
            host_speech = generate_host_speech(recent_speeches)

            if host_speech:
                # Write host speech to forum.log
                self.write_to_forum_log(host_speech, "HOST")
                logger.info(f"ForumEngine: Host speech recorded")

                # Clear processed 5 speeches
                self.agent_speeches_buffer = self.agent_speeches_buffer[5:]
            else:
                logger.error("ForumEngine: Failed to generate host speech")

            # Reset generation flag
            self.is_host_generating = False

        except Exception as e:
            logger.exception(f"ForumEngine: Error triggering host speech: {e}")
            self.is_host_generating = False
    
    def _clean_content_tags(self, content: str, app_name: str) -> str:
        """Clean duplicate tags and redundant prefixes from content"""
        if not content:
            return content

        # First remove all possible tag formats (including [INSIGHT], [MEDIA], [QUERY], etc.)
        # Use stronger cleaning method
        all_app_names = ['MEDIA', 'QUERY']

        for name in all_app_names:
            # Remove [APP_NAME] format (case insensitive)
            content = re.sub(rf'\[{name}\]\s*', '', content, flags=re.IGNORECASE)
            # Remove standalone APP_NAME format
            content = re.sub(rf'^{name}\s+', '', content, flags=re.IGNORECASE)

        # Remove any other square bracket tags
        content = re.sub(r'^\[.*?\]\s*', '', content)

        # Remove possible duplicate spaces
        content = re.sub(r'\s+', ' ', content)

        return content.strip()
   
    def monitor_logs(self):
        """Intelligent log file monitoring"""
        logger.info("ForumEngine: Forum creating...")

        # Initialize file line counts and positions - record current state as baseline
        for app_name, log_file in self.monitored_logs.items():
            self.file_line_counts[app_name] = self.get_file_line_count(log_file)
            self.file_positions[app_name] = self.get_file_size(log_file)
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
            self.in_error_block[app_name] = False
            # logger.info(f"ForumEngine: {app_name} baseline line count: {self.file_line_counts[app_name]}")

        while self.is_monitoring:
            try:
                # Detect changes in both log files simultaneously
                any_growth = False
                any_shrink = False
                captured_any = False

                # Process each log file independently
                for app_name, log_file in self.monitored_logs.items():
                    current_lines = self.get_file_line_count(log_file)
                    previous_lines = self.file_line_counts.get(app_name, 0)

                    if current_lines > previous_lines:
                        any_growth = True
                        # Read new content immediately
                        new_lines = self.read_new_lines(log_file, app_name)

                        # First check if search needs to be triggered (only trigger once)
                        if not self.is_searching:
                            for line in new_lines:
                                # Check if it contains target node pattern (supports multiple formats)
                                if line.strip() and self.is_target_log_line(line):
                                    # Further confirm it's a first summary node (FirstSummaryNode or contains "Generating first paragraph summary")
                                    if 'FirstSummaryNode' in line or 'Generating first paragraph summary' in line:
                                        logger.info(f"ForumEngine: First forum content detected in {app_name}")
                                        self.is_searching = True
                                        self.search_inactive_count = 0
                                        # Clear forum.log to start a new session
                                        self.clear_forum_log()
                                        break  # One is enough, break the loop

                        # Process all new content (if in searching state)
                        if self.is_searching:
                            # Use new processing logic
                            captured_contents = self.process_lines_for_json(new_lines, app_name)

                            for content in captured_contents:
                                # Convert app_name to uppercase as tag (e.g., media -> MEDIA)
                                source_tag = app_name.upper()
                                self.write_to_forum_log(content, source_tag)
                                # logger.info(f"ForumEngine: Captured - {content}")
                                captured_any = True

                                # Add speech to buffer (formatted as complete log line)
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                log_line = f"[{timestamp}] [{source_tag}] {content}"
                                self.agent_speeches_buffer.append(log_line)

                                # Check if host speech needs to be triggered
                                if len(self.agent_speeches_buffer) >= self.host_speech_threshold and not self.is_host_generating:
                                    # Synchronously trigger host speech
                                    self._trigger_host_speech()

                    elif current_lines < previous_lines:
                        any_shrink = True
                        # logger.info(f"ForumEngine: Detected {app_name} log shrink, will reset baseline")
                        # Reset file position to new end of file
                        self.file_positions[app_name] = self.get_file_size(log_file)
                        # Reset JSON capture state
                        self.capturing_json[app_name] = False
                        self.json_buffer[app_name] = []
                        self.in_error_block[app_name] = False

                    # Update line count record
                    self.file_line_counts[app_name] = current_lines

                # Check if current search session should end
                if self.is_searching:
                    if any_shrink:
                        # log shortened, end current search session, reset to waiting state
                        # logger.info("ForumEngine: Log shortened, ending current search session, returning to waiting state")
                        self.is_searching = False
                        self.search_inactive_count = 0
                        # Reset host-related state
                        self.agent_speeches_buffer = []
                        self.is_host_generating = False
                        # Write end marker
                        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.write_to_forum_log(f"=== ForumEngine Forum Ended - {end_time} ===", "SYSTEM")
                        # logger.info("ForumEngine: Baseline reset, waiting for next FirstSummaryNode trigger")
                    elif not any_growth and not captured_any:
                        # No growth and no content captured, increase inactivity count
                        self.search_inactive_count += 1
                        if self.search_inactive_count >= 7200:  # Auto-end after timeout with no activity
                            logger.info("ForumEngine: No activity for long time, ending forum")
                            self.is_searching = False
                            self.search_inactive_count = 0
                            # Reset host-related state
                            self.agent_speeches_buffer = []
                            self.is_host_generating = False
                            # Write end marker
                            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            self.write_to_forum_log(f"=== ForumEngine Forum Ended - {end_time} ===", "SYSTEM")
                    else:
                        self.search_inactive_count = 0  # Reset counter

                # Brief sleep
                time.sleep(1)

            except Exception as e:
                logger.exception(f"ForumEngine: Error in forum recording: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2)

        logger.info("ForumEngine: Stopping forum log monitoring")
   
    def start_monitoring(self):
        """Start intelligent monitoring"""
        if self.is_monitoring:
            logger.info("ForumEngine: Forum already running")
            return False
       
        try:
            # Start monitoring
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_logs, daemon=True)
            self.monitor_thread.start()
           
            logger.info("ForumEngine: Forum started")
            return True
           
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to start forum: {e}")
            self.is_monitoring = False
            return False
   
    def stop_monitoring(self):
        """Stop monitoring"""
        if not self.is_monitoring:
            logger.info("ForumEngine: Forum not running")
            return
       
        try:
            self.is_monitoring = False
           
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
           
            # Write end marker
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.write_to_forum_log(f"=== ForumEngine Forum Ended - {end_time} ===", "SYSTEM")
           
            logger.info("ForumEngine: Forum stopped")
           
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to stop forum: {e}")
   
    def get_forum_log_content(self) -> List[str]:
        """Get forum.log content"""
        try:
            if not self.forum_log_file.exists():
                return []

            with open(self.forum_log_file, 'r', encoding='utf-8') as f:
                return [line.rstrip('\n\r') for line in f.readlines()]

        except Exception as e:
            logger.exception(f"ForumEngine: Failed to read forum.log: {e}")
            return []

    def fix_json_string(self, json_text: str) -> str:
        """Fix common issues in JSON strings, especially unescaped double quotes"""
        try:
            # Try to parse directly, if successful return original text
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass

        # Fix unescaped double quote issues
        # This is a smarter fix method specifically for handling quotes inside string values

        try:
            # Use state machine method to fix JSON
            # Iterate through characters, tracking whether inside a string value

            fixed_text = ""
            i = 0
            in_string = False
            escape_next = False

            while i < len(json_text):
                char = json_text[i]

                if escape_next:
                    # Handle escape characters
                    fixed_text += char
                    escape_next = False
                    i += 1
                    continue

                if char == '\\':
                    # Escape character
                    fixed_text += char
                    escape_next = True
                    i += 1
                    continue

                if char == '"' and not escape_next:
                    # Encountered double quote
                    if in_string:
                        # Inside a string, check the next character
                        # If the next character is colon, comma, or brace, this is the end of the string
                        next_char_pos = i + 1
                        while next_char_pos < len(json_text) and json_text[next_char_pos].isspace():
                            next_char_pos += 1

                        if next_char_pos < len(json_text):
                            next_char = json_text[next_char_pos]
                            if next_char in [':', ',', '}']:
                                # This is the end of the string, exit string state
                                in_string = False
                                fixed_text += char
                            else:
                                # This is a quote inside the string, needs escaping
                                fixed_text += '\\"'
                        else:
                            # End of file, exit string state
                            in_string = False
                            fixed_text += char
                    else:
                        # String start
                        in_string = True
                        fixed_text += char
                else:
                    # Other characters
                    fixed_text += char

                i += 1

            # Try to parse the fixed JSON
            try:
                json.loads(fixed_text)
                return fixed_text
            except json.JSONDecodeError:
                # Fix failed, return None
                return None

        except Exception:
            return None

# Global monitor instance
_monitor_instance = None

def get_monitor() -> LogMonitor:
    """Get global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = LogMonitor()
    return _monitor_instance

def start_forum_monitoring():
    """Start ForumEngine intelligent monitoring"""
    return get_monitor().start_monitoring()

def stop_forum_monitoring():
    """Stop ForumEngine monitoring"""
    get_monitor().stop_monitoring()

def get_forum_log():
    """Get forum.log content"""
    return get_monitor().get_forum_log_content()
