"""
Unified JSON parsing and repair utility.

Provides robust JSON parsing capabilities, supporting:
1. Automatic cleanup of markdown code block markers and thinking content
2. Local syntax repair (bracket balancing, comma completion, control character escaping, etc.)
3. Advanced repair using json_repair library
4. LLM-assisted repair (optional)
5. Detailed error logging and debugging information
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Callable
from loguru import logger

try:
    from json_repair import repair_json as _json_repair_fn
except ImportError:
    _json_repair_fn = None


class JSONParseError(ValueError):
    """Exception raised when JSON parsing fails, with original text attached for troubleshooting."""

    def __init__(self, message: str, raw_text: Optional[str] = None):
        """
        Construct exception and attach original output for easier log location.

        Args:
            message: Human-readable error description.
            raw_text: Complete LLM output that triggered the exception.
        """
        super().__init__(message)
        self.raw_text = raw_text


class RobustJSONParser:
    """
    Robust JSON parser.

    Integrates multiple repair strategies to ensure LLM returns can be correctly parsed:
    - Cleanup of markdown wrapping, thinking content, and other extra information
    - Fix common syntax errors (missing commas, unbalanced brackets, etc.)
    - Escape unescaped control characters
    - Use third-party library for advanced repair
    - Optional LLM-assisted repair
    """

    # Common LLM thinking content patterns
    _THINKING_PATTERNS = [
        r"^\s*<thinking>.*?</thinking>\s*",
        r"^\s*<thought>.*?</thought>\s*",
        r"^\s*让我想想.*?(?=\{|\[|$)",
        r"^\s*首先.*?(?=\{|\[|$)",
        r"^\s*分析.*?(?=\{|\[|$)",
        r"^\s*根据.*?(?=\{|\[|$)",
    ]

    # Colon-equals pattern (common LLM error)
    _COLON_EQUALS_PATTERN = re.compile(r'(":\s*)=')

    def __init__(
        self,
        llm_repair_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        enable_json_repair: bool = True,
        enable_llm_repair: bool = False,
        max_repair_attempts: int = 3,
    ):
        """
        Initialize JSON parser.

        Args:
            llm_repair_fn: Optional LLM repair function, receives (raw JSON, error message) and returns repaired JSON
            enable_json_repair: Whether to enable json_repair library
            enable_llm_repair: Whether to enable LLM-assisted repair
            max_repair_attempts: Maximum number of repair attempts
        """
        self.llm_repair_fn = llm_repair_fn
        self.enable_json_repair = enable_json_repair and _json_repair_fn is not None
        self.enable_llm_repair = enable_llm_repair
        self.max_repair_attempts = max_repair_attempts

    def parse(
        self,
        raw_text: str,
        context_name: str = "JSON",
        expected_keys: Optional[List[str]] = None,
        extract_wrapper_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse JSON text returned by LLM.

        Args:
            raw_text: LLM raw output (may contain ``` wrapping, thinking content, etc.)
            context_name: Context name for error messages
            expected_keys: List of expected keys for validation
            extract_wrapper_key: If JSON is wrapped in a key, specify the key name for extraction

        Returns:
            dict: Parsed JSON object

        Raises:
            JSONParseError: When multiple repair strategies still cannot parse valid JSON
        """
        if not raw_text or not raw_text.strip():
            raise JSONParseError(f"{context_name} returned empty content")

        # Original text for later logging
        original_text = raw_text

        # Step 1: Build candidates with different cleanup strategies
        candidates = self._build_candidate_payloads(raw_text, context_name)

        # Step 2: Try parsing all candidates
        last_error: Optional[json.JSONDecodeError] = None
        for i, candidate in enumerate(candidates):
            try:
                data = json.loads(candidate)
                logger.debug(f"{context_name} JSON parsing successful (candidate {i + 1}/{len(candidates)})")
                return self._extract_and_validate(
                    data, expected_keys, extract_wrapper_key, context_name
                )
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.debug(f"{context_name} candidate {i + 1} parsing failed: {exc}")

        cleaned = candidates[0] if candidates else original_text

        # Step 3: Use json_repair library
        if self.enable_json_repair:
            repaired = self._attempt_json_repair(cleaned, context_name)
            if repaired:
                try:
                    data = json.loads(repaired)
                    logger.info(f"{context_name} JSON repaired successfully via json_repair library")
                    return self._extract_and_validate(
                        data, expected_keys, extract_wrapper_key, context_name
                    )
                except json.JSONDecodeError as exc:
                    last_error = exc
                    logger.debug(f"{context_name} still cannot parse after json_repair: {exc}")

        # Step 4: Use LLM repair (if enabled)
        if self.enable_llm_repair and self.llm_repair_fn:
            llm_repaired = self._attempt_llm_repair(cleaned, str(last_error), context_name)
            if llm_repaired:
                try:
                    data = json.loads(llm_repaired)
                    logger.info(f"{context_name} JSON repaired successfully via LLM")
                    return self._extract_and_validate(
                        data, expected_keys, extract_wrapper_key, context_name
                    )
                except json.JSONDecodeError as exc:
                    last_error = exc
                    logger.warning(f"{context_name} still cannot parse after LLM repair: {exc}")

        # All strategies failed
        error_msg = f"{context_name} JSON parsing failed: {last_error}"
        logger.error(error_msg)
        logger.debug(f"Original text first 500 chars: {original_text[:500]}")
        raise JSONParseError(error_msg, raw_text=original_text) from last_error

    def _build_candidate_payloads(self, raw_text: str, context_name: str) -> List[str]:
        """
        Construct multiple candidate JSON strings from raw text, covering different cleanup strategies.

        Returns:
            List[str]: List of candidate JSON text
        """
        cleaned = self._clean_response(raw_text)
        candidates = [cleaned]

        local_repaired = self._apply_local_repairs(cleaned)
        if local_repaired != cleaned:
            candidates.append(local_repaired)

        # Force flatten once for content with three-layer list structure
        flattened = self._flatten_nested_arrays(local_repaired)
        if flattened not in candidates:
            candidates.append(flattened)

        return candidates

    def _clean_response(self, raw: str) -> str:
        """
        Clean LLM response, remove markdown markers and thinking content.

        Args:
            raw: LLM raw output

        Returns:
            str: Cleaned text
        """
        cleaned = raw.strip()

        # Remove thinking content (multi-language support)
        for pattern in self._THINKING_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Priority extraction of ```json``` wrapped content at any position
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        else:
            # If no complete code block found, try removing prefix and suffix
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]

            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            cleaned = cleaned.strip()

        # Try to extract the first complete JSON object or array
        cleaned = self._extract_first_json_structure(cleaned)

        return cleaned

    def _extract_first_json_structure(self, text: str) -> str:
        """
        Extract the first complete JSON object or array from text.

        Useful for handling cases where LLM adds explanatory text before or after JSON.

        Args:
            text: Text that may contain JSON

        Returns:
            str: Extracted JSON text, returns original text if not found
        """
        # Find first { or [
        start_brace = text.find("{")
        start_bracket = text.find("[")

        if start_brace == -1 and start_bracket == -1:
            return text

        # Determine start position
        if start_brace == -1:
            start = start_bracket
            opener = "["
            closer = "]"
        elif start_bracket == -1:
            start = start_brace
            opener = "{"
            closer = "}"
        else:
            start = min(start_brace, start_bracket)
            opener = text[start]
            closer = "}" if opener == "{" else "]"

        # Find corresponding end position
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if escaped:
                escaped = False
                continue

            if ch == "\\":
                escaped = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        # If complete structure not found, return from start to end
        return text[start:] if start < len(text) else text

    def _apply_local_repairs(self, text: str) -> str:
        """
        Apply local repair strategies.

        Args:
            text: Raw JSON text

        Returns:
            str: Repaired text
        """
        repaired = text
        mutated = False

        # Fix ":=" error
        new_text = self._COLON_EQUALS_PATTERN.sub(r"\1", repaired)
        if new_text != repaired:
            logger.warning("Detected \":=\" character, automatically removed extra '='")
            repaired = new_text
            mutated = True

        # Escape control characters
        repaired, escaped = self._escape_control_characters(repaired)
        if escaped:
            logger.warning("Detected unescaped control characters, automatically converted to escape sequences")
            mutated = True

        # Fix missing commas
        repaired, commas_fixed = self._fix_missing_commas(repaired)
        if commas_fixed:
            logger.warning("Detected missing commas between objects/arrays, automatically added")
            mutated = True

        # Collapse redundant brackets (LLM often writes three-layer lists for 2D structures)
        repaired, brackets_collapsed = self._collapse_redundant_brackets(repaired)
        if brackets_collapsed:
            logger.warning("Detected consecutive nested brackets, attempted to collapse to 2D structure")
            mutated = True

        # Balance brackets
        repaired, balanced = self._balance_brackets(repaired)
        if balanced:
            logger.warning("Detected unbalanced brackets, automatically fixed by adding/removing")
            mutated = True

        # Remove trailing commas
        repaired, trailing_removed = self._remove_trailing_commas(repaired)
        if trailing_removed:
            logger.warning("Detected trailing commas, automatically removed")
            mutated = True

        return repaired if mutated else text

    def _escape_control_characters(self, text: str) -> Tuple[str, bool]:
        """
        Replace raw newlines/tabs/control characters in string literals with JSON-legal escape sequences.

        Args:
            text: Raw JSON text

        Returns:
            Tuple[str, bool]: (Repaired text, whether modified)
        """
        if not text:
            return text, False

        result: List[str] = []
        in_string = False
        escaped = False
        mutated = False
        control_map = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}

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
        """
        Automatically add commas between object/array elements.

        Args:
            text: Raw JSON text

        Returns:
            Tuple[str, bool]: (Repaired text, whether modified)
        """
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
                # If we are exiting string, check if comma needed after
                if in_string:
                    # Find next non-whitespace character
                    j = i + 1
                    while j < length and text[j] in " \t\r\n":
                        j += 1
                    # If next char is " { [ or digit, comma may be needed
                    if j < length:
                        next_ch = text[j]
                        if next_ch in "\"[{" or next_ch.isdigit():
                            # Check if already inside object or array
                            # By checking if there's unclosed { or [ before
                            has_opener = False
                            for k in range(len(chars) - 1, -1, -1):
                                if chars[k] in "{[":
                                    has_opener = True
                                    break
                                elif chars[k] in "]}":
                                    break

                            if has_opener:
                                chars.append(",")
                                mutated = True

                in_string = not in_string
                i += 1
                continue

            # Check if comma needed after } or ]
            if not in_string and ch in "}]":
                j = i + 1
                # Skip whitespace
                while j < length and text[j] in " \t\r\n":
                    j += 1
                # If next non-whitespace char is { [ " or digit, add comma
                if j < length:
                    next_ch = text[j]
                    if next_ch in "{[\"" or next_ch.isdigit():
                        chars.append(",")
                        mutated = True

            i += 1

        return "".join(chars), mutated

    def _collapse_redundant_brackets(self, text: str) -> Tuple[str, bool]:
        """
        Collapse three or more layer arrays generated by LLM (like ]]], [[ / [[[) to avoid extra dimensions in tables/lists.

        Returns:
            Tuple[str, bool]: (Repaired text, whether modified)
        """
        if not text:
            return text, False

        mutated = False

        patterns = [
            # 典型错误: "]]], [[{...}" -> "]], [{...}"
            (re.compile(r"\]\s*\]\s*\]\s*,\s*\[\s*\["), "]],["),
            # 极端情况: 连续三层开头 "[[[" -> "[["
            (re.compile(r"\[\s*\[\s*\["), "[["),
            # 极端情况: 结尾 "]]]" -> "]]"
            (re.compile(r"\]\s*\]\s*\]"), "]]"),
        ]

        repaired = text
        for pattern, replacement in patterns:
            new_text, count = pattern.subn(replacement, repaired)
            if count > 0:
                mutated = True
                repaired = new_text

        return repaired, mutated

    def _flatten_nested_arrays(self, text: str) -> str:
        """
        Collapse obviously redundant nested lists, e.g., [[[x]]] -> [[x]].
        """
        if not text:
            return text
        text = re.sub(r"\]\s*\]\s*\]", "]]", text)
        text = re.sub(r"\[\s*\[\s*\[", "[[", text)
        return text

    def _balance_brackets(self, text: str) -> Tuple[str, bool]:
        """
        Attempt to repair unbalanced structures caused by LLM writing too many/few brackets.

        Args:
            text: Raw JSON text

        Returns:
            Tuple[str, bool]: (Repaired text, whether modified)
        """
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
                if stack and (
                    (ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")
                ):
                    stack.pop()
                    result.append(ch)
                else:
                    # Unmatched closing bracket, ignore
                    mutated = True
                continue

            result.append(ch)

        # Add missing closing brackets
        while stack:
            opener = stack.pop()
            result.append(opener_map[opener])
            mutated = True

        return "".join(result), mutated

    def _remove_trailing_commas(self, text: str) -> Tuple[str, bool]:
        """
        Remove trailing commas in JSON objects and arrays.

        Args:
            text: Raw JSON text

        Returns:
            Tuple[str, bool]: (Repaired text, whether modified)
        """
        if not text:
            return text, False

        # Use regex to remove trailing commas
        # Match comma followed by whitespace and } or ]
        pattern = r",(\s*[}\]])"
        new_text = re.sub(pattern, r"\1", text)

        return new_text, new_text != text

    def _attempt_json_repair(self, text: str, context_name: str) -> Optional[str]:
        """
        Use json_repair library for advanced repair.

        Args:
            text: Raw JSON text
            context_name: Context name

        Returns:
            Optional[str]: Repaired JSON text, returns None if failed
        """
        if not _json_repair_fn:
            return None

        try:
            fixed = _json_repair_fn(text)
            if fixed and fixed != text:
                logger.info(f"{context_name} JSON automatically repaired using json_repair library")
                return fixed
        except Exception as exc:
            logger.debug(f"{context_name} json_repair repair failed: {exc}")

        return None

    def _attempt_llm_repair(
        self, text: str, error_msg: str, context_name: str
    ) -> Optional[str]:
        """
        Use LLM for JSON repair.

        Args:
            text: Raw JSON text
            error_msg: Parsing error message
            context_name: Context name

        Returns:
            Optional[str]: Repaired JSON text, returns None if failed
        """
        if not self.llm_repair_fn:
            return None

        try:
            logger.info(f"{context_name} Attempting to repair JSON using LLM")
            repaired = self.llm_repair_fn(text, error_msg)
            if repaired and repaired != text:
                return repaired
        except Exception as exc:
            logger.warning(f"{context_name} LLM repair failed: {exc}")

        return None

    def _extract_and_validate(
        self,
        data: Any,
        expected_keys: Optional[List[str]],
        extract_wrapper_key: Optional[str],
        context_name: str,
    ) -> Dict[str, Any]:
        """
        Extract and validate JSON data.

        Args:
            data: Parsed data
            expected_keys: List of expected keys
            extract_wrapper_key: Wrapper key name
            context_name: Context name

        Returns:
            Dict[str, Any]: Extracted and validated data

        Raises:
            JSONParseError: If data format does not meet expectations
        """
        # Extract wrapped data
        if extract_wrapper_key and isinstance(data, dict):
            if extract_wrapper_key in data:
                data = data[extract_wrapper_key]
            else:
                logger.warning(
                    f"{context_name} Wrapper key '{extract_wrapper_key}' not found, using original data"
                )

        # Validate data type
        if not isinstance(data, dict):
            if isinstance(data, list):
                if len(data) > 0:
                    # Try to find the best matching element
                    best_match = None
                    max_match_count = 0

                    for item in data:
                        if isinstance(item, dict):
                            if expected_keys:
                                # Calculate number of matching keys
                                match_count = sum(1 for key in expected_keys if key in item)
                                if match_count > max_match_count:
                                    max_match_count = match_count
                                    best_match = item
                            elif best_match is None:
                                best_match = item

                    if best_match:
                        logger.warning(
                            f"{context_name} Returned array, automatically extracted best matching element (matched {max_match_count}/{len(expected_keys or [])} keys)"
                        )
                        data = best_match
                    else:
                        raise JSONParseError(
                            f"{context_name} No valid object found in returned array"
                        )
                else:
                    raise JSONParseError(f"{context_name} Returned empty array")
            else:
                raise JSONParseError(
                    f"{context_name} Returned value is not a JSON object: {type(data).__name__}"
                )

        # Validate required keys
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                logger.warning(
                    f"{context_name} Missing expected keys: {', '.join(missing_keys)}"
                )
                # Try to repair common key name variants
                data = self._try_recover_missing_keys(data, missing_keys, context_name)

        return data

    def _try_recover_missing_keys(
        self, data: Dict[str, Any], missing_keys: List[str], context_name: str
    ) -> Dict[str, Any]:
        """
        Try to recover missing keys from data by looking for similar key names.

        Args:
            data: Original data
            missing_keys: List of missing keys
            context_name: Context name

        Returns:
            Dict[str, Any]: Repaired data
        """
        # Common key name mappings
        key_aliases = {
            "template_name": ["templateName", "name", "template"],
            "selection_reason": ["selectionReason", "reason", "explanation"],
            "title": ["reportTitle", "documentTitle"],
            "chapters": ["chapterList", "chapterPlan", "sections"],
            "totalWords": ["total_words", "wordCount", "totalWordCount"],
        }

        for missing_key in missing_keys:
            if missing_key in key_aliases:
                for alias in key_aliases[missing_key]:
                    if alias in data:
                        logger.info(
                            f"{context_name} Found alias '{alias}' for key '{missing_key}', automatically mapping"
                        )
                        data[missing_key] = data[alias]
                        break

        return data


__all__ = ["RobustJSONParser", "JSONParseError"]
