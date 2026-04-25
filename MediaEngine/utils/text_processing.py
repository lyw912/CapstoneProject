"""
Text processing utility functions
Used for cleaning LLM output, parsing JSON, etc.
"""

import re
import json
from typing import Dict, Any, List, Optional
from json.decoder import JSONDecodeError


def clean_json_tags(text: str) -> str:
    """
    Clean JSON tags from text
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove ```json and ``` tags
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def clean_markdown_tags(text: str) -> str:
    """
    Clean Markdown tags from text
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove ```markdown and ``` tags
    text = re.sub(r'```markdown\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def remove_reasoning_from_output(text: str) -> str:
    """
    Remove reasoning process text from output
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Find JSON start position
    json_start = -1
    
    # Try to find first { or [
    for i, char in enumerate(text):
        if char in '{[':
            json_start = i
            break
    
    if json_start != -1:
        # Truncate from JSON start position
        return text[json_start:].strip()
    
    # If JSON markers not found, try other methods
    # Remove common reasoning markers
    patterns = [
        r'(?:reasoning|推理|思考|分析)[:：]\s*.*?(?=\{|\[)',  # Remove reasoning section
        r'(?:explanation|解释|说明)[:：]\s*.*?(?=\{|\[)',   # Remove explanation section
        r'^.*?(?=\{|\[)',  # Remove all text before JSON
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def extract_clean_response(text: str) -> Dict[str, Any]:
    """
    Extract and clean JSON content from response
    
    Args:
        text: Raw response text
        
    Returns:
        Parsed JSON dictionary
    """
    # Clean text
    cleaned_text = clean_json_tags(text)
    cleaned_text = remove_reasoning_from_output(cleaned_text)
    
    # Try direct parsing
    try:
        return json.loads(cleaned_text)
    except JSONDecodeError:
        pass
    
    # Try to fix incomplete JSON
    fixed_text = fix_incomplete_json(cleaned_text)
    if fixed_text:
        try:
            return json.loads(fixed_text)
        except JSONDecodeError:
            pass
    
    # Try to find JSON objects
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # Try to find JSON arrays
    array_pattern = r'\[.*\]'
    match = re.search(array_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # If all methods fail, return error message
    print(f"Unable to parse JSON response: {cleaned_text[:200]}...")
    return {"error": "JSON parsing failed", "raw_text": cleaned_text}


def fix_incomplete_json(text: str) -> str:
    """
    Fix incomplete JSON response
    
    Args:
        text: Raw text
        
    Returns:
        Fixed JSON text, returns empty string if unable to fix
    """
    # Remove extra commas and whitespace
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Check if already valid JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        pass
    
    # Check if missing opening array bracket
    if text.strip().startswith('{') and not text.strip().startswith('['):
        # If starts with object, try wrapping as array
        if text.count('{') > 1:
            # Multiple objects, wrap as array
            text = '[' + text + ']'
        else:
            # Single object, wrap as array
            text = '[' + text + ']'
    
    # Check if missing closing array bracket
    if text.strip().endswith('}') and not text.strip().endswith(']'):
        # If ends with object, try wrapping as array
        if text.count('}') > 1:
            # Multiple objects, wrap as array
            text = '[' + text + ']'
        else:
            # Single object, wrap as array
            text = '[' + text + ']'
    
    # Check bracket matching
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Fix mismatched brackets
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    # Validate if fixed JSON is valid
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        # If still invalid, try more aggressive fix
        return fix_aggressive_json(text)


def fix_aggressive_json(text: str) -> str:
    """
    More aggressive JSON repair method
    
    Args:
        text: Raw text
        
    Returns:
        Fixed JSON text
    """
    # Find all possible JSON objects
    objects = re.findall(r'\{[^{}]*\}', text)
    
    if len(objects) >= 2:
        # If multiple objects, wrap as array
        return '[' + ','.join(objects) + ']'
    elif len(objects) == 1:
        # If only one object, wrap as array
        return '[' + objects[0] + ']'
    else:
        # If no objects found, return empty array
        return '[]'


def update_state_with_search_results(search_results: List[Dict[str, Any]], 
                                   paragraph_index: int, state: Any) -> Any:
    """
    Update search results to state
    
    Args:
        search_results: List of search results
        paragraph_index: Paragraph index
        state: State object
        
    Returns:
        Updated state object
    """
    if 0 <= paragraph_index < len(state.paragraphs):
        # Get the last search query (assumed to be current query)
        current_query = ""
        if search_results:
            # Infer query from search results (needs improvement to get actual query)
            current_query = "Search query"
        
        # Add search results to state
        state.paragraphs[paragraph_index].research.add_search_results(
            current_query, search_results
        )
    
    return state


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate whether JSON data contains required fields
    
    Args:
        data: Data to validate
        required_fields: List of required fields
        
    Returns:
        Whether validation passed
    """
    return all(field in data for field in required_fields)


def image_results_to_search_dicts(
    images: Optional[List[Any]],
    max_images: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convert search API returned image metadata to consistent structure with webpage results, for format_search_results_for_prompt use.
    Only includes text description and links, no pixel data.
    """
    if not images:
        return []
    out: List[Dict[str, Any]] = []
    for img in images[:max_images]:
        name = getattr(img, "name", None) or "图片"
        content_url = (getattr(img, "content_url", None) or "").strip()
        host = (getattr(img, "host_page_url", None) or "").strip() or None
        thumb = (getattr(img, "thumbnail_url", None) or "").strip() or None
        w, h = getattr(img, "width", None), getattr(img, "height", None)
        lines = [f"[图片] {name}"]
        lines.append(f"图片链接: {content_url}" if content_url else "图片链接: （无）")
        if host:
            lines.append(f"来源页面: {host}")
        if thumb:
            lines.append(f"缩略图: {thumb}")
        if w and h:
            lines.append(f"尺寸: {w}x{h}")
        text = "\n".join(lines)
        primary_url = content_url or host or ""
        out.append({
            "title": name,
            "url": primary_url,
            "content": text,
            "raw_content": text,
            "score": None,
            "published_date": None,
            "result_type": "image",
        })
    return out


def build_search_results_from_response(
    response: Any,
    max_webpages: int = 10,
    max_images: int = 10,
) -> List[Dict[str, Any]]:
    """
    Assemble webpage snippets + image metadata from search response, for paragraph summary and reflection.
    """
    if not response:
        return []
    out: List[Dict[str, Any]] = []
    webpages = getattr(response, "webpages", None) or []
    for result in webpages[:max_webpages]:
        out.append({
            "title": result.name,
            "url": result.url,
            "content": result.snippet,
            "score": None,
            "raw_content": result.snippet,
            "published_date": result.date_last_crawled,
            "result_type": "webpage",
        })
    out.extend(image_results_to_search_dicts(
        getattr(response, "images", None) or [],
        max_images=max_images,
    ))
    return out


def truncate_content(content: str, max_length: int = 20000) -> str:
    """
    Truncate content to specified length
    
    Args:
        content: Original content
        max_length: Maximum length
        
    Returns:
        Truncated content
    """
    if len(content) <= max_length:
        return content
    
    # Try to truncate at word boundary
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If last space position is reasonable
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def format_search_results_for_prompt(search_results: List[Dict[str, Any]], 
                                   max_length: int = 20000) -> List[str]:
    """
    Format search results for prompt
    
    Args:
        search_results: List of search results
        max_length: Maximum length for each result
        
    Returns:
        List of formatted content
    """
    formatted_results = []
    
    for result in search_results:
        content = result.get('content', '')
        if content:
            truncated_content = truncate_content(content, max_length)
            formatted_results.append(truncated_content)
    
    return formatted_results
