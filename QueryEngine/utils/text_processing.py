"""
Text Processing Utility Functions
Used for cleaning LLM output, parsing JSON, etc.
"""

import re
import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError


def clean_json_tags(text: str) -> str:
    """
    Clean JSON tags from text
    
    Args:
        text: Original text
        
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
        text: Original text
        
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
        text: Original text
        
    Returns:
        Cleaned text
    """
    # Find JSON start position
    json_start = -1
    
    # Try to find the first { or [
    for i, char in enumerate(text):
        if char in '{[':
            json_start = i
            break
    
    if json_start != -1:
        # Extract from JSON start position
        return text[json_start:].strip()
    
    # If no JSON marker found, try other methods
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
        text: Original response text
        
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
    
    # Try to find JSON object
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # Try to find JSON array
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
        text: Original text
        
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
    
    # Verify if fixed JSON is valid
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        # If still invalid, try more aggressive fix
        return fix_aggressive_json(text)


def fix_aggressive_json(text: str) -> str:
    """
    More aggressive JSON fix method
    
    Args:
        text: Original text
        
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
            current_query = "search query"
        
        # Add search results to state
        state.paragraphs[paragraph_index].research.add_search_results(
            current_query, search_results
        )
    
    return state


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate if JSON data contains required fields
    
    Args:
        data: Data to validate
        required_fields: List of required fields
        
    Returns:
        Whether validation passed
    """
    return all(field in data for field in required_fields)


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
