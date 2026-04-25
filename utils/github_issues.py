"""
GitHub Issues utility module

Provides functionality for creating GitHub Issues URLs and displaying error messages with links
Data model definitions:
- No data models
"""

from datetime import datetime
from urllib.parse import quote

# GitHub repository information
# Configure this to point to your own repository
GITHUB_REPO = "your-username/your-repo"
GITHUB_ISSUES_URL = f"https://github.com/{GITHUB_REPO}/issues/new"


def create_issue_url(title: str, body: str = "") -> str:
    """
    Create GitHub Issues URL with pre-filled title and content
    
    Args:
        title: Issue title
        body: Issue content (optional)
    
    Returns:
        Complete GitHub Issues URL
    """
    encoded_title = quote(title)
    encoded_body = quote(body) if body else ""
    
    if encoded_body:
        return f"{GITHUB_ISSUES_URL}?title={encoded_title}&body={encoded_body}"
    else:
        return f"{GITHUB_ISSUES_URL}?title={encoded_title}"


def error_with_issue_link(
    error_message: str,
    error_details: str = "",
    app_name: str = "Streamlit App"
) -> str:
    """
    Generate error message string with GitHub Issues link
    
    Used only in generic exception handling, not for user configuration errors
    
    Args:
        error_message: Error message
        error_details: Error details (optional, used to populate Issue body)
        app_name: Application name for identifying error source
    
    Returns:
        Markdown formatted string containing error message and GitHub Issues link
    """
    issue_title = f"[{app_name}] {error_message[:50]}"
    issue_body = f"## Error Information\n\n{error_message}\n\n"
    
    if error_details:
        issue_body += f"## Error Details\n\n```\n{error_details}\n```\n\n"
    
    issue_body += f"## Environment Info\n\n- Application: {app_name}\n- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    issue_url = create_issue_url(issue_title, issue_body)
    
    # Add hyperlink using markdown format
    error_display = f"{error_message}\n\n[📝 Submit Error Report]({issue_url})"
    
    if error_details:
        error_display = f"{error_message}\n\n```\n{error_details}\n```\n\n[📝 Submit Error Report]({issue_url})"
    
    return error_display


__all__ = [
    "create_issue_url",
    "error_with_issue_link",
    "GITHUB_REPO",
    "GITHUB_ISSUES_URL",
]

