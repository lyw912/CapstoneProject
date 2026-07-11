"""
Deep Search Agent state management
Defines all state data structures and operations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


@dataclass
class Search:
    """Status of a single search result"""
    query: str = ""                    # Search query
    url: str = ""                      # URL of search result
    title: str = ""                    # Title of search result
    content: str = ""                  # Content returned from search
    score: Optional[float] = None      # Relevance score
    paragraph_title: str = ""          # Paragraph title for display归属
    search_tool: str = ""              # Search tool used
    has_result: bool = True            # Whether result is returned
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    result_type: str = "webpage"
    image_url: str = ""
    published_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "query": self.query,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "paragraph_title": self.paragraph_title,
            "search_tool": self.search_tool,
            "has_result": self.has_result,
            "timestamp": self.timestamp,
            "result_type": self.result_type,
            "image_url": self.image_url,
            "published_at": self.published_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Search":
        """Create Search object from dictionary"""
        return cls(
            query=data.get("query", ""),
            url=data.get("url", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            score=data.get("score"),
            paragraph_title=data.get("paragraph_title", ""),
            search_tool=data.get("search_tool", ""),
            has_result=data.get("has_result", True),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            result_type=data.get("result_type", "webpage"),
            image_url=data.get("image_url", ""),
            published_at=data.get("published_at") or data.get("published_date"),
            metadata=data.get("metadata", {}),
        )


def _json_safe_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata_value(item) for key, item in value.items()}
    return str(value)


@dataclass
class Research:
    """段落研究过程的状态"""
    search_history: List[Search] = field(default_factory=list)     # 搜索记录列表
    latest_summary: str = ""                                       # 当前段落的最新总结
    reflection_iteration: int = 0                                  # 反思迭代次数
    is_completed: bool = False                                     # 是否完成研究
    
    def add_search(self, search: Search):
        """添加搜索记录"""
        self.search_history.append(search)
    
    def add_search_results(self, query: str, results: List[Dict[str, Any]], search_tool: str = "", paragraph_title: str = ""):
        """批量添加搜索结果"""
        if not results:
            # 记录一次“无结果”搜索，方便前端显示搜索轨迹
            self.add_search(
                Search(
                    query=query or "",
                    title="No results found",
                    content="This search returned no results or the call failed",
                    url="",
                    score=None,
                    paragraph_title=paragraph_title,
                    search_tool=search_tool,
                    has_result=False,
                    result_type="none",
                )
            )
            return

        for result in results:
            url = result.get("url") or ""
            title = result.get("title") or ""
            content = result.get("content") or result.get("raw_content") or ""
            if not isinstance(content, str):
                content = str(content)
            self.add_search(
                Search(
                    query=query or "",
                    url=url,
                    title=title,
                    content=content,
                    score=result.get("score"),
                    paragraph_title=paragraph_title or result.get("paragraph_title", ""),
                    search_tool=search_tool or result.get("search_tool", ""),
                    has_result=True,
                    result_type=result.get("result_type", "webpage"),
                    image_url=result.get("image_url") or (url if result.get("result_type") == "image" else ""),
                    published_at=result.get("published_at") or result.get("published_date"),
                    metadata={
                        key: _json_safe_metadata_value(value)
                        for key, value in result.items()
                        if key not in {"url", "title", "content", "raw_content", "score"}
                    },
                )
            )

    def get_search_count(self) -> int:
        """Get search count"""
        return len(self.search_history)
    
    def increment_reflection(self):
        """Increment reflection count"""
        self.reflection_iteration += 1
    
    def mark_completed(self):
        """Mark as completed"""
        self.is_completed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "search_history": [search.to_dict() for search in self.search_history],
            "latest_summary": self.latest_summary,
            "reflection_iteration": self.reflection_iteration,
            "is_completed": self.is_completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Research":
        """Create Research object from dictionary"""
        search_history = [Search.from_dict(search_data) for search_data in data.get("search_history", [])]
        return cls(
            search_history=search_history,
            latest_summary=data.get("latest_summary", ""),
            reflection_iteration=data.get("reflection_iteration", 0),
            is_completed=data.get("is_completed", False)
        )


@dataclass
class Paragraph:
    """Status of a single paragraph in the report"""
    title: str = ""                                                # Paragraph title
    content: str = ""                                              # Expected paragraph content (initial planning)
    research: Research = field(default_factory=Research)          # Research progress
    order: int = 0                                                 # Paragraph order
    
    def is_completed(self) -> bool:
        """Check if paragraph is completed"""
        return self.research.is_completed and bool(self.research.latest_summary)
    
    def get_final_content(self) -> str:
        """Get final content"""
        return self.research.latest_summary or self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "title": self.title,
            "content": self.content,
            "research": self.research.to_dict(),
            "order": self.order
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paragraph":
        """Create Paragraph object from dictionary"""
        research_data = data.get("research", {})
        research = Research.from_dict(research_data) if research_data else Research()
        
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            research=research,
            order=data.get("order", 0)
        )


@dataclass
class State:
    """Status of the entire report"""
    query: str = ""                                                # Original query
    report_title: str = ""                                         # Report title
    paragraphs: List[Paragraph] = field(default_factory=list)     # List of paragraphs
    final_report: str = ""                                         # Final report content
    is_completed: bool = False                                     # Whether completed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_paragraph(self, title: str, content: str) -> int:
        """
        Add paragraph
        
        Args:
            title: Paragraph title
            content: Paragraph content
            
        Returns:
            Paragraph index
        """
        order = len(self.paragraphs)
        paragraph = Paragraph(title=title, content=content, order=order)
        self.paragraphs.append(paragraph)
        self.update_timestamp()
        return order
    
    def get_paragraph(self, index: int) -> Optional[Paragraph]:
        """Get paragraph at specified index"""
        if 0 <= index < len(self.paragraphs):
            return self.paragraphs[index]
        return None
    
    def get_completed_paragraphs_count(self) -> int:
        """Get count of completed paragraphs"""
        return sum(1 for p in self.paragraphs if p.is_completed())
    
    def get_total_paragraphs_count(self) -> int:
        """Get total paragraph count"""
        return len(self.paragraphs)
    
    def is_all_paragraphs_completed(self) -> bool:
        """Check if all paragraphs are completed"""
        return all(p.is_completed() for p in self.paragraphs) if self.paragraphs else False
    
    def mark_completed(self):
        """Mark entire report as completed"""
        self.is_completed = True
        self.update_timestamp()
    
    def update_timestamp(self):
        """Update timestamp"""
        self.updated_at = datetime.now().isoformat()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        completed = self.get_completed_paragraphs_count()
        total = self.get_total_paragraphs_count()
        
        return {
            "total_paragraphs": total,
            "completed_paragraphs": completed,
            "progress_percentage": (completed / total * 100) if total > 0 else 0,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "query": self.query,
            "report_title": self.report_title,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "final_report": self.final_report,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Create State object from dictionary"""
        paragraphs = [Paragraph.from_dict(p_data) for p_data in data.get("paragraphs", [])]
        
        return cls(
            query=data.get("query", ""),
            report_title=data.get("report_title", ""),
            paragraphs=paragraphs,
            final_report=data.get("final_report", ""),
            is_completed=data.get("is_completed", False),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "State":
        """Create State object from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str):
        """Save state to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "State":
        """Load state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return cls.from_json(json_str)
