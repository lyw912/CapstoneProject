"""
Test log parsing functions in ForumEngine/monitor.py

Tests parsing capabilities under various log formats, including:
1. Old format: [HH:MM:SS]
2. New format: loguru default format (YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...)
3. Should only receive output from SummaryNodes like FirstSummaryNode, ReflectionSummaryNode, not SearchNode
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ForumEngine.monitor import LogMonitor
from tests import forum_log_test_data as test_data


class TestLogMonitor:
    """Test LogMonitor log parsing functionality"""
    
    def setup_method(self):
        """Initialize before each test method"""
        self.monitor = LogMonitor(log_dir="tests/test_logs")
    
    def test_is_target_log_line_old_format(self):
        """Test old format target node identification"""
        # Should identify lines containing FirstSummaryNode
        assert self.monitor.is_target_log_line(test_data.OLD_FORMAT_FIRST_SUMMARY) == True
        # Should identify lines containing ReflectionSummaryNode
        assert self.monitor.is_target_log_line(test_data.OLD_FORMAT_REFLECTION_SUMMARY) == True
        # Should not identify non-target nodes
        assert self.monitor.is_target_log_line(test_data.OLD_FORMAT_NON_TARGET) == False
    
    def test_is_target_log_line_new_format(self):
        """Test new format target node identification"""
        # Should identify lines containing FirstSummaryNode
        assert self.monitor.is_target_log_line(test_data.NEW_FORMAT_FIRST_SUMMARY) == True
        # Should identify lines containing ReflectionSummaryNode
        assert self.monitor.is_target_log_line(test_data.NEW_FORMAT_REFLECTION_SUMMARY) == True
        # Should not identify non-target nodes
        assert self.monitor.is_target_log_line(test_data.NEW_FORMAT_NON_TARGET) == False
    
    def test_is_json_start_line_old_format(self):
        """Test old format JSON start line identification"""
        assert self.monitor.is_json_start_line(test_data.OLD_FORMAT_SINGLE_LINE_JSON) == True
        assert self.monitor.is_json_start_line(test_data.OLD_FORMAT_MULTILINE_JSON[0]) == True
        assert self.monitor.is_json_start_line(test_data.OLD_FORMAT_NON_TARGET) == False
    
    def test_is_json_start_line_new_format(self):
        """Test new format JSON start line identification"""
        assert self.monitor.is_json_start_line(test_data.NEW_FORMAT_SINGLE_LINE_JSON) == True
        assert self.monitor.is_json_start_line(test_data.NEW_FORMAT_MULTILINE_JSON[0]) == True
        assert self.monitor.is_json_start_line(test_data.NEW_FORMAT_NON_TARGET) == False
    
    def test_is_json_end_line(self):
        """Test JSON end line identification"""
        assert self.monitor.is_json_end_line("}") == True
        assert self.monitor.is_json_end_line("] }") == True
        assert self.monitor.is_json_end_line("[17:42:31] }") == False  # 需要先清理时间戳
        assert self.monitor.is_json_end_line("2025-11-05 17:42:31.289 | INFO | module:function:133 - }") == False  # 需要先清理时间戳
    
    def test_extract_json_content_old_format_single_line(self):
        """Test old format single-line JSON extraction"""
        lines = [test_data.OLD_FORMAT_SINGLE_LINE_JSON]
        result = self.monitor.extract_json_content(lines)
        assert result is not None
        assert "这是首次总结内容" in result
    
    def test_extract_json_content_new_format_single_line(self):
        """Test new format single-line JSON extraction"""
        lines = [test_data.NEW_FORMAT_SINGLE_LINE_JSON]
        result = self.monitor.extract_json_content(lines)
        assert result is not None
        assert "这是首次总结内容" in result
    
    def test_extract_json_content_old_format_multiline(self):
        """Test old format multi-line JSON extraction"""
        result = self.monitor.extract_json_content(test_data.OLD_FORMAT_MULTILINE_JSON)
        assert result is not None
        assert "多行" in result
        assert "JSON内容" in result
    
    def test_extract_json_content_new_format_multiline(self):
        """Test new format multi-line JSON extraction (supports loguru format timestamp removal)"""
        result = self.monitor.extract_json_content(test_data.NEW_FORMAT_MULTILINE_JSON)
        assert result is not None
        assert "多行" in result
        assert "JSON内容" in result
    
    def test_extract_json_content_updated_priority(self):
        """Test updated_paragraph_latest_state priority extraction"""
        result = self.monitor.extract_json_content(test_data.COMPLEX_JSON_WITH_UPDATED)
        assert result is not None
        assert "更新版" in result
        assert "核心发现" in result
    
    def test_extract_json_content_paragraph_only(self):
        """Test paragraph_latest_state only case"""
        result = self.monitor.extract_json_content(test_data.COMPLEX_JSON_WITH_PARAGRAPH)
        assert result is not None
        assert "首次总结" in result or "核心发现" in result
    
    def test_format_json_content(self):
        """Test JSON content formatting"""
        # Test updated_paragraph_latest_state priority
        json_obj = {
            "updated_paragraph_latest_state": "Updated content",
            "paragraph_latest_state": "Initial content"
        }
        result = self.monitor.format_json_content(json_obj)
        assert result == "Updated content"
        
        # Test paragraph_latest_state only
        json_obj = {
            "paragraph_latest_state": "Initial content"
        }
        result = self.monitor.format_json_content(json_obj)
        assert result == "Initial content"
        
        # Test neither field exists
        json_obj = {"other_field": "Other content"}
        result = self.monitor.format_json_content(json_obj)
        assert "Cleaned output" in result
    
    def test_extract_node_content_old_format(self):
        """Test old format node content extraction"""
        line = "[17:42:31] [MEDIA] [FirstSummaryNode] 清理后的输出: 这是测试内容"
        result = self.monitor.extract_node_content(line)
        assert result is not None
        assert "测试内容" in result
    
    def test_extract_node_content_new_format(self):
        """Test new format node content extraction"""
        line = "2025-11-05 17:42:31.287 | INFO | InsightEngine.nodes.summary_node:process_output:131 - FirstSummaryNode 清理后的输出: 这是测试内容"
        result = self.monitor.extract_node_content(line)
        assert result is not None
        assert "测试内容" in result
    
    def test_process_lines_for_json_old_format(self):
        """Test old format complete processing flow"""
        lines = [
            test_data.OLD_FORMAT_NON_TARGET,  # Should be ignored
            test_data.OLD_FORMAT_MULTILINE_JSON[0],
            test_data.OLD_FORMAT_MULTILINE_JSON[1],
            test_data.OLD_FORMAT_MULTILINE_JSON[2],
        ]
        result = self.monitor.process_lines_for_json(lines, "media")
        assert len(result) > 0
        assert any("多行" in content for content in result)
    
    def test_process_lines_for_json_new_format(self):
        """Test new format complete processing flow"""
        lines = [
            test_data.NEW_FORMAT_NON_TARGET,  # Should be ignored
            test_data.NEW_FORMAT_MULTILINE_JSON[0],
            test_data.NEW_FORMAT_MULTILINE_JSON[1],
            test_data.NEW_FORMAT_MULTILINE_JSON[2],
        ]
        result = self.monitor.process_lines_for_json(lines, "media")
        assert len(result) > 0
        assert any("多行" in content for content in result)
        assert any("JSON内容" in content for content in result)
    
    def test_process_lines_for_json_mixed_format(self):
        """Test mixed format processing"""
        result = self.monitor.process_lines_for_json(test_data.MIXED_FORMAT_LINES, "media")
        assert len(result) > 0
        assert any("混合格式内容" in content for content in result)
    
    def test_is_valuable_content(self):
        """Test valuable content judgment"""
        # Containing "Cleaned output" should be valuable
        assert self.monitor.is_valuable_content(test_data.OLD_FORMAT_SINGLE_LINE_JSON) == True
        
        # Exclude short prompt messages
        assert self.monitor.is_valuable_content("JSON parsing successful") == False
        assert self.monitor.is_valuable_content("Generation successful") == False
        
        # Empty lines should be filtered
        assert self.monitor.is_valuable_content("") == False
    
    def test_extract_json_content_real_query_engine(self):
        """Test QueryEngine production environment log extraction"""
        result = self.monitor.extract_json_content(test_data.REAL_QUERY_ENGINE_REFLECTION)
        assert result is not None
        assert "洛阳栾川钼业集团" in result
        assert "CMOC" in result
        assert "updated_paragraph_latest_state" not in result  # 应该已经提取内容，不包含字段名
    
    def test_extract_json_content_real_insight_engine(self):
        """Test InsightEngine production environment log extraction (includes marker lines)"""
        # First test if marker lines can be identified
        assert self.monitor.is_target_log_line(test_data.REAL_INSIGHT_ENGINE_REFLECTION[0]) == True  # Contains "Generating reflection summary"
        assert self.monitor.is_target_log_line(test_data.REAL_INSIGHT_ENGINE_REFLECTION[1]) == True  # Contains nodes.summary_node
        
        # Test JSON extraction (start from second line since first is marker)
        json_lines = test_data.REAL_INSIGHT_ENGINE_REFLECTION[1:]  # Skip marker line
        result = self.monitor.extract_json_content(json_lines)
        assert result is not None
        assert "核心发现" in result
        assert "更新版" in result
        assert "洛阳钼业2025年第三季度" in result
    
    def test_extract_json_content_real_media_engine(self):
        """Test MediaEngine production environment log extraction (single-line JSON)"""
        # MediaEngine uses single-line JSON format, need to split into lines first
        lines = test_data.REAL_MEDIA_ENGINE_REFLECTION.split('\n')
        
        # Test if marker lines can be identified
        assert self.monitor.is_target_log_line(lines[0]) == True  # Contains "Generating reflection summary"
        assert self.monitor.is_target_log_line(lines[1]) == True  # Contains nodes.summary_node and "Cleaned output"
        
        # Test JSON extraction (start from line containing JSON)
        json_line = lines[1]  # Second line contains complete single-line JSON
        result = self.monitor.extract_json_content([json_line])
        assert result is not None
        assert "综合信息概览" in result
        assert "洛阳钼业" in result
        assert "updated_paragraph_latest_state" not in result  # 应该已经提取内容
    
    def test_process_lines_for_json_real_query_engine(self):
        """Test QueryEngine actual log complete processing flow"""
        result = self.monitor.process_lines_for_json(test_data.REAL_QUERY_ENGINE_REFLECTION, "query")
        assert len(result) > 0
        assert any("洛阳栾川钼业集团" in content for content in result)
    
    def test_process_lines_for_json_real_insight_engine(self):
        """Test InsightEngine actual log complete processing flow (includes marker lines)"""
        result = self.monitor.process_lines_for_json(test_data.REAL_INSIGHT_ENGINE_REFLECTION, "media")
        assert len(result) > 0
        assert any("核心发现" in content for content in result)
        assert any("更新版" in content for content in result)
    
    def test_process_lines_for_json_real_media_engine(self):
        """Test MediaEngine actual log complete processing flow (single-line JSON)"""
        # Split single-line string into multiple lines
        lines = test_data.REAL_MEDIA_ENGINE_REFLECTION.split('\n')
        result = self.monitor.process_lines_for_json(lines, "media")
        assert len(result) > 0
        assert any("综合信息概览" in content for content in result)
        assert any("洛阳钼业" in content for content in result)
    
    def test_filter_search_node_output(self):
        """Test filtering SearchNode output (Important: SearchNode should not enter forum)"""
        # SearchNode output contains "Cleaned output: {" but not target node patterns
        search_lines = test_data.SEARCH_NODE_FIRST_SEARCH
        result = self.monitor.process_lines_for_json(search_lines, "media")
        # SearchNode output should be filtered and not captured
        assert len(result) == 0
    
    def test_filter_search_node_output_single_line(self):
        """Test filtering SearchNode single-line JSON output"""
        # SearchNode single-line JSON format
        search_line = test_data.SEARCH_NODE_REFLECTION_SEARCH
        result = self.monitor.process_lines_for_json([search_line], "media")
        # SearchNode output should be filtered
        assert len(result) == 0
    
    def test_search_node_vs_summary_node_mixed(self):
        """Test mixed scenario: SearchNode and SummaryNode coexist, only capture SummaryNode"""
        lines = [
            # SearchNode output (should be filtered)
            "[11:16:35] 2025-11-06 11:16:35.567 | INFO | InsightEngine.nodes.search_node:process_output:97 - Cleaned output: {",
            "[11:16:35] \"search_query\": \"Test query\"",
            "[11:16:35] }",
            # SummaryNode output (should be captured)
            "[11:17:05] 2025-11-06 11:17:05.547 | INFO | InsightEngine.nodes.summary_node:process_output:131 - Cleaned output: {",
            "[11:17:05] \"paragraph_latest_state\": \"This is summary content\"",
            "[11:17:05] }",
        ]
        result = self.monitor.process_lines_for_json(lines, "media")
        # Should only capture SummaryNode output, not SearchNode output
        assert len(result) > 0
        assert any("summary content" in content for content in result)
        # Ensure search query content is not included
        assert not any("search_query" in content for content in result)
        assert not any("Test query" in content for content in result)
    
    def test_filter_error_logs_from_summary_node(self):
        """Test filtering SummaryNode error logs (Important: error logs should not enter forum)"""
        # JSON parsing failure error log
        assert self.monitor.is_target_log_line(test_data.SUMMARY_NODE_JSON_ERROR) == False
        
        # JSON repair failure error log
        assert self.monitor.is_target_log_line(test_data.SUMMARY_NODE_JSON_FIX_ERROR) == False
        
        # ERROR level log
        assert self.monitor.is_target_log_line(test_data.SUMMARY_NODE_ERROR_LOG) == False
        
        # Traceback error log
        for line in test_data.SUMMARY_NODE_TRACEBACK.split('\n'):
            assert self.monitor.is_target_log_line(line) == False
    
    def test_error_logs_not_captured(self):
        """Test error logs are not captured into forum"""
        error_lines = [
            test_data.SUMMARY_NODE_JSON_ERROR,
            test_data.SUMMARY_NODE_JSON_FIX_ERROR,
            test_data.SUMMARY_NODE_ERROR_LOG,
        ]
        
        for line in error_lines:
            result = self.monitor.process_lines_for_json([line], "media")
            # Error logs should not be captured
            assert len(result) == 0
    
    def test_mixed_valid_and_error_logs(self):
        """Test mixed scenario: valid logs and error logs coexist, only capture valid logs"""
        lines = [
            # Error logs (should be filtered)
            test_data.SUMMARY_NODE_JSON_ERROR,
            test_data.SUMMARY_NODE_JSON_FIX_ERROR,
            # Valid SummaryNode output (should be captured)
            "[11:55:31] 2025-11-06 11:55:31.762 | INFO | MediaEngine.nodes.summary_node:process_output:134 - Cleaned output: {",
            "[11:55:31] \"paragraph_latest_state\": \"This is valid summary content\"",
            "[11:55:31] }",
        ]
        result = self.monitor.process_lines_for_json(lines, "media")
        # Should only capture valid logs, not error logs
        assert len(result) > 0
        assert any("valid summary content" in content for content in result)
        # Ensure error information is not included
        assert not any("JSON parsing failed" in content for content in result)
        assert not any("JSON repair failed" in content for content in result)


def run_tests():
    """Run all tests"""
    import pytest
    
    # Run tests
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()

