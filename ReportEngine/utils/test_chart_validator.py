"""
Test cases for chart validator and repairer.

Run tests:
    python -m pytest ReportEngine/utils/test_chart_validator.py -v
"""

import pytest
from ReportEngine.utils.chart_validator import (
    ChartValidator,
    ChartRepairer,
    ValidationResult,
    RepairResult,
    create_chart_validator,
    create_chart_repairer
)


class TestChartValidator:
    """Test ChartValidator class"""

    def setup_method(self):
        """Initialize before each test"""
        self.validator = create_chart_validator()

    def test_valid_bar_chart(self):
        """Test valid bar chart"""
        widget_block = {
            "type": "widget",
            "widgetType": "chart.js/bar",
            "widgetId": "chart-001",
            "props": {
                "type": "bar",
                "title": "Sales Data"
            },
            "data": {
                "labels": ["Jan", "Feb", "Mar"],
                "datasets": [
                    {
                        "label": "Sales",
                        "data": [100, 200, 150]
                    }
                ]
            }
        }

        result = self.validator.validate(widget_block)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_valid_line_chart(self):
        """Test valid line chart"""
        widget_block = {
            "type": "widget",
            "widgetType": "chart.js/line",
            "widgetId": "chart-002",
            "props": {
                "type": "line"
            },
            "data": {
                "labels": ["Mon", "Tue", "Wed"],
                "datasets": [
                    {
                        "label": "Visits",
                        "data": [50, 75, 60]
                    }
                ]
            }
        }

        result = self.validator.validate(widget_block)
        assert result.is_valid

    def test_valid_pie_chart(self):
        """Test valid pie chart"""
        widget_block = {
            "widgetType": "chart.js/pie",
            "props": {"type": "pie"},
            "data": {
                "labels": ["A", "B", "C"],
                "datasets": [
                    {
                        "data": [30, 40, 30]
                    }
                ]
            }
        }

        result = self.validator.validate(widget_block)
        assert result.is_valid

    def test_missing_widgetType(self):
        """Test missing widgetType"""
        widget_block = {
            "props": {},
            "data": {}
        }

        result = self.validator.validate(widget_block)
        assert not result.is_valid
        assert "widgetType" in result.errors[0]

    def test_missing_data_field(self):
        """Test missing data field"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"}
        }

        result = self.validator.validate(widget_block)
        assert not result.is_valid
        assert "data" in result.errors[0]

    def test_missing_datasets(self):
        """Test missing datasets"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"]
            }
        }

        result = self.validator.validate(widget_block)
        assert not result.is_valid
        assert "datasets" in result.errors[0]

    def test_empty_datasets(self):
        """Test empty datasets"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"],
                "datasets": []
            }
        }

        result = self.validator.validate(widget_block)
        assert not result.is_valid
        assert "empty" in result.errors[0].lower() or "空" in result.errors[0]

    def test_missing_labels_for_bar_chart(self):
        """Test bar chart missing labels"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": [10, 20, 30]
                    }
                ]
            }
        }

        result = self.validator.validate(widget_block)
        assert not result.is_valid
        assert "labels" in result.errors[0]

    def test_invalid_data_type(self):
        """Test invalid data type"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"],
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": ["abc", "def"]  # should be numeric
                    }
                ]
            }
        }

        result = self.validator.validate(widget_block)
        assert not result.is_valid
        assert "numeric" in result.errors[0].lower() or "数值" in result.errors[0]

    def test_data_length_mismatch_warning(self):
        """Test data length mismatch (warning)"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B", "C"],
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": [10, 20]  # length mismatch
                    }
                ]
            }
        }

        result = self.validator.validate(widget_block)
        # Length mismatch is a warning, not an error
        assert len(result.warnings) > 0
        assert "match" in result.warnings[0].lower() or "不匹配" in result.warnings[0]

    def test_scatter_chart(self):
        """Test scatter chart (special data format)"""
        widget_block = {
            "widgetType": "chart.js/scatter",
            "props": {"type": "scatter"},
            "data": {
                "datasets": [
                    {
                        "label": "Data Points",
                        "data": [
                            {"x": 10, "y": 20},
                            {"x": 15, "y": 25}
                        ]
                    }
                ]
            }
        }

        result = self.validator.validate(widget_block)
        assert result.is_valid

    def test_non_chart_widget(self):
        """Test non-chart type widget (should skip validation)"""
        widget_block = {
            "widgetType": "custom/widget",
            "props": {},
            "data": {}
        }

        result = self.validator.validate(widget_block)
        # Non-chart.js type, skip validation, return valid
        assert result.is_valid


class TestChartRepairer:
    """Test ChartRepairer class"""

    def setup_method(self):
        """Initialize before each test"""
        self.validator = create_chart_validator()
        self.repairer = create_chart_repairer(validator=self.validator)

    def test_repair_missing_props(self):
        """Test repair missing props field"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "data": {
                "labels": ["A", "B"],
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": [10, 20]
                    }
                ]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        assert "props" in result.repaired_block
        assert result.method == "local"

    def test_repair_missing_chart_type(self):
        """Test repair missing chart type"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {},
            "data": {
                "labels": ["A", "B"],
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": [10, 20]
                    }
                ]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        assert result.repaired_block["props"]["type"] == "bar"
        assert "chart type" in str(result.changes).lower() or "图表类型" in str(result.changes)

    def test_repair_missing_datasets(self):
        """Test repair missing datasets"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        assert "datasets" in result.repaired_block["data"]
        assert isinstance(result.repaired_block["data"]["datasets"], list)

    def test_repair_missing_labels(self):
        """Test repair missing labels"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": [10, 20, 30]
                    }
                ]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        assert "labels" in result.repaired_block["data"]
        assert len(result.repaired_block["data"]["labels"]) == 3

    def test_repair_data_length_mismatch(self):
        """Test repair data length mismatch"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B", "C", "D"],
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": [10, 20]  # insufficient length
                    }
                ]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        # Should pad to 4 elements
        assert len(result.repaired_block["data"]["datasets"][0]["data"]) == 4

    def test_repair_string_to_number(self):
        """Test repair string type numeric values"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"],
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": ["10", "20"]  # string numerics
                    }
                ]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        # Should convert to numeric
        assert isinstance(result.repaired_block["data"]["datasets"][0]["data"][0], float)

    def test_repair_construct_datasets_from_values(self):
        """Test construct datasets from values field"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"],
                "values": [10, 20]  # using values instead of datasets
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        assert "datasets" in result.repaired_block["data"]
        assert len(result.repaired_block["data"]["datasets"]) > 0

    def test_no_repair_needed(self):
        """Test case where no repair is needed"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"],
                "datasets": [
                    {
                        "label": "Series 1",
                        "data": [10, 20]
                    }
                ]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        assert result.method == "none"
        assert len(result.changes) == 0

    def test_repair_adds_default_label(self):
        """Test repair adds default label"""
        widget_block = {
            "widgetType": "chart.js/bar",
            "props": {"type": "bar"},
            "data": {
                "labels": ["A", "B"],
                "datasets": [
                    {
                        # missing label
                        "data": [10, 20]
                    }
                ]
            }
        }

        result = self.repairer.repair(widget_block)
        assert result.success
        assert "label" in result.repaired_block["data"]["datasets"][0]


class TestValidatorIntegration:
    """Integration tests"""

    def test_full_validation_and_repair_workflow(self):
        """Test complete validation and repair workflow"""
        validator = create_chart_validator()
        repairer = create_chart_repairer(validator=validator)

        # A chart with multiple issues
        widget_block = {
            "widgetType": "chart.js/bar",
            "data": {
                "datasets": [
                    {
                        "data": ["10", "20", "30"]  # string numerics
                    }
                ]
            }
        }

        # 1. Validate (should fail)
        validation = validator.validate(widget_block)
        assert not validation.is_valid

        # 2. Repair
        repair_result = repairer.repair(widget_block, validation)
        assert repair_result.success

        # 3. Validate again (should pass)
        final_validation = validator.validate(repair_result.repaired_block)
        assert final_validation.is_valid


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
