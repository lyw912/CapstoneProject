"""
Chart validation and repair utilities.

Provides validation and repair capabilities for Chart.js chart data:
1. Validate chart data format against Chart.js requirements
2. Local rule-based repair for common issues
3. LLM API-assisted repair for complex issues
4. Follows the principle of "better not to change than to change incorrectly"

Supported chart types:
- line (line chart)
- bar (bar chart)
- pie (pie chart)
- doughnut (doughnut chart)
- radar (radar chart)
- polararea (polar area chart)
- scatter (scatter chart)
- bubble (bubble chart)
- horizontalbar (horizontal bar chart)
"""

from __future__ import annotations

import copy
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def has_critical_errors(self) -> bool:
        """Check if there are critical errors (will cause rendering failure)"""
        return not self.is_valid and len(self.errors) > 0


@dataclass
class RepairResult:
    """Repair result"""
    success: bool
    repaired_block: Optional[Dict[str, Any]]
    method: str  # 'none', 'local', 'api'
    changes: List[str]

    def has_changes(self) -> bool:
        """Check if there are any modifications"""
        return len(self.changes) > 0


class ChartValidator:
    """
    Chart Validator - Validates Chart.js chart data format.

    Validation rules:
    1. Basic structure validation: widgetType, props, data fields
    2. Chart type validation: supported chart types
    3. Data format validation: labels and datasets structure
    4. Data consistency validation: labels and datasets length match
    5. Numeric type validation: correct data value types
    """

    # Supported chart types
    SUPPORTED_CHART_TYPES = {
        'line', 'bar', 'pie', 'doughnut', 'radar', 'polararea', 'scatter',
        'bubble', 'horizontalbar'
    }

    # Chart types that require labels
    LABEL_REQUIRED_TYPES = {
        'line', 'bar', 'radar', 'polararea', 'pie', 'doughnut'
    }

    # Chart types that require numeric data
    NUMERIC_DATA_TYPES = {
        'line', 'bar', 'radar', 'polararea', 'pie', 'doughnut'
    }

    # Chart types that require special data format
    SPECIAL_DATA_TYPES = {
        'scatter': {'x', 'y'},
        'bubble': {'x', 'y', 'r'}
    }

    def __init__(self):
        """Initialize validator and reserve cache structure for reuse of validation/repair results"""

    def validate(self, widget_block: Dict[str, Any]) -> ValidationResult:
        """
        Validate chart format.

        Args:
            widget_block: widget-type block containing widgetId/widgetType/props/data

        Returns:
            ValidationResult: Validation result
        """
        errors = []
        warnings = []

        # 1. Basic structure validation
        if not isinstance(widget_block, dict):
            errors.append("widget_block must be a dictionary")
            return ValidationResult(False, errors, warnings)

        # 2. Check widgetType
        widget_type = widget_block.get('widgetType', '')
        if not widget_type or not isinstance(widget_type, str):
            errors.append("Missing widgetType field or incorrect type")
            return ValidationResult(False, errors, warnings)

        # Check if it's a chart.js type
        if not widget_type.startswith('chart.js'):
            # Not a chart type, skip validation
            return ValidationResult(True, errors, warnings)

        # 3. Extract chart type
        chart_type = self._extract_chart_type(widget_block)
        if not chart_type:
            errors.append("Unable to determine chart type")
            return ValidationResult(False, errors, warnings)

        # 4. Check if chart type is supported
        if chart_type not in self.SUPPORTED_CHART_TYPES:
            warnings.append(f"Chart type '{chart_type}' may not be supported, will attempt degraded rendering")

        # 5. Validate data structure
        data = widget_block.get('data')
        if not isinstance(data, dict):
            errors.append("data field must be a dictionary")
            return ValidationResult(False, errors, warnings)

        # Detect if {x, y} format data points are used (typically for time axis/scatter)
        def contains_object_points(ds_list: List[Any] | None) -> bool:
            """Check if dataset contains object points represented by x/y keys, for switching validation branches"""
            if not isinstance(ds_list, list):
                return False
            for point in ds_list:
                if isinstance(point, dict) and any(key in point for key in ('x', 'y', 't')):
                    return True
            return False

        datasets_for_detection = data.get('datasets') or []
        uses_object_points = any(
            isinstance(ds, dict) and contains_object_points(ds.get('data'))
            for ds in datasets_for_detection
        )

        # 6. Validate data based on chart type
        if chart_type in self.SPECIAL_DATA_TYPES:
            # Special data format (scatter, bubble)
            self._validate_special_data(data, chart_type, errors, warnings)
        else:
            # Standard data format (labels + datasets)
            self._validate_standard_data(data, chart_type, errors, warnings, uses_object_points)

        # 7. Validate props
        props = widget_block.get('props')
        if props is not None and not isinstance(props, dict):
            warnings.append("props field should be a dictionary")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings)

    def _extract_chart_type(self, widget_block: Dict[str, Any]) -> Optional[str]:
        """
        Extract chart type.

        Priority:
        1. props.type
        2. Type in widgetType (chart.js/bar -> bar)
        3. data.type
        """
        # 1. Get from props
        props = widget_block.get('props') or {}
        if isinstance(props, dict):
            chart_type = props.get('type')
            if chart_type and isinstance(chart_type, str):
                return chart_type.lower()

        # 2. Extract from widgetType
        widget_type = widget_block.get('widgetType', '')
        if '/' in widget_type:
            chart_type = widget_type.split('/')[-1]
            if chart_type:
                return chart_type.lower()

        # 3. Get from data
        data = widget_block.get('data') or {}
        if isinstance(data, dict):
            chart_type = data.get('type')
            if chart_type and isinstance(chart_type, str):
                return chart_type.lower()

        return None

    def _validate_standard_data(
        self,
        data: Dict[str, Any],
        chart_type: str,
        errors: List[str],
        warnings: List[str],
        uses_object_points: bool = False
    ):
        """Validate standard data format (labels + datasets)"""
        labels = data.get('labels')
        datasets = data.get('datasets')

        # Validate labels
        if chart_type in self.LABEL_REQUIRED_TYPES:
            if not labels:
                if uses_object_points:
                    warnings.append(
                        f"{chart_type} chart missing labels, will render based on data points (using x values)"
                    )
                else:
                    errors.append(f"{chart_type} chart must include labels field")
            elif not isinstance(labels, list):
                errors.append("labels must be an array")
            elif len(labels) == 0:
                warnings.append("labels array is empty, chart may not display properly")

        # Validate datasets
        if datasets is None:
            errors.append("Missing datasets field")
            return

        if not isinstance(datasets, list):
            errors.append("datasets must be an array")
            return

        if len(datasets) == 0:
            errors.append("datasets array is empty")
            return

        # Validate each dataset
        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                errors.append(f"datasets[{idx}] must be an object")
                continue

            # Validate data field
            ds_data = dataset.get('data')
            if ds_data is None:
                errors.append(f"datasets[{idx}] missing data field")
                continue

            if not isinstance(ds_data, list):
                errors.append(f"datasets[{idx}].data must be an array")
                continue

            if len(ds_data) == 0:
                warnings.append(f"datasets[{idx}].data array is empty")
                continue

            # If using {x, y} object format data points, skip labels length and numeric validation
            object_points = any(
                isinstance(value, dict) and any(key in value for key in ('x', 'y', 't'))
                for value in ds_data
            )

            # Validate data length consistency
            if labels and isinstance(labels, list) and not object_points:
                if len(ds_data) != len(labels):
                    warnings.append(
                        f"datasets[{idx}].data length({len(ds_data)}) doesn't match labels length({len(labels)})"
                    )

            # Validate numeric types
            if chart_type in self.NUMERIC_DATA_TYPES and not object_points:
                for data_idx, value in enumerate(ds_data):
                    if value is not None and not isinstance(value, (int, float)):
                        errors.append(
                            f"datasets[{idx}].data[{data_idx}] value '{value}' is not a valid numeric type"
                        )
                        break  # Only report the first error

    def _validate_special_data(
        self,
        data: Dict[str, Any],
        chart_type: str,
        errors: List[str],
        warnings: List[str]
    ):
        """Validate special data format (scatter, bubble)"""
        datasets = data.get('datasets')

        if not datasets:
            errors.append("Missing datasets field")
            return

        if not isinstance(datasets, list):
            errors.append("datasets must be an array")
            return

        if len(datasets) == 0:
            errors.append("datasets array is empty")
            return

        required_keys = self.SPECIAL_DATA_TYPES.get(chart_type, set())

        # Validate each dataset
        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                errors.append(f"datasets[{idx}] must be an object")
                continue

            ds_data = dataset.get('data')
            if ds_data is None:
                errors.append(f"datasets[{idx}] missing data field")
                continue

            if not isinstance(ds_data, list):
                errors.append(f"datasets[{idx}].data must be an array")
                continue

            if len(ds_data) == 0:
                warnings.append(f"datasets[{idx}].data array is empty")
                continue

            # Validate data point format
            for data_idx, point in enumerate(ds_data):
                if not isinstance(point, dict):
                    errors.append(
                        f"datasets[{idx}].data[{data_idx}] must be an object (containing {required_keys} fields)"
                    )
                    break

                # Check required keys
                missing_keys = required_keys - set(point.keys())
                if missing_keys:
                    errors.append(
                        f"datasets[{idx}].data[{data_idx}] missing required fields: {missing_keys}"
                    )
                    break

                # Validate numeric types
                for key in required_keys:
                    value = point.get(key)
                    if value is not None and not isinstance(value, (int, float)):
                        errors.append(
                            f"datasets[{idx}].data[{data_idx}].{key} value '{value}' is not a valid numeric type"
                        )
                        break

    def can_render(self, widget_block: Dict[str, Any]) -> bool:
        """
        Check if chart can render properly (quick check).

        Args:
            widget_block: widget-type block

        Returns:
            bool: Whether it can render properly
        """
        result = self.validate(widget_block)
        return result.is_valid


class ChartRepairer:
    """
    Chart Repairer - Attempts to repair chart data.

    Repair strategies:
    1. Local rule-based repair: fix common issues
    2. API repair: use LLM for complex issues
    3. Validate repair results: ensure repaired data can render properly
    """

    def __init__(
        self,
        validator: ChartValidator,
        llm_repair_fns: Optional[List[Callable]] = None
    ):
        """
        Initialize repairer.

        Args:
            validator: Chart validator instance
            llm_repair_fns: List of LLM repair functions (corresponding to 4 Engines)
        """
        self.validator = validator
        self.llm_repair_fns = llm_repair_fns or []
        # Cache repair results to avoid duplicate LLM calls for the same chart
        self._result_cache: Dict[str, RepairResult] = {}

    def build_cache_key(self, widget_block: Dict[str, Any]) -> str:
        """
        Generate a stable cache key for the chart to prevent duplicate repairs for the same data.

        - Prioritize using widgetId;
        - Combine with data content hash to avoid reusing old results when ID is same but content changes.
        """
        widget_id = ""
        if isinstance(widget_block, dict):
            widget_id = widget_block.get('widgetId') or widget_block.get('id') or ""
        try:
            serialized = json.dumps(
                widget_block,
                ensure_ascii=False,
                sort_keys=True,
                default=str
            )
        except Exception:
            serialized = repr(widget_block)
        digest = hashlib.md5(serialized.encode('utf-8', errors='ignore')).hexdigest()
        return f"{widget_id}:{digest}"

    def repair(
        self,
        widget_block: Dict[str, Any],
        validation_result: Optional[ValidationResult] = None
    ) -> RepairResult:
        """
        Attempt to repair chart data.

        Args:
            widget_block: widget-type block
            validation_result: Validation result (optional, will validate first if not provided)

        Returns:
            RepairResult: Repair result
        """
        cache_key = self.build_cache_key(widget_block)

        cached = self._result_cache.get(cache_key)
        if cached:
            # Return deep copy of cached result to avoid external modifications affecting cache
            return copy.deepcopy(cached)

        def _cache_and_return(res: RepairResult) -> RepairResult:
            """Write repair result to cache and return, avoiding duplicate calls to downstream repair logic"""
            try:
                self._result_cache[cache_key] = copy.deepcopy(res)
            except Exception:
                self._result_cache[cache_key] = res
            return res

        # 1. Validate first if no validation result provided
        if validation_result is None:
            validation_result = self.validator.validate(widget_block)

        # Track the current latest validation result and data
        current_validation = validation_result
        current_block = widget_block

        # 2. Try local repair (even if validation passed, there may be warnings)
        logger.info(f"Attempting local chart repair")
        local_result = self.repair_locally(widget_block, validation_result)

        # 3. Validate local repair result
        if local_result.has_changes():
            repaired_validation = self.validator.validate(local_result.repaired_block)
            if repaired_validation.is_valid:
                logger.info(f"Local repair successful: {local_result.changes}")
                return _cache_and_return(
                    RepairResult(True, local_result.repaired_block, 'local', local_result.changes)
                )
            else:
                logger.warning(f"Still invalid after local repair: {repaired_validation.errors}")
                # Update current state to locally repaired result for API repair
                current_validation = repaired_validation
                current_block = local_result.repaired_block

        # 4. If still has critical errors, try API repair
        # Note: use current_validation instead of original validation_result
        if current_validation.has_critical_errors() and len(self.llm_repair_fns) > 0:
            logger.info("Local repair failed or insufficient, trying API repair")
            # Pass locally repaired data (if any) to avoid wasting local repair work
            api_result = self.repair_with_api(current_block, current_validation)

            if api_result.success:
                # Validate repair result
                api_repaired_validation = self.validator.validate(api_result.repaired_block)
                if api_repaired_validation.is_valid:
                    logger.info(f"API repair successful: {api_result.changes}")
                    return _cache_and_return(api_result)
                else:
                    logger.warning(f"Still invalid after API repair: {api_repaired_validation.errors}")

        # 5. If original validation passed, return original or repaired data
        if validation_result.is_valid:
            if local_result.has_changes():
                return _cache_and_return(
                    RepairResult(True, local_result.repaired_block, 'local', local_result.changes)
                )
            else:
                return _cache_and_return(RepairResult(True, widget_block, 'none', []))

        # 6. All repairs failed, return original data (or locally partially repaired data)
        logger.warning("All repair attempts failed, keeping original data")
        # If local has partial repairs, return locally repaired data (may be better than original even if still invalid)
        final_block = local_result.repaired_block if local_result.has_changes() else widget_block
        return _cache_and_return(RepairResult(False, final_block, 'none', []))

    def repair_locally(
        self,
        widget_block: Dict[str, Any],
        validation_result: ValidationResult
    ) -> RepairResult:
        """
        Use local rule-based repair.

        Repair rules:
        1. Fill in missing basic fields
        2. Fix data type errors
        3. Fix data length mismatches
        4. Clean invalid data
        5. Add default values
        """
        repaired = copy.deepcopy(widget_block)
        changes = []

        # 1. Ensure basic structure exists
        if 'props' not in repaired or not isinstance(repaired.get('props'), dict):
            repaired['props'] = {}
            changes.append("Added missing props field")

        if 'data' not in repaired or not isinstance(repaired.get('data'), dict):
            repaired['data'] = {}
            changes.append("Added missing data field")

        # 2. Ensure chart type exists
        chart_type = self.validator._extract_chart_type(repaired)
        props = repaired['props']

        if not chart_type:
            # Try to infer from widgetType
            widget_type = repaired.get('widgetType', '')
            if '/' in widget_type:
                chart_type = widget_type.split('/')[-1].lower()
                props['type'] = chart_type
                changes.append(f"Inferred chart type from widgetType: {chart_type}")
            else:
                # Default to bar type
                chart_type = 'bar'
                props['type'] = chart_type
                changes.append("Set default chart type: bar")
        elif 'type' not in props or not props['type']:
            # chart_type exists but no type field in props, need to add
            props['type'] = chart_type
            changes.append(f"Added inferred chart type to props: {chart_type}")

        # 3. Fix data structure
        data = repaired['data']

        # Ensure datasets exists
        if 'datasets' not in data or not isinstance(data.get('datasets'), list):
            data['datasets'] = []
            changes.append("Added missing datasets field")

        # If datasets is empty but data has other fields, try to construct datasets
        if len(data['datasets']) == 0:
            constructed = self._try_construct_datasets(data, chart_type)
            if constructed:
                data['datasets'] = constructed
                changes.append("Constructed datasets from data")
            elif 'labels' in data and isinstance(data.get('labels'), list) and len(data['labels']) > 0:
                # If labels exist but no data, create an empty dataset
                data['datasets'] = [{
                    'label': 'Data',
                    'data': [0] * len(data['labels'])
                }]
                changes.append("Created default dataset from labels (using zero values)")

        # Ensure labels exists (if required)
        if chart_type in ChartValidator.LABEL_REQUIRED_TYPES:
            if 'labels' not in data or not isinstance(data.get('labels'), list):
                # Try to generate labels based on datasets length
                if data['datasets'] and len(data['datasets']) > 0:
                    first_ds = data['datasets'][0]
                    if isinstance(first_ds, dict) and isinstance(first_ds.get('data'), list):
                        data_len = len(first_ds['data'])
                        data['labels'] = [f"Item {i+1}" for i in range(data_len)]
                        changes.append(f"Generated {data_len} default labels")

        # 4. Fix data in datasets
        for idx, dataset in enumerate(data.get('datasets', [])):
            if not isinstance(dataset, dict):
                continue

            # Ensure data field exists
            if 'data' not in dataset or not isinstance(dataset.get('data'), list):
                dataset['data'] = []
                changes.append(f"Added empty data array to datasets[{idx}]")

            # Ensure label exists
            if 'label' not in dataset:
                dataset['label'] = f"Series {idx + 1}"
                changes.append(f"Added default label to datasets[{idx}]")

            # Fix data length mismatch
            labels = data.get('labels', [])
            ds_data = dataset.get('data', [])
            if isinstance(labels, list) and isinstance(ds_data, list):
                if len(ds_data) < len(labels):
                    # Data insufficient, pad with null
                    dataset['data'] = ds_data + [None] * (len(labels) - len(ds_data))
                    changes.append(f"datasets[{idx}] data length insufficient, padded with null")
                elif len(ds_data) > len(labels):
                    # Data too long, truncate
                    dataset['data'] = ds_data[:len(labels)]
                    changes.append(f"datasets[{idx}] data length too long, truncated")

            # Convert non-numeric data to numeric (if possible)
            if chart_type in ChartValidator.NUMERIC_DATA_TYPES:
                ds_data = dataset.get('data', [])
                converted = False
                for i, value in enumerate(ds_data):
                    if value is None:
                        continue
                    if not isinstance(value, (int, float)):
                        # Try to convert
                        try:
                            if isinstance(value, str):
                                # Try to convert string
                                ds_data[i] = float(value)
                                converted = True
                        except (ValueError, TypeError):
                            # Conversion failed, set to null
                            ds_data[i] = None
                            converted = True
                if converted:
                    changes.append(f"datasets[{idx}] contained non-numeric data, attempted conversion")

        # 5. Validate repair result
        success = len(changes) > 0

        return RepairResult(success, repaired, 'local', changes)

    def _try_construct_datasets(
        self,
        data: Dict[str, Any],
        chart_type: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Try to construct datasets from data"""
        # If data directly contains value array, try to construct
        if 'values' in data and isinstance(data['values'], list):
            return [{
                'label': 'Data',
                'data': data['values']
            }]

        # If data contains series field
        if 'series' in data and isinstance(data['series'], list):
            datasets = []
            for idx, series in enumerate(data['series']):
                if isinstance(series, dict):
                    datasets.append({
                        'label': series.get('name', f'Series {idx + 1}'),
                        'data': series.get('data', [])
                    })
                elif isinstance(series, list):
                    datasets.append({
                        'label': f'Series {idx + 1}',
                        'data': series
                    })
            if datasets:
                return datasets

        return None

    def repair_with_api(
        self,
        widget_block: Dict[str, Any],
        validation_result: ValidationResult
    ) -> RepairResult:
        """
        Use API repair (calling LLM from 4 Engines).

        Strategy: Try different Engines in sequence until repair succeeds
        """
        if not self.llm_repair_fns:
            logger.debug("No LLM repair functions available, skipping API repair")
            return RepairResult(False, None, 'api', [])

        widget_id = widget_block.get('widgetId', 'unknown')
        logger.info(f"Chart {widget_id} starting API repair, {len(self.llm_repair_fns)} Engine(s) available")

        for idx, repair_fn in enumerate(self.llm_repair_fns):
            try:
                logger.info(f"Attempting to repair chart {widget_id} with Engine {idx + 1}/{len(self.llm_repair_fns)}")
                repaired = repair_fn(widget_block, validation_result.errors)

                if repaired and isinstance(repaired, dict):
                    # Validate repair result
                    repaired_validation = self.validator.validate(repaired)
                    if repaired_validation.is_valid:
                        logger.info(f"Chart {widget_id} repaired successfully with Engine {idx + 1}")
                        return RepairResult(
                            True,
                            repaired,
                            'api',
                            [f"Repair successful with Engine {idx + 1}"]
                        )
                    else:
                        logger.warning(
                            f"Chart {widget_id} Engine {idx + 1} returned data that failed validation: "
                            f"{repaired_validation.errors}"
                        )
                else:
                    logger.warning(f"Chart {widget_id} Engine {idx + 1} returned empty or invalid response")
            except Exception as e:
                # Use exception to log full stack trace
                logger.exception(f"Chart {widget_id} Engine {idx + 1} encountered exception during repair: {e}")
                continue

        logger.warning(f"Chart {widget_id} all {len(self.llm_repair_fns)} Engine(s) failed to repair")
        return RepairResult(False, None, 'api', [])


def create_chart_validator() -> ChartValidator:
    """Create chart validator instance"""
    return ChartValidator()


def create_chart_repairer(
    validator: Optional[ChartValidator] = None,
    llm_repair_fns: Optional[List[Callable]] = None
) -> ChartRepairer:
    """Create chart repairer instance"""
    if validator is None:
        validator = create_chart_validator()
    return ChartRepairer(validator, llm_repair_fns)
