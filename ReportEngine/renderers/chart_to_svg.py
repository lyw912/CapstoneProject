"""
Chart to SVG Converter - Converts Chart.js data to vector SVG graphics

Supported chart types:
- line: Line chart
- bar: Bar chart
- pie: Pie chart
- doughnut: Doughnut chart
- radar: Radar chart
- polarArea: Polar area chart
- scatter: Scatter chart
"""

from __future__ import annotations

import base64
import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.font_manager as fm
    from matplotlib.patches import Wedge, Rectangle
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not installed, PDF chart vector rendering will be unavailable")

# Optional dependency: scipy for curve smoothing
try:
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.info("Scipy not installed, line charts will not support curve smoothing (basic rendering unaffected)")


class ChartToSVGConverter:
    """
    Converts Chart.js chart data to SVG vector graphics
    """

    # Default color palette (optimized: bright and distinguishable)
    DEFAULT_COLORS = [
        '#4A90E2', '#E85D75', '#50C878', '#FFB347',  # 明亮蓝、珊瑚红、翠绿、橙黄
        '#9B59B6', '#3498DB', '#E67E22', '#16A085',  # 紫色、天蓝、橙色、青色
        '#F39C12', '#D35400', '#27AE60', '#8E44AD'   # 金色、深橙、绿色、紫罗兰
    ]

    # CSS variable to color mapping table (optimized: using brighter, lighter colors)
    CSS_VAR_COLOR_MAP = {
        'var(--color-accent)': '#4A90E2',        # 明亮蓝色（从#007AFF改为更浅）
        'var(--re-accent-color)': '#4A90E2',     # 明亮蓝色
        'var(--re-accent-color-translucent)': (0.29, 0.565, 0.886, 0.08),  # 蓝色极浅透明 rgba(74, 144, 226, 0.08)
        'var(--color-kpi-down)': '#E85D75',      # 珊瑚红色（从#DC3545改为更柔和）
        'var(--re-danger-color)': '#E85D75',     # 珊瑚红色
        'var(--re-danger-color-translucent)': (0.91, 0.365, 0.459, 0.08),  # 红色极浅透明 rgba(232, 93, 117, 0.08)
        'var(--color-warning)': '#FFB347',       # 柔和橙黄色（从#FFC107改为更浅）
        'var(--re-warning-color)': '#FFB347',    # 柔和橙黄色
        'var(--re-warning-color-translucent)': (1.0, 0.702, 0.278, 0.08),  # 黄色极浅透明 rgba(255, 179, 71, 0.08)
        'var(--color-success)': '#50C878',       # 翠绿色（从#28A745改为更明亮）
        'var(--re-success-color)': '#50C878',    # 翠绿色
        'var(--re-success-color-translucent)': (0.314, 0.784, 0.471, 0.08),  # 绿色极浅透明 rgba(80, 200, 120, 0.08)
        'var(--color-accent-positive)': '#50C878',
        'var(--color-accent-negative)': '#E85D75',
        'var(--color-text-secondary)': '#6B7280',
        'var(--accentPositive)': '#50C878',
        'var(--accentNegative)': '#E85D75',
        'var(--sentiment-positive, #28A745)': '#28A745',
        'var(--sentiment-negative, #E53E3E)': '#E53E3E',
        'var(--sentiment-neutral, #FFC107)': '#FFC107',
        'var(--sentiment-positive)': '#28A745',
        'var(--sentiment-negative)': '#E53E3E',
        'var(--sentiment-neutral)': '#FFC107',
        'var(--color-primary)': '#3498DB',       # 天蓝色
        'var(--color-secondary)': '#95A5A6',     # 浅灰色
    }

    # Fallback mapping for parsing rgba(var(--color-primary-rgb), 0.5) format
    CSS_VAR_RGB_MAP = {
        'color-primary-rgb': (52, 152, 219),
        'color-tone-up-rgb': (80, 200, 120),
        'color-tone-down-rgb': (232, 93, 117),
        'color-accent-positive-rgb': (80, 200, 120),
        'color-accent-neutral-rgb': (149, 165, 166),
    }

    def __init__(self, font_path: Optional[str] = None):
        """
        Initialize converter

        Parameters:
            font_path: Chinese font path (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not installed, please run: pip install matplotlib")

        self.font_path = font_path
        self._setup_chinese_font()

    def _setup_chinese_font(self):
        """Configure Chinese font"""
        if self.font_path:
            try:
                # Add custom font
                fm.fontManager.addfont(self.font_path)
                # Set default font
                font_prop = fm.FontProperties(fname=self.font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
                logger.info(f"Chinese font loaded: {self.font_path}")
            except Exception as e:
                logger.warning(f"Failed to load Chinese font: {e}, will use system default font")
        else:
            # Try to use system Chinese fonts
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as e:
                logger.warning(f"Failed to configure Chinese font: {e}")

    def convert_widget_to_svg(
        self,
        widget_data: Dict[str, Any],
        width: int = 800,
        height: int = 500,
        dpi: int = 100
    ) -> Optional[str]:
        """
        Convert widget data to SVG string

        Parameters:
            widget_data: Widget block data (contains widgetType, data, props)
            width: Chart width (pixels)
            height: Chart height (pixels)
            dpi: DPI setting

        Returns:
            str: SVG string, returns None on failure
        """
        try:
            # Extract chart type
            widget_type = widget_data.get('widgetType', '')
            if not widget_type or not widget_type.startswith('chart.js'):
                logger.warning(f"Unsupported widget type: {widget_type}")
                return None

            # Extract chart type from widgetType, e.g., "chart.js/line" -> "line"
            chart_type = widget_type.split('/')[-1] if '/' in widget_type else 'bar'

            # Also check type in props
            props = widget_data.get('props', {})
            if props.get('type'):
                chart_type = props['type']

            # Chart.js v4 removed horizontalBar type, automatically downgrade to bar with horizontal axis
            horizontal_bar = False
            if chart_type and str(chart_type).lower() == 'horizontalbar':
                chart_type = 'bar'
                horizontal_bar = True

            # Support forced horizontal bar chart via indexAxis: 'y'
            if isinstance(props, dict):
                options = props.get('options') or {}
                index_axis = (options.get('indexAxis') or props.get('indexAxis') or '').lower()
                if index_axis == 'y':
                    horizontal_bar = True

            # Extract data
            data = widget_data.get('data', {})
            if not data:
                logger.warning("Chart data is empty")
                return None

            # Dispatch rendering method based on chart type
            if 'wordcloud' in str(chart_type).lower():
                # Word cloud is handled by dedicated rendering logic, skip SVG conversion here to avoid warnings
                logger.debug("Detected word cloud chart, skipping chart_to_svg conversion")
                return None

            # Dispatch rendering method, special handling for horizontal bar charts
            if chart_type == 'bar':
                return self._render_bar(data, props, width, height, dpi, horizontal=horizontal_bar)
            elif chart_type == 'bubble':
                return self._render_bubble(data, props, width, height, dpi)
            else:
                render_method = getattr(self, f'_render_{chart_type}', None)
                if not render_method:
                    logger.warning(f"Unsupported chart type: {chart_type}")
                    return None

            # Create chart and convert to SVG
            return render_method(data, props, width, height, dpi)

        except Exception as e:
            logger.error(f"Failed to convert chart to SVG: {e}", exc_info=True)
            return None

    def _create_figure(
        self,
        width: int,
        height: int,
        dpi: int,
        title: Optional[str] = None
    ) -> Tuple[Any, Any]:
        """
        Create matplotlib figure

        Returns:
            tuple: (fig, ax)
        """
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        return fig, ax

    def _parse_color(self, color: Any) -> Any:
        """
        Parse color value, convert CSS format to matplotlib-supported format

        Parameters:
            color: Color value (may be CSS format like rgba() or hex or CSS variable)

        Returns:
            Matplotlib-supported color format (hex string or RGB(A) tuple)
        """
        if color is None:
            return None

        # Process numpy arrays, convert to native list
        _np = globals().get("np")
        if _np is not None and hasattr(_np, "ndarray") and isinstance(color, _np.ndarray):
            color = color.tolist()

        # Directly pass colors that are already sequences (like (r,g,b,a)), avoid being converted to strings
        if isinstance(color, (list, tuple)):
            if len(color) in (3, 4) and all(isinstance(c, (int, float)) for c in color):
                normalized = []
                for idx, channel in enumerate(color):
                    # Matplotlib accepts floats between 0-1, if value > 1, normalize from 0-255
                    value = float(channel)
                    if value > 1:
                        value = value / 255.0
                    # Only force clipping for RGB channels, alpha clipped to 0-1
                    if idx < 3:
                        value = max(0.0, min(value, 1.0))
                    else:
                        value = max(0.0, min(value, 1.0))
                    normalized.append(value)
                return tuple(normalized)

            try:
                return tuple(color)
            except Exception:
                return color

        # Other non-string types maintain original string fallback strategy
        if not isinstance(color, str):
            return str(color)

        color = color.strip()

        # Process rgba(var(--color-primary-rgb), 0.5) / rgb(var(--color-primary-rgb))
        var_rgba_pattern = r'rgba?\(var\(--([\w-]+)\)\s*(?:,\s*([\d.]+))?\)'
        match = re.match(var_rgba_pattern, color)
        if match:
            var_name, alpha_str = match.groups()
            rgb_tuple = self.CSS_VAR_RGB_MAP.get(var_name)

            # Compatibility for missing -rgb suffix
            if not rgb_tuple:
                if var_name.endswith('-rgb'):
                    rgb_tuple = self.CSS_VAR_RGB_MAP.get(var_name[:-4])
                else:
                    rgb_tuple = self.CSS_VAR_RGB_MAP.get(f"{var_name}-rgb")

            if rgb_tuple:
                r, g, b = rgb_tuple
                alpha = float(alpha_str) if alpha_str is not None else 1.0
                return (r / 255, g / 255, b / 255, alpha)

        # [Enhanced] Process CSS variables, e.g., var(--color-accent)
        # Use predefined color mapping table to replace CSS variables, ensure different variables have different colors
        if color.startswith('var('):
            # Parse var(--token, fallback) format
            fb_match = re.match(r'^var\(\s*--[^,)+]+,\s*([^)]+)\)', color)
            if fb_match:
                fb_raw = fb_match.group(1).strip()
                fb_color = self._parse_color(fb_raw)
                if fb_color:
                    return fb_color
            # Try to find corresponding color from mapping table
            mapped_color = self.CSS_VAR_COLOR_MAP.get(color)
            if mapped_color:
                return mapped_color
            # If not in mapping table, try to infer color type from variable name
            if 'accent' in color or 'primary' in color:
                return '#007AFF'  # 蓝色
            elif 'danger' in color or 'down' in color or 'error' in color:
                return '#DC3545'  # 红色
            elif 'warning' in color:
                return '#FFC107'  # 黄色
            elif 'success' in color or 'up' in color:
                return '#28A745'  # 绿色
            # 默认返回蓝色
            return '#36A2EB'

        # Process rgba(r, g, b, a) format
        rgba_pattern = r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)'
        match = re.match(rgba_pattern, color)
        if match:
            r, g, b, a = match.groups()
            # Convert to matplotlib format (r/255, g/255, b/255, a)
            return (int(r)/255, int(g)/255, int(b)/255, float(a))

        # Process rgb(r, g, b) format
        rgb_pattern = r'rgb\((\d+),\s*(\d+),\s*(\d+)\)'
        match = re.match(rgb_pattern, color)
        if match:
            r, g, b = match.groups()
            # Convert to matplotlib format (r/255, g/255, b/255)
            return (int(r)/255, int(g)/255, int(b)/255)

        # Other formats (hex, color names, etc.) return directly
        return color

    def _ensure_visible_color(self, color: Any, fallback: str, min_alpha: float = 0.6) -> Any:
        """
        Ensure color is visible during rendering: avoid transparent values and boost low opacity
        """
        base_color = fallback if color in (None, "", "transparent") else color
        parsed = self._parse_color(base_color)
        fallback_parsed = self._parse_color(fallback)

        if isinstance(parsed, tuple):
            if len(parsed) == 4:
                r, g, b, a = parsed
                return (r, g, b, max(a, min_alpha))
            return parsed

        if isinstance(parsed, str) and parsed.lower() == "transparent":
            return fallback_parsed

        return parsed if parsed is not None else fallback_parsed

    def _get_colors(self, datasets: List[Dict[str, Any]]) -> List[str]:
        """
        Get chart colors

        Prioritize colors defined in dataset, otherwise use default palette
        """
        colors = []
        for i, dataset in enumerate(datasets):
            # Try to get various possible color fields
            color = (
                dataset.get('backgroundColor') or
                dataset.get('borderColor') or
                dataset.get('color') or
                self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
            )

            # If color is an array, take the first one
            if isinstance(color, list):
                color = color[0] if color else self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]

            # Parse color format
            color = self._parse_color(color)

            colors.append(color)

        return colors

    def _align_labels_and_data(
        self,
        labels: Any,
        dataset_data: Any,
        chart_type: str,
        require_positive_sum: bool = False
    ) -> Tuple[List[str], List[float]]:
        """
        Align labels and data length for categorical charts, and clean non-numeric values.

        Matplotlib pie/doughnut charts require labels and data to have consistent length, otherwise errors are thrown.
        """
        original_label_len = len(labels) if isinstance(labels, list) else 0
        original_data_len = len(dataset_data) if isinstance(dataset_data, list) else 0

        aligned_labels = [str(label) for label in labels] if isinstance(labels, list) else []
        raw_data = dataset_data if isinstance(dataset_data, list) else []

        cleaned_data: List[float] = []
        for value in raw_data:
            try:
                numeric = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                numeric = 0.0
            if numeric < 0:
                numeric = 0.0
            cleaned_data.append(numeric)

        target_len = max(len(aligned_labels), len(cleaned_data))
        if target_len == 0:
            return [], []

        if len(aligned_labels) < target_len:
            start = len(aligned_labels)
            aligned_labels.extend([f"Unnamed{start + idx + 1}" for idx in range(target_len - start)])

        if len(cleaned_data) < target_len:
            cleaned_data.extend([0.0] * (target_len - len(cleaned_data)))

        if original_label_len != original_data_len:
            logger.warning(
                f"{chart_type} chart labels length ({original_label_len}) and data length ({original_data_len}) mismatch, "
                f"aligned to {target_len}"
            )

        if require_positive_sum and not any(value > 0 for value in cleaned_data):
            logger.warning(f"{chart_type} chart data is empty, skipping rendering")
            return [], []

        return aligned_labels[:target_len], cleaned_data[:target_len]

    def _figure_to_svg(self, fig: Any) -> str:
        """
        Convert matplotlib figure to SVG string
        """
        svg_buffer = io.BytesIO()
        fig.savefig(svg_buffer, format='svg', bbox_inches='tight', transparent=False, facecolor='white')
        plt.close(fig)

        svg_buffer.seek(0)
        svg_string = svg_buffer.getvalue().decode('utf-8')

        return svg_string

    def _render_line(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> Optional[str]:
        """
        Render line chart (enhanced version)

        Supported features:
        - Multiple y-axes (yAxisID: 'y', 'y1', 'y2', 'y3'...)
        - Fill area (fill: true)
        - Transparency (alpha channel in backgroundColor)
        - Line style (tension curve smoothing)
        """
        try:
            labels = data.get('labels') or []
            datasets = data.get('datasets') or []

            has_object_points = any(
                isinstance(ds, dict)
                and isinstance(ds.get('data'), list)
                and any(isinstance(pt, dict) and ('x' in pt or 'y' in pt) for pt in ds.get('data'))
                for ds in datasets
            )

            if (not datasets) or ((not labels) and not has_object_points):
                return None

            # Collect all unique yAxisIDs
            y_axis_ids = []
            for dataset in datasets:
                y_axis_id = dataset.get('yAxisID', 'y')
                if y_axis_id not in y_axis_ids:
                    y_axis_ids.append(y_axis_id)

            # Ensure 'y' is the first axis
            if 'y' in y_axis_ids:
                y_axis_ids.remove('y')
                y_axis_ids.insert(0, 'y')

            # Check if there are multiple y-axes
            has_multiple_axes = len(y_axis_ids) > 1

            title = props.get('title')
            options = props.get('options', {})
            scales = options.get('scales', {})
            x_tick_labels = list(labels) if isinstance(labels, list) else []

            # Create figure and multiple y-axes
            fig, ax1 = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)

            if title:
                ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # Create y-axis mapping dictionary
            axes = {'y': ax1}

            if has_multiple_axes:
                # Count axes at each position (left/right) for offset calculation
                left_axes_count = 0
                right_axes_count = 0

                # Create new y-axis for each additional yAxisID
                for y_axis_id in y_axis_ids[1:]:
                    if y_axis_id == 'y':
                        continue

                    # Create new y-axis
                    new_ax = ax1.twinx()
                    axes[y_axis_id] = new_ax

                    # Get axis position from scales configuration
                    y_config = scales.get(y_axis_id, {})
                    position = y_config.get('position', 'right')

                    if position == 'left':
                        # Left additional axis, offset to left
                        if left_axes_count > 0:
                            new_ax.spines['left'].set_position(('outward', 60 * left_axes_count))
                        new_ax.yaxis.set_label_position('left')
                        new_ax.yaxis.set_ticks_position('left')
                        left_axes_count += 1
                    else:
                        # Right additional axis, offset to right
                        if right_axes_count > 0:
                            new_ax.spines['right'].set_position(('outward', 60 * right_axes_count))
                        right_axes_count += 1

            colors = self._get_colors(datasets)

            # Collect line and fill info for each y-axis for legend
            axis_lines = {axis_id: [] for axis_id in y_axis_ids}
            legend_handles = []  # Legend handles
            legend_labels = []   # Legend labels

            # Draw each data series
            for i, dataset in enumerate(datasets):
                dataset_data = dataset.get('data', [])
                label = dataset.get('label', f'Series {i+1}')
                color = colors[i]

                # Get configuration
                y_axis_id = dataset.get('yAxisID', 'y')
                fill = True  # Force enable fill for comparison
                tension = dataset.get('tension', 0)  # 0 means straight line, 0.4 means smooth curve
                border_color = self._parse_color(dataset.get('borderColor', color))
                background_color = self._parse_color(dataset.get('backgroundColor', color))

                # Select corresponding axis
                ax = axes.get(y_axis_id, ax1)

                is_object_data = isinstance(dataset_data, list) and any(
                    isinstance(point, dict) and ('x' in point or 'y' in point)
                    for point in dataset_data
                )

                if is_object_data:
                    x_data = []
                    y_data = []
                    annotations = []

                    for idx, point in enumerate(dataset_data):
                        if not isinstance(point, dict):
                            continue

                        label_text = str(point.get('x', f"Point{idx + 1}"))
                        if len(x_tick_labels) < len(dataset_data):
                            x_tick_labels.append(label_text)

                        x_data.append(len(x_data))

                        y_val = point.get('y', 0)
                        try:
                            y_val = float(y_val)
                        except (TypeError, ValueError):
                            y_val = 0
                        y_data.append(y_val)
                        annotations.append(point.get('event'))

                    if not x_data:
                        continue

                    line, = ax.plot(x_data, y_data, marker='o', label=label,
                                    color=border_color, linewidth=2, markersize=6)

                    if fill:
                        ax.fill_between(x_data, y_data, alpha=0.2, color=background_color)

                    for pos, y_val, text in zip(x_data, y_data, annotations):
                        if text:
                            ax.annotate(
                                text,
                                (pos, y_val),
                                textcoords='offset points',
                                xytext=(0, 8),
                                ha='center',
                                fontsize=8,
                                rotation=20
                            )
                else:
                    # Draw line chart
                    x_data = range(len(labels))

                    # Determine smoothing based on tension value
                    if tension > 0 and SCIPY_AVAILABLE:
                        # Use spline interpolation for smooth curves (requires scipy)
                        if len(dataset_data) >= 4:  # Need at least 4 points for smoothing
                            try:
                                x_smooth = np.linspace(0, len(labels)-1, len(labels)*3)
                                spl = make_interp_spline(x_data, dataset_data, k=min(3, len(dataset_data)-1))
                                y_smooth = spl(x_smooth)
                                line, = ax.plot(x_smooth, y_smooth, label=label, color=border_color, linewidth=2)

                                # If fill needed (use very low transparency to avoid occlusion)
                                if fill:
                                    ax.fill_between(x_smooth, y_smooth, alpha=0.2, color=background_color)
                            except:
                                # If smoothing fails, use normal line
                                line, = ax.plot(x_data, dataset_data, marker='o', label=label,
                                              color=border_color, linewidth=2, markersize=6)
                                if fill:
                                    ax.fill_between(x_data, dataset_data, alpha=0.2, color=background_color)
                        else:
                            line, = ax.plot(x_data, dataset_data, marker='o', label=label,
                                          color=border_color, linewidth=2, markersize=6)
                            if fill:
                                ax.fill_between(x_data, dataset_data, alpha=0.2, color=background_color)
                    else:
                        # Straight line connection (tension=0 or scipy unavailable)
                        line, = ax.plot(x_data, dataset_data, marker='o', label=label,
                                      color=border_color, linewidth=2, markersize=6)

                        # If fill needed (use very low transparency to avoid occlusion)
                        if fill:
                            ax.fill_between(x_data, dataset_data, alpha=0.2, color=background_color)

                # Record which axis this line belongs to
                axis_lines[y_axis_id].append(line)

                # Create legend item: if filled, create legend with fill background
                if fill:
                    # Create a rectangle patch as fill background (use slightly higher transparency for legend visibility)
                    fill_patch = Rectangle((0, 0), 1, 1,
                                          facecolor=background_color,
                                          edgecolor='none',
                                          alpha=0.15)
                    # Combine line and fill patch
                    legend_handles.append((line, fill_patch))
                    legend_labels.append(label)
                else:
                    legend_handles.append(line)
                    legend_labels.append(label)

            # Set x-axis labels
            if x_tick_labels:
                ax1.set_xticks(range(len(x_tick_labels)))
                ax1.set_xticklabels(x_tick_labels, rotation=45, ha='right')

            # Set y-axis labels and titles
            for y_axis_id, ax in axes.items():
                y_config = scales.get(y_axis_id, {})
                y_title = y_config.get('title', {}).get('text', '')

                if y_title:
                    ax.set_ylabel(y_title, fontsize=11)

                # Set y-axis label color (if axis has only one line, use that line's color)
                if len(axis_lines[y_axis_id]) == 1:
                    line_color = axis_lines[y_axis_id][0].get_color()
                    ax.tick_params(axis='y', labelcolor=line_color)
                    ax.yaxis.label.set_color(line_color)

            # Set grid (only show on main axis)
            ax1.grid(True, alpha=0.3, linestyle='--')
            for y_axis_id in y_axis_ids[1:]:
                if y_axis_id in axes:
                    axes[y_axis_id].grid(False)

            # Create legend
            if has_multiple_axes or len(datasets) > 1:
                # Use custom legend_handles and legend_labels
                from matplotlib.legend_handler import HandlerTuple

                ax1.legend(legend_handles, legend_labels,
                          loc='best',
                          framealpha=0.9,
                          handler_map={tuple: HandlerTuple(ndivide=None)})

            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render line chart: {e}", exc_info=True)
            return None

    def _render_bar(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int,
        horizontal: bool = False
    ) -> Optional[str]:
        """Render bar chart (supports horizontal barh)"""
        try:
            labels = data.get('labels', [])
            datasets = data.get('datasets', [])

            if not labels or not datasets:
                return None

            title = props.get('title')
            fig, ax = self._create_figure(width, height, dpi, title)

            colors = self._get_colors(datasets)

            # Calculate bar positions
            positions = np.arange(len(labels))
            width_bar = 0.8 / len(datasets) if len(datasets) > 1 else 0.6

            # Horizontal/vertical drawing
            for i, dataset in enumerate(datasets):
                dataset_data = dataset.get('data', [])
                label = dataset.get('label', f'Series {i+1}')
                color = colors[i]

                offset = (i - len(datasets)/2 + 0.5) * width_bar

                if horizontal:
                    ax.barh(
                        positions + offset,
                        dataset_data,
                        height=width_bar,
                        label=label,
                        color=color,
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.5
                    )
                else:
                    ax.bar(
                        positions + offset,
                        dataset_data,
                        width_bar,
                        label=label,
                        color=color,
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.5
                    )

            # Axis labels/grid
            if horizontal:
                ax.set_yticks(positions)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()  # Consistent with Chart.js horizontal layout
                ax.grid(True, alpha=0.3, linestyle='--', axis='x')
            else:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            # Show legend
            if len(datasets) > 1:
                ax.legend(loc='best', framealpha=0.9)

            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render bar chart: {e}")
            return None

    def _render_bubble(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> Optional[str]:
        """Render bubble chart"""
        try:
            datasets = data.get('datasets', [])
            if not datasets:
                return None

            title = props.get('title')
            fig, ax = self._create_figure(width, height, dpi, title)
            colors = self._get_colors(datasets)

            def _safe_radius(raw) -> float:
                """Safely convert input radius to float and set minimum threshold to avoid bubbles disappearing completely"""
                try:
                    val = float(raw)
                    return max(val, 0.5)
                except Exception:
                    return 1.0

            all_x: list[float] = []
            all_y: list[float] = []
            max_r: float = 0.0

            for i, dataset in enumerate(datasets):
                points = dataset.get('data', [])
                label = dataset.get('label', f'Series {i+1}')
                color = colors[i]

                if points and isinstance(points[0], dict):
                    xs = [p.get('x', 0) for p in points]
                    ys = [p.get('y', 0) for p in points]
                    rs = [_safe_radius(p.get('r', 1)) for p in points]
                else:
                    xs = list(range(len(points)))
                    ys = points
                    rs = [1.0 for _ in points]

                all_x.extend(xs)
                all_y.extend(ys)
                if rs:
                    max_r = max(max_r, max(rs))

                # Moderately enlarge radius, approximate Chart.js pixel size (dynamic scale, avoid too large occlusion)
                size_scale = 8.0 if max_r <= 20 else 6.5
                sizes = [(r * size_scale) ** 2 for r in rs]

                ax.scatter(
                    xs,
                    ys,
                    s=sizes,
                    label=label,
                    color=color,
                    alpha=0.45,
                    edgecolors='white',
                    linewidth=0.6
                )

            if len(datasets) > 1:
                ax.legend(loc='best', framealpha=0.9)

            # Add appropriate padding to avoid large bubbles being cropped
            if all_x and all_y:
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                x_span = max(x_max - x_min, 1e-6)
                y_span = max(y_max - y_min, 1e-6)
                pad_x = max(x_span * 0.12, max_r * 1.2)
                pad_y = max(y_span * 0.12, max_r * 1.2)
                ax.set_xlim(x_min - pad_x, x_max + pad_x)
                ax.set_ylim(y_min - pad_y, y_max + pad_y)
                # Extra safety margin
                ax.margins(x=0.05, y=0.05)

            ax.grid(True, alpha=0.3, linestyle='--')
            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render bubble chart: {e}", exc_info=True)
            return None

    def _render_pie(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> Optional[str]:
        """Render pie chart"""
        try:
            labels = data.get('labels', [])
            datasets = data.get('datasets', [])

            if not labels or not datasets:
                return None

            # Pie chart only uses the first dataset
            dataset = datasets[0]
            dataset_data = dataset.get('data', [])

            labels, dataset_data = self._align_labels_and_data(
                labels,
                dataset_data,
                chart_type="pie",
                require_positive_sum=True
            )

            if not labels or not dataset_data:
                return None

            title = props.get('title')
            fig, ax = self._create_figure(width, height, dpi, title)

            # Get colors
            raw_colors = dataset.get('backgroundColor', self.DEFAULT_COLORS[:len(labels)])
            if not isinstance(raw_colors, list):
                raw_colors = self.DEFAULT_COLORS[:len(labels)]

            colors = [
                self._ensure_visible_color(
                    raw_colors[i] if i < len(raw_colors) else None,
                    self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                )
                for i in range(len(labels))
            ]

            # Draw pie chart
            wedges, texts, autotexts = ax.pie(
                dataset_data,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 10}
            )

            # Set percentage text to white
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.axis('equal')  # Keep circular

            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render pie chart: {e}")
            return None

    def _render_doughnut(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> Optional[str]:
        """Render doughnut chart"""
        try:
            labels = data.get('labels', [])
            datasets = data.get('datasets', [])

            if not labels or not datasets:
                return None

            # Doughnut chart only uses the first dataset
            dataset = datasets[0]
            dataset_data = dataset.get('data', [])

            labels, dataset_data = self._align_labels_and_data(
                labels,
                dataset_data,
                chart_type="doughnut",
                require_positive_sum=True
            )

            if not labels or not dataset_data:
                return None

            title = props.get('title')
            fig, ax = self._create_figure(width, height, dpi, title)

            # Get colors
            raw_colors = dataset.get('backgroundColor', self.DEFAULT_COLORS[:len(labels)])
            if not isinstance(raw_colors, list):
                raw_colors = self.DEFAULT_COLORS[:len(labels)]

            colors = [
                self._ensure_visible_color(
                    raw_colors[i] if i < len(raw_colors) else None,
                    self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                )
                for i in range(len(labels))
            ]

            # Draw doughnut chart (hollow effect via wedgeprops)
            wedges, texts, autotexts = ax.pie(
                dataset_data,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.5, edgecolor='white'),
                textprops={'fontsize': 10}
            )

            # Set percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.axis('equal')

            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render doughnut chart: {e}")
            return None

    def _render_radar(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> Optional[str]:
        """Render radar chart"""
        try:
            labels = data.get('labels', [])
            datasets = data.get('datasets', [])

            if not labels or not datasets:
                return None

            title = props.get('title')
            fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)

            # Create polar subplot
            ax = fig.add_subplot(111, projection='polar')

            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            colors = self._get_colors(datasets)

            # Calculate angles
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]  # Close the shape

            # Draw each data series
            for i, dataset in enumerate(datasets):
                dataset_data = dataset.get('data', [])
                label = dataset.get('label', f'Series {i+1}')
                color = colors[i]

                # Close the data
                values = dataset_data + dataset_data[:1]

                # Draw radar chart
                ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
                ax.fill(angles, values, alpha=0.25, color=color)

            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)

            # Show legend
            if len(datasets) > 1:
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render radar chart: {e}")
            return None

    def _render_scatter(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> Optional[str]:
        """Render scatter chart"""
        try:
            datasets = data.get('datasets', [])

            if not datasets:
                return None

            title = props.get('title')
            fig, ax = self._create_figure(width, height, dpi, title)

            colors = self._get_colors(datasets)

            # Draw each data series
            for i, dataset in enumerate(datasets):
                dataset_data = dataset.get('data', [])
                label = dataset.get('label', f'Series {i+1}')
                color = colors[i]

                # Extract x and y coordinates
                if dataset_data and isinstance(dataset_data[0], dict):
                    x_values = [point.get('x', 0) for point in dataset_data]
                    y_values = [point.get('y', 0) for point in dataset_data]
                else:
                    # If not in {x,y} format, use index as x
                    x_values = range(len(dataset_data))
                    y_values = dataset_data

                ax.scatter(
                    x_values,
                    y_values,
                    label=label,
                    color=color,
                    s=50,
                    alpha=0.6,
                    edgecolors='white',
                    linewidth=0.5
                )

            # Show legend
            if len(datasets) > 1:
                ax.legend(loc='best', framealpha=0.9)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')

            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render scatter chart: {e}")
            return None

    def _render_polarArea(
        self,
        data: Dict[str, Any],
        props: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> Optional[str]:
        """Render polar area chart"""
        try:
            labels = data.get('labels', [])
            datasets = data.get('datasets', [])

            if not labels or not datasets:
                return None

            # Only use first dataset
            dataset = datasets[0]
            dataset_data = dataset.get('data', [])

            labels, dataset_data = self._align_labels_and_data(
                labels,
                dataset_data,
                chart_type="polarArea",
                require_positive_sum=False
            )

            if not labels or not dataset_data:
                return None

            title = props.get('title')
            fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            ax = fig.add_subplot(111, projection='polar')

            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # Get colors
            raw_colors = dataset.get('backgroundColor', self.DEFAULT_COLORS[:len(labels)])
            if not isinstance(raw_colors, list):
                raw_colors = self.DEFAULT_COLORS[:len(labels)]

            colors = [
                self._ensure_visible_color(
                    raw_colors[i] if i < len(raw_colors) else None,
                    self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                )
                for i in range(len(labels))
            ]

            # Calculate angles
            theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            width_bar = 2 * np.pi / len(labels)

            # Draw polar area chart
            bars = ax.bar(
                theta,
                dataset_data,
                width=width_bar,
                bottom=0.0,
                color=colors,
                alpha=0.7,
                edgecolor='white',
                linewidth=1
            )

            # Set labels
            ax.set_xticks(theta)
            ax.set_xticklabels(labels)

            return self._figure_to_svg(fig)

        except Exception as e:
            logger.error(f"Failed to render polar area chart: {e}")
            return None


def create_chart_converter(font_path: Optional[str] = None) -> ChartToSVGConverter:
    """
    Create chart converter instance

    Parameters:
        font_path: Chinese font path (optional)

    Returns:
        ChartToSVGConverter: Converter instance
    """
    return ChartToSVGConverter(font_path=font_path)


__all__ = ["ChartToSVGConverter", "create_chart_converter"]
