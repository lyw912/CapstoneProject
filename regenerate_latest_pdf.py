"""
使用新的SVG矢量图表功能重新生成最新报告的PDF
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ReportEngine.renderers import PDFRenderer

def find_latest_report():
    """
    在 `final_reports/ir` 中查找最新的报告 IR JSON。

    按修改时间倒序选择第一条，若目录或文件缺失则记录错误并返回 None。

    返回:
        Path | None: 最新 IR 文件路径；未找到则为 None。
    """
    ir_dir = Path("final_reports/ir")

    if not ir_dir.exists():
        logger.error(f"Report directory does not exist: {ir_dir}")
        return None

    # 获取所有JSON文件并按修改时间排序
    json_files = sorted(ir_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not json_files:
        logger.error("No report file found")
        return None

    latest_file = json_files[0]
    logger.info(f"Found latest report: {latest_file.name}")

    return latest_file

def load_document_ir(file_path):
    """
    读取指定路径的 Document IR JSON，并统计章节/图表数量。

    解析失败时返回 None；成功时会打印章节数与图表数，便于确认
    输入报告的规模。

    参数:
        file_path: IR 文件路径

    返回:
        dict | None: 解析后的 Document IR；失败返回 None。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        logger.info(f"Successfully loaded report: {file_path.name}")

        # Count charts
        chart_count = 0
        chapters = document_ir.get('chapters', [])

        def count_charts(blocks):
            """Recursively count Chart.js charts in block list"""
            count = 0
            for block in blocks:
                if isinstance(block, dict):
                    if block.get('type') == 'widget' and block.get('widgetType', '').startswith('chart.js'):
                        count += 1
                    # 递归处理嵌套blocks
                    nested = block.get('blocks')
                    if isinstance(nested, list):
                        count += count_charts(nested)
            return count

        for chapter in chapters:
            blocks = chapter.get('blocks', [])
            chart_count += count_charts(blocks)

        logger.info(f"Report contains {len(chapters)} chapters, {chart_count} charts")

        return document_ir

    except Exception as e:
        logger.error(f"Failed to load report: {e}")
        return None

def generate_pdf_with_vector_charts(document_ir, output_path, ir_file_path=None):
    """
    使用 PDFRenderer 将 Document IR 渲染为包含 SVG 矢量图表的 PDF。

    启用布局优化，生成后输出文件大小与成功提示；异常时返回 None。

    参数:
        document_ir: 完整的 Document IR
        output_path: 目标 PDF 路径
        ir_file_path: 可选，IR 文件路径，提供时修复后会自动保存

    返回:
        Path | None: 成功时返回生成的 PDF 路径，失败返回 None。
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting PDF generation (with vector charts)")
        logger.info("=" * 60)

        # Create PDF renderer
        renderer = PDFRenderer()

        # Render PDF, pass ir_file_path for saving after repair
        result_path = renderer.render_to_pdf(
            document_ir,
            output_path,
            optimize_layout=True,
            ir_file_path=str(ir_file_path) if ir_file_path else None
        )

        logger.info("=" * 60)
        logger.info(f"✓ PDF generated successfully: {result_path}")
        logger.info("=" * 60)

        # Display file size
        file_size = result_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.info(f"File size: {size_mb:.2f} MB")

        return result_path

    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}", exc_info=True)
        return None

def main():
    """
    主入口：重新生成最新报告的矢量 PDF。

    步骤：
        1) 查找最新 IR 文件；
        2) 读取并统计报告结构；
        3) 构造输出文件名并确保目录存在；
        4) 调用渲染函数生成 PDF，输出路径与特性说明。

    返回:
        int: 0 表示成功，非 0 表示失败。
    """
    logger.info("🚀 Regenerating PDF for latest report using SVG vector charts")
    logger.info("")

    # 1. Find latest report
    latest_report = find_latest_report()
    if not latest_report:
        logger.error("No report file found")
        return 1

    # 2. Load report data
    document_ir = load_document_ir(latest_report)
    if not document_ir:
        logger.error("Failed to load report")
        return 1

    # 3. Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = latest_report.stem.replace("report_ir_", "")
    output_filename = f"report_vector_{report_name}_{timestamp}.pdf"
    output_path = Path("final_reports/pdf") / output_filename

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output path: {output_path}")
    logger.info("")

    # 4. Generate PDF, pass IR file path for saving after repair
    result = generate_pdf_with_vector_charts(document_ir, output_path, ir_file_path=latest_report)

    if result:
        logger.info("")
        logger.info("🎉 PDF generation complete!")
        logger.info("")
        logger.info("Features:")
        logger.info("  ✓ Charts rendered in SVG vector format")
        logger.info("  ✓ Infinite zoom without quality loss")
        logger.info("  ✓ Full chart visual effects preserved")
        logger.info("  ✓ Line charts, bar charts, pie charts, etc. are all vector curves")
        logger.info("")
        logger.info(f"PDF file location: {result.absolute()}")
        return 0
    else:
        logger.error("❌ PDF generation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
