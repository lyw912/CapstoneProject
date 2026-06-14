#!/usr/bin/env python
"""
Report Agent 命令行入口（调用 ReportEngine 模块）

这是一个不需要前端的命令行报告生成程序。
主要流程：
1. 检查PDF依赖
2. 获取最新的log、md文件
3. 直接调用 Report Agent（ReportEngine）生成报告（跳过文件增加审核）
4. 自动保存HTML、PDF（如果有依赖）和Markdown到final_reports/（Markdown 会在 PDF 之后生成）

使用方法：
    python report_engine_only.py [选项]

选项：
    --query QUERY     指定报告主题（可选，默认从文件名提取）
    --skip-pdf        跳过PDF生成（即使有依赖）
    --skip-markdown   跳过Markdown生成
    --verbose         显示详细日志
    --help            显示帮助信息
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from config import settings
from utils.sensitive_input_filter import check_sensitive_input, filter_settings_from_config, SENSITIVE_INPUT_MESSAGE
from loguru import logger

# 全局配置
VERBOSE = False

# 配置日志
def setup_logger(verbose: bool = False):
    """设置日志配置"""
    global VERBOSE
    VERBOSE = verbose

    logger.remove()  # 移除默认处理器
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG" if verbose else "INFO"
    )


def check_dependencies() -> tuple[bool, Optional[str]]:
    """
    Check system dependencies required for PDF generation

    Returns:
        tuple: (is_available: bool, message: str)
            - is_available: Whether PDF feature is available
            - message: Dependency check result message
    """
    logger.info("=" * 70)
    logger.info("Step 1/4: Checking system dependencies")
    logger.info("=" * 70)

    try:
        from ReportEngine.utils.dependency_check import check_pango_available
        is_available, message = check_pango_available()

        if is_available:
            logger.success("✓ PDF dependency check passed, will generate both HTML and PDF files")
        else:
            logger.warning("⚠ PDF dependencies missing, will only generate HTML file")
            logger.info("\n" + message)

        return is_available, message
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False, str(e)


def get_latest_engine_reports() -> Dict[str, str]:
    """
    Get latest report files from Media/Query engine directories

    Returns:
        Dict[str, str]: Mapping of engine name to file path
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 2/4: Getting latest analysis engine reports")
    logger.info("=" * 70)

    directories = {
        'query': 'query_engine_streamlit_reports',
        'media': 'media_engine_streamlit_reports',
    }

    latest_files = {}

    for engine, directory in directories.items():
        if not os.path.exists(directory):
            logger.warning(f"⚠ {engine.capitalize()} Engine directory does not exist: {directory}")
            continue

        # Get all .md files
        md_files = [f for f in os.listdir(directory) if f.endswith('.md')]

        if not md_files:
            logger.warning(f"⚠ No .md files found in {engine.capitalize()} Engine directory")
            continue

        # Get latest file
        latest_file = max(
            md_files,
            key=lambda x: os.path.getmtime(os.path.join(directory, x))
        )
        latest_path = os.path.join(directory, latest_file)
        latest_files[engine] = latest_path

        logger.info(f"✓ Found latest {engine.capitalize()} Engine report")

    if not latest_files:
        logger.error("❌ No engine report files found, please run analysis engines first")
        sys.exit(1)

    logger.info(f"\nFound {len(latest_files)} engine reports in total")

    return latest_files


def confirm_file_selection(latest_files: Dict[str, str]) -> bool:
    """
    Confirm selected files with user

    Args:
        latest_files: Mapping of engine name to file path

    Returns:
        bool: True if user confirms, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("Please confirm the following selected files:")
    logger.info("=" * 70)

    for engine, file_path in latest_files.items():
        filename = os.path.basename(file_path)
        # Get file modification time
        mtime = os.path.getmtime(file_path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"  {engine.capitalize()} Engine:")
        logger.info(f"    Filename: {filename}")
        logger.info(f"    Path: {file_path}")
        logger.info(f"    Modified: {mtime_str}")
        logger.info("")

    logger.info("=" * 70)

    # Prompt user for confirmation
    try:
        response = input("Use above files to generate report? [Y/n]: ").strip().lower()

        # Default is y, so empty input or y means confirm
        if response == '' or response == 'y' or response == 'yes':
            logger.success("✓ User confirmed, continuing report generation")
            return True
        else:
            logger.warning("✗ User cancelled operation")
            return False
    except (KeyboardInterrupt, EOFError):
        logger.warning("\n✗ User cancelled operation")
        return False


def load_engine_reports(latest_files: Dict[str, str]) -> list[str]:
    """
    Load engine report contents

    Args:
        latest_files: Mapping of engine name to file path

    Returns:
        list[str]: List of report contents
    """
    reports = []

    for engine, file_path in latest_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                reports.append(content)
                logger.debug(f"Loaded {engine} report, length: {len(content)} characters")
        except Exception as e:
            logger.error(f"Failed to load {engine} report: {e}")

    return reports


def extract_query_from_reports(latest_files: Dict[str, str]) -> str:
    """
    Extract query topic from report filenames

    Args:
        latest_files: Mapping of engine name to file path

    Returns:
        str: Extracted query topic
    """
    # Try to extract topic from filename
    for engine, file_path in latest_files.items():
        filename = os.path.basename(file_path)
        # Assume filename format: report_topic_timestamp.md
        if '_' in filename:
            parts = filename.replace('.md', '').split('_')
            if len(parts) >= 2:
                # Extract middle part as topic
                topic = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                if topic:
                    return topic

    # Return default if unable to extract
    return "Comprehensive Analysis Report"


def generate_report(reports: list[str], query: str, pdf_available: bool) -> Dict[str, Any]:
    """
    Call Report Agent (ReportEngine) to generate report

    Args:
        reports: List of report contents
        query: Report topic
        pdf_available: Whether PDF feature is available

    Returns:
        Dict[str, Any]: Dictionary containing generation results
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 3/4: Generating comprehensive report")
    logger.info("=" * 70)
    logger.info(f"Report topic: {query}")
    logger.info(f"Input reports count: {len(reports)}")

    try:
        from ReportEngine.agent import ReportAgent

        # Initialize Report Agent
        logger.info("Initializing Report Agent (ReportEngine)...")
        agent = ReportAgent()

        # Define streaming event handler
        def stream_handler(event_type: str, payload: Dict[str, Any]):
            """Handle Report Agent streaming events"""
            if event_type == 'stage':
                stage = payload.get('stage', '')
                if stage == 'agent_start':
                    logger.info(f"Starting report generation: {payload.get('report_id', '')}")
                elif stage == 'template_selected':
                    logger.info(f"✓ Template selected: {payload.get('template', '')}")
                elif stage == 'template_sliced':
                    logger.info(f"✓ Template parsing complete, {payload.get('section_count', 0)} sections total")
                elif stage == 'layout_designed':
                    logger.info(f"✓ Document layout design complete")
                    logger.info(f"  Title: {payload.get('title', '')}")
                elif stage == 'word_plan_ready':
                    logger.info(f"✓ Word plan ready, target chapters: {payload.get('chapter_targets', 0)}")
                elif stage == 'chapters_compiled':
                    logger.info(f"✓ Chapter generation complete, {payload.get('chapter_count', 0)} chapters total")
                elif stage == 'html_rendered':
                    logger.info(f"✓ HTML rendering complete")
                elif stage == 'report_saved':
                    logger.info(f"✓ Report saved")
            elif event_type == 'chapter_status':
                chapter_id = payload.get('chapterId', '')
                title = payload.get('title', '')
                status = payload.get('status', '')
                if status == 'generating':
                    logger.info(f"  Generating chapter: {title}")
                elif status == 'completed':
                    attempt = payload.get('attempt', 1)
                    warning = payload.get('warning', '')
                    if warning:
                        logger.warning(f"  ✓ Chapter complete: {title} (Attempt {attempt}, {payload.get('warningMessage', '')})")
                    else:
                        logger.success(f"  ✓ Chapter complete: {title}")
            elif event_type == 'error':
                logger.error(f"Error: {payload.get('message', '')}")

        # Generate report
        logger.info("Starting report generation, this may take a few minutes...")
        result = agent.generate_report(
            query=query,
            reports=reports,
            forum_logs="",  # No forum logs
            custom_template="",  # Use auto template selection
            save_report=True,  # Auto save report
            stream_handler=stream_handler
        )

        logger.success("✓ Report generation successful!")
        return result

    except Exception as e:
        logger.exception(f"❌ Report generation failed: {e}")
        sys.exit(1)


def save_pdf(document_ir_path: str, query: str) -> Optional[str]:
    """
    Generate and save PDF from IR file

    Args:
        document_ir_path: Document IR file path
        query: Report topic

    Returns:
        Optional[str]: PDF file path, None if failed
    """
    logger.info("\nGenerating PDF file...")

    try:
        # Read IR data
        with open(document_ir_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        # Create PDF renderer
        from ReportEngine.renderers import PDFRenderer
        renderer = PDFRenderer()

        # Prepare output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(
            c for c in query if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        query_safe = query_safe.replace(" ", "_")[:30] or "report"

        pdf_dir = Path("final_reports") / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_filename = f"final_report_{query_safe}_{timestamp}.pdf"
        pdf_path = pdf_dir / pdf_filename

        # Use render_to_pdf to directly generate PDF file, pass IR file path for saving after repair
        logger.info(f"Starting PDF rendering: {pdf_path}")
        result_path = renderer.render_to_pdf(
            document_ir,
            pdf_path,
            optimize_layout=True,
            ir_file_path=document_ir_path
        )

        # Display file size
        file_size = result_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.success(f"✓ PDF saved: {pdf_path}")
        logger.info(f"  File size: {size_mb:.2f} MB")

        return str(result_path)

    except Exception as e:
        logger.exception(f"❌ PDF generation failed: {e}")
        return None


def save_markdown(document_ir_path: str, query: str) -> Optional[str]:
    """
    Generate and save Markdown from IR file

    Args:
        document_ir_path: Document IR file path
        query: Report topic

    Returns:
        Optional[str]: Markdown file path, None if failed
    """
    logger.info("\nGenerating Markdown file...")

    try:
        with open(document_ir_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        from ReportEngine.renderers import MarkdownRenderer
        renderer = MarkdownRenderer()
        # Pass IR file path for saving after repair
        markdown_content = renderer.render(document_ir, ir_file_path=document_ir_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(
            c for c in query if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        query_safe = query_safe.replace(" ", "_")[:30] or "report"

        md_dir = Path("final_reports") / "md"
        md_dir.mkdir(parents=True, exist_ok=True)

        md_filename = f"final_report_{query_safe}_{timestamp}.md"
        md_path = md_dir / md_filename

        md_path.write_text(markdown_content, encoding='utf-8')

        file_size_kb = md_path.stat().st_size / 1024
        logger.success(f"✓ Markdown saved: {md_path}")
        logger.info(f"  File size: {file_size_kb:.1f} KB")

        return str(md_path)

    except Exception as e:
        logger.exception(f"❌ Markdown generation failed: {e}")
        return None


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Report Agent CLI - Frontend-free report generation tool (ReportEngine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python report_engine_only.py
  python report_engine_only.py --query "Civil Engineering Industry Analysis"
  python report_engine_only.py --skip-pdf --verbose

Note:
  The program will automatically get latest report files from engine directories,
  skip file addition review, directly generate comprehensive report,
  and generate Markdown after PDF by default.
        """
    )

    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Specify report topic (auto-extracted from filename by default)'
    )

    parser.add_argument(
        '--skip-pdf',
        action='store_true',
        help='Skip PDF generation (even if system supports it)'
    )

    parser.add_argument(
        '--skip-markdown',
        action='store_true',
        help='Skip Markdown generation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed log messages'
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 设置日志
    setup_logger(verbose=args.verbose)

    logger.info("\n")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 18 + "Report Agent CLI Version" + " " * 26 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("\n")

    # Step 1: Check dependencies
    pdf_available, _ = check_dependencies()
    markdown_enabled = not args.skip_markdown

    # If user specified skip PDF, disable PDF generation
    if args.skip_pdf:
        logger.info("User specified --skip-pdf, will skip PDF generation")
        pdf_available = False

    if not markdown_enabled:
        logger.info("User specified --skip-markdown, will skip Markdown generation")

    # Step 2: Get latest files
    latest_files = get_latest_engine_reports()

    # Confirm file selection
    if not confirm_file_selection(latest_files):
        logger.info("\nProgram exited")
        sys.exit(0)

    # Load report contents
    reports = load_engine_reports(latest_files)

    if not reports:
        logger.error("❌ Failed to load any report content")
        sys.exit(1)

    # Extract or use specified query topic
    query = args.query if args.query else extract_query_from_reports(latest_files)
    logger.info(f"Using report topic: {query}")

    if args.query:
        enabled, words_file = filter_settings_from_config(settings)
        if check_sensitive_input(query, enabled=enabled, words_file=words_file):
            logger.error(f"❌ {SENSITIVE_INPUT_MESSAGE}")
            sys.exit(1)

    # Step 3: Generate report
    result = generate_report(reports, query, pdf_available)

    # Step 4: Save files
    logger.info("\n" + "=" * 70)
    logger.info("Step 4/4: Saving generated files")
    logger.info("=" * 70)

    # HTML is already auto-saved in generate_report
    html_path = result.get('report_filepath', '')
    ir_path = result.get('ir_filepath', '')
    pdf_path = None
    markdown_path = None

    if html_path:
        logger.success(f"✓ HTML saved: {result.get('report_relative_path', html_path)}")

    # If PDF dependencies available, generate and save PDF
    if pdf_available:
        if ir_path and os.path.exists(ir_path):
            pdf_path = save_pdf(ir_path, query)
        else:
            logger.warning("⚠ IR file not found, cannot generate PDF")
    else:
        logger.info("⚠ Skipping PDF generation (missing dependencies or user specified)")

    # Generate and save Markdown (after PDF)
    if markdown_enabled:
        if ir_path and os.path.exists(ir_path):
            markdown_path = save_markdown(ir_path, query)
        else:
            logger.warning("⚠ IR file not found, cannot generate Markdown")
    else:
        logger.info("⚠ Skipping Markdown generation (user specified)")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.success("✓ Report generation complete!")
    logger.info("=" * 70)
    logger.info(f"Report ID: {result.get('report_id', 'N/A')}")
    logger.info(f"HTML file: {result.get('report_relative_path', 'N/A')}")
    if pdf_available:
        if pdf_path:
            logger.info(f"PDF file: {os.path.relpath(pdf_path, os.getcwd())}")
        else:
            logger.info("PDF file: Generation failed, please check logs")
    else:
        logger.info("PDF file: Skipped")
    if markdown_enabled:
        if markdown_path:
            logger.info(f"Markdown file: {os.path.relpath(markdown_path, os.getcwd())}")
        else:
            logger.info("Markdown file: Generation failed, please check logs")
    else:
        logger.info("Markdown file: Skipped")
    logger.info("=" * 70)
    logger.info("\nProgram finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\nUser interrupted program")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"\nProgram exited with error: {e}")
        sys.exit(1)
