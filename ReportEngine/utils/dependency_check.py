"""
System dependency detection utility
Used for detecting system dependencies required for PDF generation
"""
import os
import sys
import platform
from pathlib import Path
from loguru import logger
from ctypes import util as ctypes_util

BOX_CONTENT_WIDTH = 62


def _box_line(text: str = "") -> str:
    """Render a single line inside the 66-char help box."""
    return f"║  {text:<{BOX_CONTENT_WIDTH}}║\n"


def _get_platform_specific_instructions():
    """
    Get installation instructions for the current platform

    Returns:
        str: Platform-specific installation instructions
    """
    system = platform.system()

    def _box_lines(lines):
        """Batch wrap multiple lines of text into bordered prompt blocks"""
        return "".join(_box_line(line) for line in lines)

    if system == "Darwin":  # macOS
        return _box_lines(
            [
                "🍎 macOS System Solution:",
                "",
                "Step 1: Install dependencies (run on host machine)",
                "  brew install pango gdk-pixbuf libffi",
                "",
                "Step 2: Set DYLD_LIBRARY_PATH (required)",
                "  Apple Silicon:",
                " export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH",
                "  Intel:",
                " export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH",
                "",
                "Step 3: Make permanent (recommended)",
                "  Append export DYLD_LIBRARY_PATH=... to ~/.zshrc",
                "  Apple uses /opt/homebrew/lib, Intel uses /usr/local/lib",
                "  Run source ~/.zshrc then open new terminal",
                "",
                "Step 4: Verify in new terminal",
                "  python -m ReportEngine.utils.dependency_check",
                "  Output containing 'Pango dependency check passed' means correct config",
            ]
        )
    elif system == "Linux":
        return _box_lines(
            [
                "🐧 Linux System Solution:",
                "",
                "Ubuntu/Debian (run on host machine):",
                "  sudo apt-get update",
                "  sudo apt-get install -y \\",
                "    libpango-1.0-0 libpangoft2-1.0-0 libffi-dev libcairo2",
                "    libgdk-pixbuf-2.0-0 (use libgdk-pixbuf2.0-0 if missing)",
                "",
                "CentOS/RHEL:",
                "  sudo yum install -y pango gdk-pixbuf2 libffi-devel cairo",
                "",
                "Docker deployment requires no additional installation, image already includes dependencies",
            ]
        )
    elif system == "Windows":
        return _box_lines(
            [
                "🪟 Windows System Solution:",
                "",
                "Step 1: Install GTK3 Runtime (run on host machine)",
                "  Install GTK3/Pango runtime libraries as documented in docs/operations/setup.md",
                "",
                "Step 2: Add GTK bin directory to PATH (requires new terminal)",
                "  set PATH=C:\\Program Files\\GTK3-Runtime Win64\\bin;%PATH%",
                "  Replace with custom path, or set GTK_BIN_PATH environment variable",
                "  Optional: Permanently add PATH example:",
                "    setx PATH \"C:\\Program Files\\GTK3-Runtime Win64\\bin;%PATH%\"",
                "",
                "Step 3: Verify (run in new terminal)",
                "  python -m ReportEngine.utils.dependency_check",
                "  Output containing '✓ Pango dependency check passed' means correct config",
            ]
        )
    else:
        return _box_lines(["Please see docs/operations/setup.md#3-pdf-export-dependencies for installation instructions for your system"])


def _ensure_windows_gtk_paths():
    """
    Auto-supplement GTK/Pango runtime search paths for Windows to resolve DLL not found issues.

    Returns:
        str | None: Successfully added path (None if no match found)
    """
    if platform.system() != "Windows":
        return None

    candidates = []
    seen = set()

    def _add_candidate(path_like):
        """Collect possible GTK installation paths, avoid duplicates and support user custom directories"""
        if not path_like:
            return
        p = Path(path_like)
        # If the installation root directory is passed, try appending bin
        if p.is_dir() and p.name.lower() == "bin":
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(p)
        else:
            for maybe in (p, p / "bin"):
                key = str(maybe.resolve()).lower()
                if maybe.exists() and key not in seen:
                    seen.add(key)
                    candidates.append(maybe)

    # User custom environment variables take priority
    for env_var in ("GTK3_RUNTIME_PATH", "GTK_RUNTIME_PATH", "GTK_BIN_PATH", "GTK_BIN_DIR", "GTK_PATH"):
        _add_candidate(os.environ.get(env_var))

    program_files = os.environ.get("ProgramFiles", r"C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")
    default_dirs = [
        Path(program_files) / "GTK3-Runtime Win64",
        Path(program_files_x86) / "GTK3-Runtime Win64",
        Path(program_files) / "GTK3-Runtime Win32",
        Path(program_files_x86) / "GTK3-Runtime Win32",
        Path(program_files) / "GTK3-Runtime",
        Path(program_files_x86) / "GTK3-Runtime",
    ]

    # Common custom installation locations (other drives / DevelopSoftware directories)
    common_drives = ["C", "D", "E", "F"]
    common_names = ["GTK3-Runtime Win64", "GTK3-Runtime Win32", "GTK3-Runtime"]
    for drive in common_drives:
        root = Path(f"{drive}:/")
        # Check if path exists and is accessible
        try:
            if root.exists():
                for name in common_names:
                    default_dirs.append(root / name)
                    default_dirs.append(root / "DevelopSoftware" / name)
        except OSError as e:
            # Skip if drive doesn't exist or is encrypted
            pass

    # Scan Program Files for all directories starting with GTK, to support custom installation directory names
    for root in (program_files, program_files_x86):
        root_path = Path(root)
        if root_path.exists():
            for child in root_path.glob("GTK*"):
                default_dirs.append(child)

    for d in default_dirs:
        _add_candidate(d)

    # Also try to recognize if user has added custom path to PATH
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for entry in path_entries:
        if not entry:
            continue
        # Rough filter for directories containing gtk or pango
        if "gtk" in entry.lower() or "pango" in entry.lower():
            _add_candidate(entry)

    for path in candidates:
        if not path or not path.exists():
            continue
        if not any(path.glob("pango*-1.0-*.dll")) and not (path / "pango-1.0-0.dll").exists():
            continue

        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(path))
        except Exception:
            # If adding fails, continue trying the PATH method
            pass

        current_path = os.environ.get("PATH", "")
        if str(path) not in current_path.split(";"):
            os.environ["PATH"] = f"{path};{current_path}"

        return str(path)

    return None


def prepare_pango_environment():
    """
    Initialize local dependency search paths for runtime (currently mainly for Windows and macOS).

    Returns:
        str | None: Successfully added path (None if no match found)
    """
    system = platform.system()
    if system == "Windows":
        return _ensure_windows_gtk_paths()
    if system == "Darwin":
        # Auto-complete DYLD_LIBRARY_PATH, compatible with Apple Silicon and Intel
        candidates = [Path("/opt/homebrew/lib"), Path("/usr/local/lib")]
        current = os.environ.get("DYLD_LIBRARY_PATH", "")
        added = []
        for c in candidates:
            if c.exists() and str(c) not in current.split(":"):
                added.append(str(c))
        if added:
            os.environ["DYLD_LIBRARY_PATH"] = ":".join(added + ([current] if current else []))
            return os.environ["DYLD_LIBRARY_PATH"]
    return None


def _probe_native_libs():
    """
    Use ctypes to find key native libraries to help locate missing components.

    Returns:
        list[str]: Identifiers of libraries not found
    """
    system = platform.system()
    targets = []

    if system == "Windows":
        targets = [
            ("pango", ["pango-1.0-0"]),
            ("gobject", ["gobject-2.0-0"]),
            ("gdk-pixbuf", ["gdk_pixbuf-2.0-0"]),
            ("cairo", ["cairo-2"]),
        ]
    else:
        targets = [
            ("pango", ["pango-1.0"]),
            ("gobject", ["gobject-2.0"]),
            ("gdk-pixbuf", ["gdk_pixbuf-2.0"]),
            ("cairo", ["cairo", "cairo-2"]),
        ]

    missing = []
    for key, variants in targets:
        found = any(ctypes_util.find_library(v) for v in variants)
        if not found:
            missing.append(key)
    return missing


def check_pango_available():
    """
    Check if Pango library is available

    Returns:
        tuple: (is_available: bool, message: str)
    """
    added_path = prepare_pango_environment()
    missing_native = _probe_native_libs()

    try:
        # Try importing weasyprint and initializing Pango
        from weasyprint import HTML
        from weasyprint.text.ffi import ffi, pango

        # Try to call Pango function to confirm library is available
        pango.pango_version()

        return True, "✓ Pango dependency check passed, PDF export functionality available"
    except OSError as e:
        # Pango library not installed or cannot be loaded
        error_msg = str(e)
        platform_instructions = _get_platform_specific_instructions()
        windows_hint = ""
        if platform.system() == "Windows":
            prefix = "Attempted to auto-add GTK path: "
            max_path_len = BOX_CONTENT_WIDTH - len(prefix)
            path_display = added_path or "Default path not found"
            if len(path_display) > max_path_len:
                path_display = path_display[: max_path_len - 3] + "..."
            windows_hint = _box_line(prefix + path_display)
            arch_note = _box_line("🔍 If still errors after install: Verify Python and GTK architecture match, then restart terminal")
        else:
            arch_note = ""

        missing_note = ""
        if missing_native:
            missing_str = ", ".join(missing_native)
            missing_note = _box_line(f"Unrecognized dependencies: {missing_str}")

        if 'gobject' in error_msg.lower() or 'pango' in error_msg.lower() or 'gdk' in error_msg.lower():
            box_top = "╔" + "═" * 64 + "╗\n"
            box_bottom = "╚" + "═" * 64 + "╝"
            return False, (
                box_top
                + _box_line("⚠️  PDF export dependencies missing")
                + _box_line()
                + _box_line("📄 PDF export functionality will be unavailable (other features unaffected)")
                + _box_line()
                + windows_hint
                + arch_note
                + missing_note
                + platform_instructions
                + _box_line()
                + _box_line("Documentation: docs/operations/troubleshooting.md#report-issues")
                + box_bottom
            )
        return False, f"⚠ PDF dependency loading failed: {error_msg}; Missing/unrecognized: {', '.join(missing_native) if missing_native else 'unknown'}"
    except ImportError as e:
        # weasyprint not installed
        return False, (
            "⚠ WeasyPrint not installed\n"
            "Solution: pip install weasyprint"
        )
    except Exception as e:
        # Other unknown errors
        return False, f"⚠ PDF dependency detection failed: {e}"


def log_dependency_status():
    """
    Log system dependency status
    """
    is_available, message = check_pango_available()

    if is_available:
        logger.success(message)
    else:
        logger.warning(message)
        logger.info("💡 Tip: PDF export feature requires Pango library support, but does not affect other system functions")
        logger.info("Installation instructions: docs/operations/setup.md and docs/operations/troubleshooting.md")

    return is_available


if __name__ == "__main__":
    # For standalone testing
    is_available, message = check_pango_available()
    print(message)
    sys.exit(0 if is_available else 1)
