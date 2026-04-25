"""
Chapter JSON persistence and manifest management.

Each chapter is written to a raw file immediately during streaming generation, then written to
formatted chapter.json after validation, with metadata recorded in the manifest for subsequent stitching.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional


@dataclass
class ChapterRecord:
    """
    Chapter metadata recorded in the manifest.

    This structure is used to track each chapter's status, file location,
    and potential error list in `manifest.json`, facilitating reading by frontend or debugging tools.
    """

    chapter_id: str
    slug: str
    title: str
    order: int
    status: str
    files: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, object]:
        """Convert the record to a serialized dictionary for easy writing to manifest.json"""
        return {
            "chapterId": self.chapter_id,
            "slug": self.slug,
            "title": self.title,
            "order": self.order,
            "status": self.status,
            "files": self.files,
            "errors": self.errors,
            "updatedAt": self.updated_at,
        }


class ChapterStorage:
    """
    Chapter JSON writer and manifest manager.

    Responsible for:
        - Creating independent run directories and manifest snapshots for each report;
        - Writing to `stream.raw` immediately during chapter streaming generation;
        - Persisting `chapter.json` and updating manifest status after validation passes.
    """

    def __init__(self, base_dir: str):
        """
        Create chapter storage.

        Args:
            base_dir: Root path for all output run directories
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifests: Dict[str, Dict[str, object]] = {}

    # ======== Session and Manifest ========

    def start_session(self, report_id: str, metadata: Dict[str, object]) -> Path:
        """
        Create independent chapter output directory and manifest for this report.

        Also writes global metadata to `manifest.json` for rendering/debugging queries.

        Args:
            report_id: Task ID.
            metadata: Report metadata (title, theme, etc.).

        Returns:
            Path: Newly created run directory.
        """
        run_dir = self.base_dir / report_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "reportId": report_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata,
            "chapters": [],
        }
        self._manifests[self._key(run_dir)] = manifest
        self._write_manifest(run_dir, manifest)
        return run_dir

    def begin_chapter(self, run_dir: Path, chapter_meta: Dict[str, object]) -> Path:
        """
        Create chapter subdirectory and mark as streaming status in manifest.

        Generates `order-slug` style subdirectory and pre-registers raw file path.

        Args:
            run_dir: Session root directory.
            chapter_meta: Metadata containing chapterId/title/slug/order.

        Returns:
            Path: Chapter directory.
        """
        slug_value = str(
            chapter_meta.get("slug") or chapter_meta.get("chapterId") or "section"
        )
        chapter_dir = self._chapter_dir(
            run_dir,
            slug_value,
            int(chapter_meta.get("order", 0)),
        )
        record = ChapterRecord(
            chapter_id=str(chapter_meta.get("chapterId")),
            slug=slug_value,
            title=str(chapter_meta.get("title")),
            order=int(chapter_meta.get("order", 0)),
            status="streaming",
            files={"raw": str(self._raw_stream_path(chapter_dir).relative_to(run_dir))},
        )
        self._upsert_record(run_dir, record)
        return chapter_dir

    def persist_chapter(
        self,
        run_dir: Path,
        chapter_meta: Dict[str, object],
        payload: Dict[str, object],
        errors: Optional[List[str]] = None,
    ) -> Path:
        """
        Write final JSON and update manifest status after chapter streaming generation completes.

        If validation fails, error information is written to manifest for frontend display.

        Args:
            run_dir: Session root directory.
            chapter_meta: Chapter metadata.
            payload: Validated chapter JSON.
            errors: Optional error list for marking invalid status.

        Returns:
            Path: Final `chapter.json` file path.
        """
        slug_value = str(
            chapter_meta.get("slug") or chapter_meta.get("chapterId") or "section"
        )
        chapter_dir = self._chapter_dir(
            run_dir,
            slug_value,
            int(chapter_meta.get("order", 0)),
        )
        final_path = chapter_dir / "chapter.json"
        final_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        record = ChapterRecord(
            chapter_id=str(chapter_meta.get("chapterId")),
            slug=slug_value,
            title=str(chapter_meta.get("title")),
            order=int(chapter_meta.get("order", 0)),
            status="ready" if not errors else "invalid",
            files={
                "raw": str(self._raw_stream_path(chapter_dir).relative_to(run_dir)),
                "json": str(final_path.relative_to(run_dir)),
            },
            errors=errors or [],
        )
        self._upsert_record(run_dir, record)
        return final_path

    def load_chapters(self, run_dir: Path) -> List[Dict[str, object]]:
        """
        Read all chapter.json from specified run directory and return sorted by order.

        Commonly used by DocumentComposer to stitch multiple chapters into complete IR.

        Args:
            run_dir: Session root directory.

        Returns:
            list[dict]: Chapter payload list.
        """
        payloads: List[Dict[str, object]] = []
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            chapter_path = child / "chapter.json"
            if not chapter_path.exists():
                continue
            try:
                payload = json.loads(chapter_path.read_text(encoding="utf-8"))
                payloads.append(payload)
            except json.JSONDecodeError:
                continue
        payloads.sort(key=lambda x: x.get("order", 0))
        return payloads

    # ======== File Operations ========

    @contextmanager
    def capture_stream(self, chapter_dir: Path) -> Generator:
        """
        Write streaming output to raw file in real-time.

        Exposes file handle through contextmanager, simplifying chapter node write logic.

        Args:
            chapter_dir: Current chapter directory.

        Returns:
            Generator[TextIO]: File object used as context manager.
        """
        raw_path = self._raw_stream_path(chapter_dir)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as fp:
            yield fp

    # ======== Internal Utilities ========

    def _chapter_dir(self, run_dir: Path, slug: str, order: int) -> Path:
        """Generate stable directory based on slug/order, ensuring chapters are stored separately and sortable."""
        safe_slug = self._safe_slug(slug)
        folder = f"{order:03d}-{safe_slug}"
        path = run_dir / folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_slug(self, slug: str) -> str:
        """Remove dangerous characters to avoid generating illegal folder names."""
        slug = slug.replace(" ", "-").replace("/", "-")
        return slug or "section"

    def _raw_stream_path(self, chapter_dir: Path) -> Path:
        """Return raw file path corresponding to a chapter's streaming output."""
        return chapter_dir / "stream.raw"

    def _key(self, run_dir: Path) -> str:
        """Parse run directory as dictionary cache key to avoid repeated disk reads."""
        return str(run_dir.resolve())

    def _manifest_path(self, run_dir: Path) -> Path:
        """Get actual file path for manifest.json."""
        return run_dir / "manifest.json"

    def _write_manifest(self, run_dir: Path, manifest: Dict[str, object]):
        """Write complete manifest snapshot from memory back to disk."""
        self._manifest_path(run_dir).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_manifest(self, run_dir: Path) -> Dict[str, object]:
        """
        Read existing manifest from disk.

        Can be used to restore context when process restarts or multiple instances write to disk.
        """
        manifest_path = self._manifest_path(run_dir)
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"reportId": run_dir.name, "chapters": []}

    def _upsert_record(self, run_dir: Path, record: ChapterRecord):
        """
        Update or append chapter records in manifest, ensuring consistent order.

        Internally auto-sorts and writes back to cache+disk.
        """
        key = self._key(run_dir)
        manifest = self._manifests.get(key) or self._read_manifest(run_dir)
        chapters: List[Dict[str, object]] = manifest.get("chapters", [])
        chapters = [c for c in chapters if c.get("chapterId") != record.chapter_id]
        chapters.append(record.to_dict())
        chapters.sort(key=lambda x: x.get("order", 0))
        manifest["chapters"] = chapters
        manifest.setdefault("updatedAt", datetime.utcnow().isoformat() + "Z")
        self._manifests[key] = manifest
        self._write_manifest(run_dir, manifest)


__all__ = ["ChapterStorage", "ChapterRecord"]
