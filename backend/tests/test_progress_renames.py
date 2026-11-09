"""A progress row stored under a pre-shift slug follows renames.json."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO / "scripts" / "insert_lesson.py"
    spec = importlib.util.spec_from_file_location("insert_lesson_progress", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


insert_lesson = _load_tool()


class ProgressRenameTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self._write_course()
        os.environ["DATA_DIR"] = str(self.root / "data")
        os.environ["CONTENT_DIR"] = str(self.root / "content")
        os.environ["FRONTEND_DIST"] = str(self.root / "missing-dist")
        from backend.app.config import get_settings
        from backend.app import curriculum

        get_settings.cache_clear()
        curriculum.reload_content()
        self.get_settings = get_settings
        self.curriculum = curriculum
        from backend.app import db

        self.db = db
        self.assertTrue(str(get_settings().db_path).startswith(str(self.root)))
        db.init()

    def tearDown(self) -> None:
        os.environ.pop("DATA_DIR", None)
        os.environ.pop("CONTENT_DIR", None)
        os.environ.pop("FRONTEND_DIST", None)
        self.get_settings.cache_clear()
        self.curriculum.reload_content()
        self._dir.cleanup()

    def _write_course(self) -> None:
        lessons = self.root / "content" / "lessons"
        exercises = self.root / "content" / "exercises"
        docs = self.root / "docs"
        lessons.mkdir(parents=True)
        exercises.mkdir(parents=True)
        docs.mkdir(parents=True)
        for slug, prereq in (("001-alpha", None), ("002-beta", "001-alpha")):
            prereq_block = f"prereqs:\n  - {prereq}\n" if prereq else ""
            (lessons / f"{slug}.md").write_text(
                "---\n"
                f"title: {slug}\n"
                f"slug: {slug}\n"
                'part: "Part 1 — Foundations"\n'
                f"summary: {slug} in one sentence.\n"
                "minutes: 20\n"
                "gpu: false\n"
                "objectives:\n"
                "  - State one fact.\n"
                "  - State a second fact.\n"
                "  - State a third fact.\n"
                f"{prereq_block}"
                f"exercise: {slug}\n"
                "---\n"
                "\n"
                f"# {slug}\n",
                encoding="utf-8",
            )
            directory = exercises / slug
            directory.mkdir()
            (directory / "meta.json").write_text(
                json.dumps({"id": slug, "title": slug, "brief": "Brief.", "needs_gpu": False, "timeout": 30})
                + "\n",
                encoding="utf-8",
            )
        (docs / "CURRICULUM.md").write_text(
            "\n".join(
                [
                    "| Slug | Title | Exercise delivers |",
                    "|---|---|---|",
                    "| `001-alpha` | Alpha | Alpha |",
                    "| `002-beta` | Beta | Beta |",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (self.root / "content" / "renames.json").write_text(
            json.dumps({"old-beta": ["002-beta"]}, indent=1) + "\n",
            encoding="utf-8",
        )

    def test_preshift_slug_resolves_to_the_new_slug(self) -> None:
        self.db.set_progress("ada", "002-beta", "solved")
        insert_lesson.apply(self.root, "001-alpha", "added-lesson", 1, dry_run=False)
        self.get_settings.cache_clear()
        self.curriculum.reload_content()

        mapping = self.curriculum.renames()
        self.assertEqual(mapping["002-beta"], ["003-beta"])
        carried = self.db.carry_progress(mapping)
        self.assertEqual(carried, 1)
        progress = self.db.get_progress("ada")
        self.assertEqual(progress["003-beta"]["status"], "solved")
        self.assertNotIn("002-beta", progress)
        self.assertEqual(self.db.carry_progress(mapping), 0)

    def test_older_progress_row_follows_rewritten_rename_value(self) -> None:
        self.db.set_progress("ada", "old-beta", "solved")
        insert_lesson.apply(self.root, "001-alpha", "added-lesson", 1, dry_run=False)
        self.get_settings.cache_clear()
        self.curriculum.reload_content()

        mapping = self.curriculum.renames()
        self.assertEqual(mapping["old-beta"], ["003-beta"])
        self.db.carry_progress(mapping)
        progress = self.db.get_progress("ada")
        self.assertEqual(progress["003-beta"]["status"], "solved")
        self.assertNotIn("old-beta", progress)
        self.assertNotIn("002-beta", progress)


if __name__ == "__main__":
    unittest.main()
