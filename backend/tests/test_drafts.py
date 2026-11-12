"""Draft lessons are visible in a checkout and hidden when the built frontend is served."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


def _lesson(slug: str, draft: bool) -> str:
    draft_line = "draft: true\n" if draft else ""
    return (
        "---\n"
        f"title: {slug}\n"
        f"slug: {slug}\n"
        'part: "Part 1 — Foundations"\n'
        f"summary: {slug} in one sentence.\n"
        "minutes: 20\n"
        "gpu: false\n"
        f"{draft_line}"
        "objectives:\n"
        "  - State one fact.\n"
        "  - State a second fact.\n"
        "  - State a third fact.\n"
        f"exercise: {slug}\n"
        "---\n"
        "\n"
        f"# {slug}\n"
        "\n"
        "Body.\n"
        "\n"
        "## First *section*\n"
        "\n"
        "```python\n"
        "## not a heading\n"
        "```\n"
        "\n"
        "### A `code` subsection with $x$\n"
        "\n"
        "## Second section\n"
    )


class DraftVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        lessons = self.root / "content" / "lessons"
        exercises = self.root / "content" / "exercises"
        lessons.mkdir(parents=True)
        exercises.mkdir(parents=True)
        published = exercises / "001-alpha"
        draft = exercises / "002-draft"
        published.mkdir()
        draft.mkdir()
        (lessons / "001-alpha.md").write_text(_lesson("001-alpha", draft=False), encoding="utf-8")
        (lessons / "002-draft.md").write_text(_lesson("002-draft", draft=True), encoding="utf-8")
        for directory, slug in ((published, "001-alpha"), (draft, "002-draft")):
            (directory / "meta.json").write_text(
                '{"id": "%s", "title": "%s", "brief": "Brief.", "needs_gpu": false, "timeout": 30}\n'
                % (slug, slug),
                encoding="utf-8",
            )
            (directory / "starter.py").write_text("raise NotImplementedError\n", encoding="utf-8")
        os.environ["DATA_DIR"] = str(self.root / "data")
        os.environ["CONTENT_DIR"] = str(self.root / "content")
        os.environ["FRONTEND_DIST"] = str(self.root / "missing-dist")
        from backend.app.config import get_settings
        from backend.app import curriculum

        self.get_settings = get_settings
        self.curriculum = curriculum
        get_settings.cache_clear()
        curriculum.reload_content()

    def tearDown(self) -> None:
        os.environ.pop("DATA_DIR", None)
        os.environ.pop("CONTENT_DIR", None)
        os.environ.pop("FRONTEND_DIST", None)
        self.get_settings.cache_clear()
        self.curriculum.reload_content()
        self._dir.cleanup()

    def _reload(self) -> None:
        self.get_settings.cache_clear()
        self.curriculum.reload_content()

    def test_drafts_are_visible_without_the_built_frontend(self) -> None:
        self.assertFalse(self.get_settings().frontend_dist.is_dir())
        self.assertTrue(self.curriculum.drafts_visible())
        slugs = [lesson["slug"] for lesson in self.curriculum.load_lessons()]
        self.assertEqual(slugs, ["001-alpha", "002-draft"])
        self.assertTrue(self.curriculum.load_lessons()[1]["draft"])
        self.assertIsNotNone(self.curriculum.get_lesson("002-draft"))
        self.assertIn("002-draft", self.curriculum.load_exercises())

    def test_drafts_are_hidden_when_the_built_frontend_exists(self) -> None:
        dist = self.root / "frontend-dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!DOCTYPE html>", encoding="utf-8")
        os.environ["FRONTEND_DIST"] = str(dist)
        self._reload()

        self.assertTrue(self.get_settings().frontend_dist.is_dir())
        self.assertFalse(self.curriculum.drafts_visible())
        slugs = [lesson["slug"] for lesson in self.curriculum.load_lessons()]
        self.assertEqual(slugs, ["001-alpha"])
        self.assertIsNone(self.curriculum.get_lesson("002-draft"))
        exercises = self.curriculum.load_exercises()
        self.assertIn("001-alpha", exercises)
        self.assertNotIn("002-draft", exercises)

    def test_lessons_list_plain_text_headings(self) -> None:
        listed = {lesson["slug"]: lesson for lesson in self.curriculum.lesson_list()}
        self.assertEqual(
            listed["001-alpha"]["headings"],
            [
                {"text": "First section", "level": 2},
                {"text": "A code subsection with x", "level": 3},
                {"text": "Second section", "level": 2},
            ],
        )
        self.assertNotIn("body", listed["001-alpha"])


if __name__ == "__main__":
    unittest.main()
