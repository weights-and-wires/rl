"""Projects load from content/projects/, newest first, with their slides and notes."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _project(title: str, date: str, draft: bool = False) -> str:
    return (
        "---\n"
        f"title: {title}\n"
        f"summary: {title} in one sentence.\n"
        f"date: {date}\n"
        "status: Finished\n"
        f"{'draft: true' + chr(10) if draft else ''}"
        "lessons:\n"
        "  - 001-alpha\n"
        "links:\n"
        "  - label: Code\n"
        "    url: https://example.com/code\n"
        "  - label: missing url\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        "## The goal\n"
        "\n"
        "Body.\n"
    )


class ProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        projects = self.root / "content" / "projects"
        (self.root / "content" / "lessons").mkdir(parents=True)

        older = projects / "older"
        older.mkdir(parents=True)
        (older / "project.md").write_text(_project("Older", "2026-01-02"), encoding="utf-8")

        newer = projects / "newer"
        slides = newer / "slides"
        slides.mkdir(parents=True)
        (newer / "project.md").write_text(_project("Newer", "2026-11-15"), encoding="utf-8")
        (slides / "deck.json").write_text(
            '{"title": "A deck", "order": ["cover", "gone", "../escape", "second"]}', encoding="utf-8"
        )
        (slides / "deck.css").write_text("section { color: red; }", encoding="utf-8")
        (slides / "cover.html").write_text(
            '<section id="cover"><h1>Hello</h1>\n<aside>Say  hi &amp;\n wave.</aside>\n</section>\n',
            encoding="utf-8",
        )
        (slides / "second.html").write_text('<section id="second"><h2>Two</h2></section>', encoding="utf-8")

        hidden = projects / "hidden"
        hidden.mkdir()
        (hidden / "project.md").write_text(_project("Hidden", "2026-12-01", draft=True), encoding="utf-8")
        (projects / "no-write-up").mkdir()

        os.environ["DATA_DIR"] = str(self.root / "data")
        os.environ["CONTENT_DIR"] = str(self.root / "content")
        os.environ["FRONTEND_DIST"] = str(self.root / "missing-dist")
        from backend.app import projects as loader
        from backend.app.config import get_settings

        self.get_settings = get_settings
        self.projects = loader
        self._reload()

    def tearDown(self) -> None:
        for key in ("DATA_DIR", "CONTENT_DIR", "FRONTEND_DIST"):
            os.environ.pop(key, None)
        self._reload()
        self._dir.cleanup()

    def _reload(self) -> None:
        self.get_settings.cache_clear()
        self.projects.reload_projects()

    def test_projects_are_newest_first_and_skip_folders_without_a_write_up(self) -> None:
        slugs = [project["slug"] for project in self.projects.project_list()]
        self.assertEqual(slugs, ["hidden", "newer", "older"])

    def test_drafts_are_hidden_when_the_built_frontend_exists(self) -> None:
        dist = self.root / "dist"
        dist.mkdir()
        os.environ["FRONTEND_DIST"] = str(dist)
        self._reload()
        slugs = [project["slug"] for project in self.projects.project_list()]
        self.assertEqual(slugs, ["newer", "older"])
        self.assertIsNone(self.projects.get_project("hidden"))
        self.assertIsNone(self.projects.get_slides("hidden"))

    def test_metadata_has_no_body_and_keeps_only_whole_links(self) -> None:
        listed = {project["slug"]: project for project in self.projects.project_list()}
        newer = listed["newer"]
        self.assertNotIn("body", newer)
        self.assertEqual(newer["date"], "2026-11-15")
        self.assertEqual(newer["lessons"], ["001-alpha"])
        self.assertEqual(newer["links"], [{"label": "Code", "url": "https://example.com/code"}])
        self.assertEqual(newer["slides"], {"title": "A deck", "count": 4})
        self.assertIsNone(listed["older"]["slides"])
        self.assertIn("# Newer", self.projects.get_project("newer")["body"])

    def test_slides_split_off_their_notes_and_skip_missing_or_unsafe_ids(self) -> None:
        deck = self.projects.get_slides("newer")
        self.assertEqual(deck["title"], "A deck")
        self.assertEqual(deck["css"], "section { color: red; }")
        self.assertEqual([slide["id"] for slide in deck["slides"]], ["cover", "second"])
        cover = deck["slides"][0]
        self.assertEqual(cover["notes"], "Say hi & wave.")
        self.assertEqual(cover["html"], '<section id="cover"><h1>Hello</h1>\n</section>')
        self.assertEqual(deck["slides"][1]["notes"], "")

    def test_unknown_projects_and_projects_without_slides_return_none(self) -> None:
        self.assertIsNone(self.projects.get_project("../content"))
        self.assertIsNone(self.projects.get_slides("missing"))
        self.assertIsNone(self.projects.get_slides("older"))


class RepositoryProjectTests(unittest.TestCase):
    """The projects shipped in this repository pass the content checker."""

    def test_shipped_projects_pass_the_checker(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("check_content", REPO / "scripts" / "check_content.py")
        assert spec and spec.loader
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        problems: list[str] = []
        for directory in sorted((REPO / "content" / "projects").iterdir()):
            if directory.is_dir():
                checker.check_project(directory, problems)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
