"""Unit tests for scripts/insert_lesson.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO / "scripts" / "insert_lesson.py"
    spec = importlib.util.spec_from_file_location("insert_lesson", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


insert_lesson = _load_tool()


def _snapshot(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(root))] = path.read_bytes()
    return found


def _lesson(slug: str, title: str, part: str, prereqs: list[str], extra: str) -> str:
    prereq_block = ""
    if prereqs:
        lines = "\n".join(f"  - {item}" for item in prereqs)
        prereq_block = f"prereqs:\n{lines}\n"
    return (
        f"---\n"
        f"title: {title}\n"
        f"slug: {slug}\n"
        f'part: "{part}"\n'
        f"summary: {title} in one sentence.\n"
        f"minutes: 20\n"
        f"gpu: false\n"
        f"objectives:\n"
        f"  - State the idea in {title}.\n"
        f"  - Compute one small example by hand.\n"
        f"  - Name the function the checks call.\n"
        f"{prereq_block}"
        f"exercise: {slug}\n"
        f"---\n"
        f"\n"
        f"# {title}\n"
        f"\n"
        f"{extra}\n"
    )


def _meta(slug: str) -> str:
    return json.dumps(
        {
            "id": slug,
            "title": slug,
            "brief": "A short exercise.",
            "needs_gpu": False,
            "timeout": 30,
        },
        indent=2,
    ) + "\n"


class InsertLessonTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(self.id().replace(".", "_"))
        # Each test uses a fresh directory under the system temp via TemporaryDirectory.
        import tempfile

        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.part = "Part 1 — Foundations"

    def tearDown(self) -> None:
        self._dir.cleanup()

    def write_course(self, slugs: list[str], renames: dict | None = None) -> None:
        lessons = self.root / "content" / "lessons"
        exercises = self.root / "content" / "exercises"
        docs = self.root / "docs"
        lessons.mkdir(parents=True)
        exercises.mkdir(parents=True)
        docs.mkdir(parents=True)
        rows = ["| Slug | Title | Exercise delivers |", "|---|---|---|"]
        for index, slug in enumerate(slugs):
            title = slug.split("-", 1)[1].replace("-", " ").capitalize()
            prereqs = [slugs[index - 1]] if index else []
            link = ""
            if index + 1 < len(slugs):
                link = f"See [next](/lessons/{slugs[index + 1]})."
            (lessons / f"{slug}.md").write_text(
                _lesson(slug, title, self.part, prereqs, link),
                encoding="utf-8",
            )
            directory = exercises / slug
            directory.mkdir()
            (directory / "meta.json").write_text(_meta(slug), encoding="utf-8")
            rows.append(f"| `{slug}` | {title} | What {title} asks you to build |")
        (docs / "CURRICULUM.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        payload = renames if renames is not None else {"old-course": list(slugs)}
        (self.root / "content" / "renames.json").write_text(
            json.dumps(payload, indent=1) + "\n",
            encoding="utf-8",
        )

    def test_shift_rewrites_links_prereqs_meta_renames_and_curriculum(self) -> None:
        slugs = ["001-alpha", "002-beta", "003-gamma"]
        self.write_course(slugs)
        report = insert_lesson.apply(
            self.root, after="001-alpha", slug="added-lesson", count=1, dry_run=False
        )
        self.assertIn("Inserted 1 draft lesson(s) after 001-alpha.", report)

        lessons = self.root / "content" / "lessons"
        self.assertTrue((lessons / "001-alpha.md").is_file())
        self.assertFalse((lessons / "002-beta.md").is_file())
        self.assertTrue((lessons / "003-beta.md").is_file())
        self.assertTrue((lessons / "004-gamma.md").is_file())
        stub = (lessons / "002-added-lesson.md").read_text(encoding="utf-8")
        self.assertIn("draft: true", stub)
        self.assertIn(f'part: "{self.part}"', stub)
        self.assertIn("slug: 002-added-lesson", stub)
        self.assertIn("prereqs:\n  - 001-alpha", stub)

        # One pass: 002-beta becomes 003-beta, and 003-gamma becomes 004-gamma.
        # The new 003-beta must not be shifted again.
        alpha = (lessons / "001-alpha.md").read_text(encoding="utf-8")
        self.assertIn("/lessons/003-beta", alpha)
        self.assertNotIn("/lessons/002-beta", alpha)
        self.assertNotIn("/lessons/004-beta", alpha)

        beta = (lessons / "003-beta.md").read_text(encoding="utf-8")
        self.assertIn("slug: 003-beta", beta)
        self.assertIn("exercise: 003-beta", beta)
        self.assertIn("prereqs:\n  - 001-alpha", beta)
        self.assertIn("/lessons/004-gamma", beta)

        gamma = (lessons / "004-gamma.md").read_text(encoding="utf-8")
        self.assertIn("prereqs:\n  - 003-beta", gamma)
        self.assertIn("slug: 004-gamma", gamma)

        meta = json.loads(
            (self.root / "content" / "exercises" / "003-beta" / "meta.json").read_text()
        )
        self.assertEqual(meta["id"], "003-beta")
        self.assertFalse((self.root / "content" / "exercises" / "002-beta").exists())

        renames = json.loads(
            (self.root / "content" / "renames.json").read_text(encoding="utf-8")
        )
        self.assertEqual(renames["old-course"], ["001-alpha", "003-beta", "004-gamma"])
        self.assertEqual(renames["002-beta"], ["003-beta"])
        self.assertEqual(renames["003-gamma"], ["004-gamma"])
        self.assertEqual(
            list(renames),
            ["old-course", "002-beta", "003-gamma"],
        )

        curriculum = (self.root / "docs" / "CURRICULUM.md").read_text(encoding="utf-8")
        self.assertIn("| `002-added-lesson` | Added lesson |", curriculum)
        self.assertIn("| `003-beta` |", curriculum)
        self.assertIn("| `004-gamma` |", curriculum)
        self.assertNotIn("| `002-beta` |", curriculum)
        # The draft row sits immediately after the anchor.
        anchor_at = curriculum.index("| `001-alpha` |")
        draft_at = curriculum.index("| `002-added-lesson` |")
        beta_at = curriculum.index("| `003-beta` |")
        self.assertLess(anchor_at, draft_at)
        self.assertLess(draft_at, beta_at)

    def test_dry_run_changes_nothing(self) -> None:
        self.write_course(["001-alpha", "002-beta", "003-gamma"])
        before = _snapshot(self.root)
        report = insert_lesson.apply(
            self.root, after="002-beta", slug="added-lesson", count=1, dry_run=True
        )
        self.assertTrue(report.startswith("Dry run. No files change."))
        self.assertIn("003-gamma to 004-gamma", report)
        self.assertIn("003-added-lesson", report)
        self.assertEqual(before, _snapshot(self.root))

    def test_count_two_creates_distinct_stubs_and_shifts_by_two(self) -> None:
        self.write_course(["001-alpha", "002-beta", "003-gamma"])
        insert_lesson.apply(
            self.root, after="001-alpha", slug="added-lesson", count=2, dry_run=False
        )
        lessons = sorted(path.stem for path in (self.root / "content" / "lessons").glob("*.md"))
        self.assertEqual(
            lessons,
            [
                "001-alpha",
                "002-added-lesson",
                "003-added-lesson-2",
                "004-beta",
                "005-gamma",
            ],
        )
        second = (
            self.root / "content" / "lessons" / "003-added-lesson-2.md"
        ).read_text(encoding="utf-8")
        self.assertIn("draft: true", second)
        self.assertIn("prereqs:\n  - 002-added-lesson", second)
        self.assertIn(f'part: "{self.part}"', second)

    def test_project_marker_stays_on_the_number(self) -> None:
        self.write_course(["001-alpha", "002p-project", "003-gamma"])
        insert_lesson.apply(
            self.root, after="001-alpha", slug="added-lesson", count=1, dry_run=False
        )
        lessons = sorted(path.stem for path in (self.root / "content" / "lessons").glob("*.md"))
        self.assertIn("003p-project", lessons)
        self.assertNotIn("002p-project", lessons)
        self.assertIn("002-added-lesson", lessons)
        self.assertIn("004-gamma", lessons)
        project = (
            self.root / "content" / "lessons" / "003p-project.md"
        ).read_text(encoding="utf-8")
        self.assertIn("slug: 003p-project", project)
        meta = json.loads(
            (self.root / "content" / "exercises" / "003p-project" / "meta.json").read_text()
        )
        self.assertEqual(meta["id"], "003p-project")

    def test_second_insert_rewrites_stored_rename_values(self) -> None:
        self.write_course(
            ["001-alpha", "002-beta", "003-gamma"],
            renames={"old-course": ["002-beta", "003-gamma"]},
        )
        insert_lesson.apply(self.root, "001-alpha", "added-lesson", 1, dry_run=False)
        insert_lesson.apply(self.root, "003-beta", "more-lesson", 1, dry_run=False)
        renames = json.loads(
            (self.root / "content" / "renames.json").read_text(encoding="utf-8")
        )
        self.assertEqual(renames["old-course"], ["003-beta", "005-gamma"])
        self.assertEqual(renames["002-beta"], ["003-beta"])
        self.assertEqual(renames["003-gamma"], ["005-gamma"])
        self.assertEqual(renames["004-gamma"], ["005-gamma"])

    def test_missing_anchor_raises(self) -> None:
        self.write_course(["001-alpha"])
        with self.assertRaises(insert_lesson.InsertError) as caught:
            insert_lesson.apply(self.root, "009-missing", "added-lesson", 1, False)
        self.assertIn("009-missing", str(caught.exception))

    def test_overflow_writes_nothing(self) -> None:
        self.write_course(["998-early", "999-late"])
        before = _snapshot(self.root)
        with self.assertRaises(insert_lesson.InsertError):
            insert_lesson.apply(self.root, "998-early", "added-lesson", 1, False)
        self.assertEqual(before, _snapshot(self.root))

    def test_stub_passes_checker_and_published_minimum_stays(self) -> None:
        self.write_course(["001-alpha"])
        insert_lesson.apply(self.root, "001-alpha", "added-lesson", 1, dry_run=False)
        stub = "002-added-lesson"
        passed = self._check(stub)
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.assertIn("0 problem(s)", passed.stdout)

        lesson = self.root / "content" / "lessons" / f"{stub}.md"
        lesson.write_text(
            lesson.read_text(encoding="utf-8").replace("draft: true\n", ""),
            encoding="utf-8",
        )
        failed = self._check(stub)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("the contract asks for 900 or more", failed.stdout)

    def _check(self, slug: str) -> subprocess.CompletedProcess[str]:
        env = dict(**{key: value for key, value in __import__("os").environ.items()})
        env["CONTENT_DIR"] = str(self.root / "content")
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "check_content.py"), slug],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
