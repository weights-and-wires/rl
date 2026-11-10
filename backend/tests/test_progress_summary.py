"""The hub's progress summary and its path-scoped CORS headers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

HUB = "https://courses.ifkash.dev"
SECRET = "test-proxy-secret"
SIGNED_IN = {"Remote-User": "reader", "X-SSO-Proxy": SECRET}


def _lesson(slug: str, draft: bool = False) -> str:
    draft_line = "draft: true\n" if draft else ""
    return (
        "---\n"
        f"title: Title {slug}\n"
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
    )


class ProgressSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        lessons = self.root / "content" / "lessons"
        exercises = self.root / "content" / "exercises"
        lessons.mkdir(parents=True)
        exercises.mkdir(parents=True)
        for slug, draft in (("001-a", False), ("002-b", False), ("003-c", False), ("004-d", True)):
            (lessons / f"{slug}.md").write_text(_lesson(slug, draft), encoding="utf-8")
            directory = exercises / slug
            directory.mkdir()
            (directory / "meta.json").write_text(
                '{"id": "%s", "title": "%s", "brief": "Brief.", "needs_gpu": false, "timeout": 30}\n'
                % (slug, slug),
                encoding="utf-8",
            )
            (directory / "starter.py").write_text("raise NotImplementedError\n", encoding="utf-8")
        # A built frontend hides draft lessons, as in production.
        dist = self.root / "frontend-dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!DOCTYPE html>", encoding="utf-8")
        self._env = {
            "DATA_DIR": str(self.root / "data"),
            "CONTENT_DIR": str(self.root / "content"),
            "FRONTEND_DIST": str(dist),
            "SSO_PROXY_SECRET": SECRET,
        }
        os.environ.update(self._env)
        from fastapi.testclient import TestClient
        from backend.app import curriculum, db
        from backend.app.config import get_settings
        from backend.app.main import app

        self.get_settings = get_settings
        self.curriculum = curriculum
        self.db = db
        get_settings.cache_clear()
        curriculum.reload_content()
        db.init()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        for key in self._env:
            os.environ.pop(key, None)
        self.get_settings.cache_clear()
        self.curriculum.reload_content()
        self._dir.cleanup()

    def _get(self, path: str = "/api/progress/summary", origin: str | None = None, signed_in=True):
        headers = dict(SIGNED_IN) if signed_in else {}
        if origin:
            headers["Origin"] = origin
        return self.client.get(path, headers=headers)

    def test_empty_progress(self) -> None:
        body = self._get().json()
        self.assertEqual(body, {"done": 0, "total": 3, "last": None})

    def test_counts_solved_and_picks_newest_row(self) -> None:
        self.db.set_progress("reader", "001-a", "solved")
        self.db.set_progress("reader", "002-b", "started")
        self.db.set_progress("reader", "004-d", "solved")  # draft, not served
        self.db.set_progress("reader", "gone", "solved")  # not in the list
        self.db.set_progress("other", "003-c", "solved")  # another reader
        with self.db.session() as connection:
            connection.execute("UPDATE progress SET updated_at = 100 WHERE slug = '001-a'")
            connection.execute("UPDATE progress SET updated_at = 200 WHERE slug = '002-b'")
            connection.execute("UPDATE progress SET updated_at = 900 WHERE slug IN ('004-d', 'gone')")
        response = self._get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"done": 1, "total": 3, "last": {"slug": "002-b", "title": "Title 002-b"}},
        )

    def test_signed_out_is_401_with_cors_for_the_hub(self) -> None:
        response = self._get(origin=HUB, signed_in=False)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["access-control-allow-origin"], HUB)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")
        self.assertIn("Origin", response.headers["vary"])

    def test_signed_in_response_carries_cors_for_the_hub(self) -> None:
        response = self._get(origin=HUB)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], HUB)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")

    def test_other_origin_gets_no_allow_origin(self) -> None:
        for origin in ("https://evil.example", HUB + ".evil.example", "http://courses.ifkash.dev"):
            response = self._get(origin=origin)
            self.assertNotIn("access-control-allow-origin", response.headers)
            self.assertNotIn("access-control-allow-credentials", response.headers)
            self.assertIn("Origin", response.headers["vary"])

    def test_no_origin_gets_vary_only(self) -> None:
        response = self._get()
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertIn("Origin", response.headers["vary"])

    def test_other_paths_get_no_cors_headers(self) -> None:
        response = self._get("/api/health", origin=HUB)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertNotIn("access-control-allow-credentials", response.headers)
        self.assertNotIn("vary", response.headers)


if __name__ == "__main__":
    unittest.main()
