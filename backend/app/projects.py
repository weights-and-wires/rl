"""Loads hands-on projects from `content/projects/`.

A project is one directory, named by its slug, that holds:

- `project.md`: Markdown with YAML front matter, the write-up the reader sees.
- `slides/` (optional): a slide deck. `deck.json` names the slides in order,
  `deck.css` styles them, and each `<id>.html` holds one `<section id="<id>">`
  on a 1920 by 1080 canvas. An `<aside>` as the section's last child holds the
  speaker notes.

Projects are newest first, by the `date` in their front matter. A project
marked `draft: true` follows the lesson rule in `curriculum.drafts_visible`.
The format is described in `docs/CONTENT_CONTRACT.md`.
"""

from __future__ import annotations

import html as html_text
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import frontmatter

from .config import get_settings
from .curriculum import _headings, _is_draft, _text_list, drafts_visible

# The notes are the section's last child, so they sit just before its close.
_NOTES = re.compile(r"<aside\b[^>]*>(.*?)</aside>\s*(</section>\s*)$", re.DOTALL | re.IGNORECASE)


def _projects_dir() -> Path:
    return get_settings().content_dir / "projects"


def _links(value: Any) -> list[dict[str, str]]:
    """Front-matter `links` as `{"label", "url"}` pairs, dropping malformed ones."""
    if not isinstance(value, list):
        return []
    return [
        {"label": str(item["label"]), "url": str(item["url"])}
        for item in value
        if isinstance(item, dict) and item.get("label") and item.get("url")
    ]


def _deck_index(directory: Path) -> dict[str, Any] | None:
    path = directory / "slides" / "deck.json"
    if not path.is_file():
        return None
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    order = index.get("order")
    if not isinstance(order, list):
        return None
    return {"title": str(index.get("title", "")), "order": [str(slide) for slide in order]}


@lru_cache
def load_projects() -> list[dict[str, Any]]:
    """Every project the reader should see, newest first, with its body."""
    root = _projects_dir()
    if not root.is_dir():
        return []
    projects: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        source = directory / "project.md"
        if not source.is_file():
            continue
        post = frontmatter.load(source)
        meta = dict(post.metadata)
        if _is_draft(meta) and not drafts_visible():
            continue
        deck = _deck_index(directory)
        projects.append(
            {
                "slug": directory.name,
                "title": str(meta.get("title", directory.name)),
                "summary": str(meta.get("summary", "")),
                "date": str(meta.get("date", "")),
                "status": str(meta.get("status", "")),
                "tags": _text_list(meta.get("tags", [])),
                "lessons": _text_list(meta.get("lessons", [])),
                "links": _links(meta.get("links", [])),
                "slides": {"title": deck["title"], "count": len(deck["order"])} if deck else None,
                "draft": _is_draft(meta),
                "headings": _headings(post.content),
                "body": post.content,
            }
        )
    projects.sort(key=lambda project: project["date"], reverse=True)
    return projects


def project_list() -> list[dict[str, Any]]:
    """Project metadata without the body, for the sidebar and the index page."""
    return [{k: v for k, v in project.items() if k != "body"} for project in load_projects()]


def get_project(slug: str) -> dict[str, Any] | None:
    return next((project for project in load_projects() if project["slug"] == slug), None)


def get_slides(slug: str) -> dict[str, Any] | None:
    """A project's deck: its title, shared CSS, and each slide with its notes.

    Only slugs that `load_projects` found and ids that `deck.json` lists are
    read, so a request can't reach a file outside the deck.
    """
    if get_project(slug) is None:
        return None
    directory = _projects_dir() / slug / "slides"
    deck = _deck_index(directory.parent)
    if deck is None:
        return None
    css_path = directory / "deck.css"
    slides: list[dict[str, str]] = []
    for slide_id in deck["order"]:
        path = directory / f"{slide_id}.html"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", slide_id) or not path.is_file():
            continue
        html = path.read_text(encoding="utf-8").strip()
        notes = ""
        match = _NOTES.search(html)
        if match:
            notes = html_text.unescape(re.sub(r"\s+", " ", match.group(1)).strip())
            html = html[: match.start()] + match.group(2)
        slides.append({"id": slide_id, "html": html.strip(), "notes": notes})
    return {
        "title": deck["title"],
        "css": css_path.read_text(encoding="utf-8") if css_path.is_file() else "",
        "slides": slides,
    }


def reload_projects() -> None:
    load_projects.cache_clear()
