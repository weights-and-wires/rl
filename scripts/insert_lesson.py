#!/usr/bin/env python3
"""Insert draft lessons and shift every later chapter.

    python3 scripts/insert_lesson.py --after 030-where-the-gradient-comes-from \\
        --slug added-lesson
    python3 scripts/insert_lesson.py --after 030-where-the-gradient-comes-from \\
        --slug added-lesson --count 2
    python3 scripts/insert_lesson.py --after 030-where-the-gradient-comes-from \\
        --slug added-lesson --dry-run

The lesson named by ``--after`` stays put. Every later lesson file and exercise
directory moves up by ``--count`` (the default is 1). A lesson id is three
digits, an optional project marker ``p``, and a slug. This course's files are
``NNN-slug``. A file that does carry ``p`` keeps it when the number increments,
so ``012p-name`` becomes ``013p-name``.

The new draft occupies the numbers that opened up. It is marked ``draft: true``
and copies ``part`` from the lesson that stayed put. Reader progress stays
attached: ``content/renames.json`` gains one entry from each old slug to its
new slug, and any stored value that still names a shifted slug is rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Three digits, an optional project marker, then the slug. This course does
# not use the marker; the shifter keeps it when a file has one.
ID_RE = re.compile(
    r"^(?P<number>\d{3})(?P<project>p?)-(?P<name>[a-z0-9]+(?:-[a-z0-9]+)*)$"
)
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NUMBER = 999


class InsertError(Exception):
    """The insertion cannot be applied without losing a chapter or a rename."""


@dataclass(frozen=True)
class LessonId:
    number: int
    project: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.number:03d}{self.project}-{self.name}"

    def shifted(self, count: int) -> LessonId:
        return LessonId(self.number + count, self.project, self.name)


@dataclass(frozen=True)
class Stub:
    slug: str
    title: str
    part: str
    prereq: str
    prereq_title: str


@dataclass(frozen=True)
class Plan:
    root: Path
    anchor: str
    anchor_title: str
    part: str
    count: int
    shifts: tuple[tuple[str, str], ...]
    stubs: tuple[Stub, ...]


def parse_id(slug: str) -> LessonId | None:
    """Parse ``NNN-slug`` or ``NNNp-slug``. Return None when it does not match."""
    match = ID_RE.fullmatch(slug)
    if match is None:
        return None
    return LessonId(
        number=int(match.group("number")),
        project=match.group("project"),
        name=match.group("name"),
    )


def title_from_name(name: str) -> str:
    """Sentence-case title derived from a slug name."""
    words = name.split("-")
    head = words[0][:1].upper() + words[0][1:]
    return " ".join([head, *words[1:]])


def draft_names(slug: str, count: int) -> list[str]:
    """Distinct slug names for ``count`` drafts, derived from ``slug``."""
    if count == 1:
        return [slug]
    return [slug if index == 1 else f"{slug}-{index}" for index in range(1, count + 1)]


def replace_slugs(text: str, mapping: dict[str, str]) -> str:
    """Replace shifted slugs in one pass.

    A sequential replace would turn ``002`` into ``003`` and then into ``004``
    when both steps are in the map. One pass leaves each old slug at its own
    new slug.
    """
    if not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9-])(?:"
        + "|".join(re.escape(key) for key in keys)
        + r")(?![A-Za-z0-9-])"
    )
    return pattern.sub(lambda match: mapping[match.group(0)], text)


def apply_renames(
    existing: dict[str, list[str]], mapping: dict[str, str]
) -> dict[str, list[str]]:
    """Rewrite stored slugs, then append one entry per shifted lesson.

    Keys stay as they are, because a progress row is stored under the old
    slug. Values that still name a shifted slug move to the new slug, so an
    older row keeps resolving. Each shifted lesson also gets its own entry.
    """
    updated: dict[str, list[str]] = {}
    for key, values in existing.items():
        if not isinstance(values, list):
            raise InsertError(f"renames.json entry {key} is not a list")
        updated[str(key)] = [mapping.get(str(value), str(value)) for value in values]
    for old, new in mapping.items():
        current = updated.get(old)
        if current is None:
            updated[old] = [new]
        elif new not in current:
            updated[old] = [*current, new]
    return updated


def _read_frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        raise InsertError("a lesson does not start with YAML front matter")
    try:
        end = text.index("\n---\n", 3)
    except ValueError as exc:
        raise InsertError("a lesson has no closing front-matter marker") from exc
    try:
        import yaml
    except ImportError as exc:
        raise InsertError(
            "insert_lesson needs PyYAML. pip install pyyaml"
        ) from exc
    loaded = yaml.safe_load(text[4:end]) or {}
    if not isinstance(loaded, dict):
        raise InsertError("a lesson's front matter is not a mapping")
    return loaded


def _lesson_paths(content: Path) -> list[Path]:
    directory = content / "lessons"
    if not directory.is_dir():
        raise InsertError(f"missing {directory}")
    return sorted(path for path in directory.glob("*.md") if path.is_file())


def _exercise_paths(content: Path) -> list[Path]:
    directory = content / "exercises"
    if not directory.is_dir():
        raise InsertError(f"missing {directory}")
    return sorted(path for path in directory.iterdir() if path.is_dir() and not path.name.startswith("."))


def _require_id(stem: str) -> LessonId:
    parsed = parse_id(stem)
    if parsed is None:
        raise InsertError(
            f"{stem} is not a lesson id. Use three digits, an optional p, and a slug."
        )
    return parsed


def build_plan(root: Path, after: str, slug: str, count: int) -> Plan:
    """Read the tree and decide every rename. Write nothing."""
    if count < 1:
        raise InsertError("--count must be at least 1")
    if not SLUG_RE.fullmatch(slug) or parse_id(slug) is not None:
        raise InsertError(
            "--slug takes the name only, in lowercase letters, digits, and "
            "hyphens, for example added-lesson."
        )
    for name in draft_names(slug, count):
        if not SLUG_RE.fullmatch(name):
            raise InsertError(f"derived slug {name} is not a valid name")

    content = root / "content"
    lesson_paths = _lesson_paths(content)
    exercise_paths = _exercise_paths(content)
    lesson_stems = {path.stem for path in lesson_paths}
    exercise_stems = {path.name for path in exercise_paths}
    if after not in lesson_stems:
        raise InsertError(f"no lesson named {after}")

    stems = lesson_stems | exercise_stems
    for stem in sorted(stems):
        _require_id(stem)

    anchor = _require_id(after)
    anchor_text = (content / "lessons" / f"{after}.md").read_text(encoding="utf-8")
    anchor_meta = _read_frontmatter(anchor_text)
    part = anchor_meta.get("part")
    if not isinstance(part, str) or not part.strip():
        raise InsertError(f"{after} has no part string to copy onto the draft")
    anchor_title = anchor_meta.get("title")
    if not isinstance(anchor_title, str) or not anchor_title.strip():
        anchor_title = after

    curriculum_path = root / "docs" / "CURRICULUM.md"
    if not curriculum_path.is_file():
        raise InsertError("missing docs/CURRICULUM.md")
    curriculum = curriculum_path.read_text(encoding="utf-8")
    if f"| `{after}` |" not in curriculum:
        raise InsertError(f"docs/CURRICULUM.md has no row for {after}")

    later = [stem for stem in sorted(stems) if stem > after]
    shifts: list[tuple[str, str]] = []
    for stem in later:
        moved = _require_id(stem).shifted(count)
        if moved.number > MAX_NUMBER:
            raise InsertError(
                f"shifting {stem} by {count} needs a number above {MAX_NUMBER}"
            )
        shifts.append((stem, moved.slug))

    stubs: list[Stub] = []
    previous = after
    previous_title = anchor_title
    for index, name in enumerate(draft_names(slug, count), start=1):
        number = anchor.number + index
        if number > MAX_NUMBER:
            raise InsertError(
                f"the draft at position {index} needs a number above {MAX_NUMBER}"
            )
        stub_slug = f"{number:03d}-{name}"
        stub_title = title_from_name(name)
        stubs.append(
            Stub(
                slug=stub_slug,
                title=stub_title,
                part=part,
                prereq=previous,
                prereq_title=previous_title,
            )
        )
        previous = stub_slug
        previous_title = stub_title

    unshifted = [stem for stem in stems if stem <= after]
    final = unshifted + [new for _, new in shifts] + [stub.slug for stub in stubs]
    duplicates = sorted({slug for slug in final if final.count(slug) > 1})
    if duplicates:
        raise InsertError(
            "the insertion would write two chapters at " + ", ".join(duplicates)
        )

    return Plan(
        root=root,
        anchor=after,
        anchor_title=anchor_title,
        part=part,
        count=count,
        shifts=tuple(shifts),
        stubs=tuple(stubs),
    )


def render(plan: Plan, dry_run: bool) -> str:
    """The plan, as text for the terminal and the pull request."""
    lines = []
    if dry_run:
        lines.append("Dry run. No files change.")
    else:
        lines.append(
            f"Inserted {len(plan.stubs)} draft lesson(s) after {plan.anchor}."
        )
    lines.append(f"Anchor stays: {plan.anchor}")
    lines.append(f"Shift {len(plan.shifts)} id(s) by {plan.count}:")
    if plan.shifts:
        for old, new in plan.shifts:
            lines.append(f"  {old} to {new}")
    else:
        lines.append("  (none)")
    lines.append("Create draft lesson(s):")
    for stub in plan.stubs:
        lines.append(f"  {stub.slug}  draft: true  part copied from the anchor")
    lines.append(
        "Rewrite slug, exercise, prereqs, and /lessons/ links that name a shifted slug."
    )
    lines.append("Rewrite each shifted exercise meta.json id.")
    lines.append(
        "Append one renames.json entry per shifted slug, and rewrite stored "
        "values that still name one."
    )
    lines.append("Rewrite docs/CURRICULUM.md rows and add a row for each draft.")
    return "\n".join(lines)


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _stub_lesson(stub: Stub) -> str:
    summary = (
        f"A draft chapter for {stub.slug}. "
        "It holds a place in the course while the lesson is written."
    )
    if len(summary) > 140:
        summary = f"A draft chapter for {stub.slug}."
    return f"""---
title: {_yaml_quote(stub.title)}
slug: {stub.slug}
part: {_yaml_quote(stub.part)}
summary: {_yaml_quote(summary)}
minutes: 20
gpu: false
draft: true
objectives:
  - State the question this draft chapter will answer.
  - Name the function the exercise asks you to write.
  - Run the checks and read the message from the starter file.
prereqs:
  - {stub.prereq}
exercise: {stub.slug}
---

# {stub.title}

This chapter is a draft. It keeps a number in the course so the lessons after
it can move, and the work you have already finished stays attached to those
lessons.

The exercise beside this page is a placeholder. `draft_answer` takes a step
count and returns that same count. The starter raises `NotImplementedError`,
so the checks fail until you fill the function in.

## What you already know

Three facts carry over from the previous chapter:

- The previous chapter is [{stub.prereq_title}](/lessons/{stub.prereq}).
- A lesson file and its exercise directory share one slug.
- The checks call `draft_answer` and compare the return value with the number you pass in.

## A placeholder for the idea

The following figure marks this chapter as unfinished. The chapter that replaces
this draft draws the idea in its place.

```figure
<svg viewBox="0 0 560 140" role="img" aria-label="A single bar marking this chapter as a draft">
  <line x1="40" y1="110" x2="520" y2="110" class="stroke-muted"/>
  <rect x="80" y="50" width="70" height="60" class="fill-a"/>
  <text x="170" y="88">draft chapter</text>
</svg>
Figure 1. This figure holds the place of the picture the chapter will draw.
```

> [!KEY]
> A draft chapter keeps its neighbors numbered while its own text is still being written.

## Build it step by step

Fill the function in this order:

1. Open `starter.py` and find `draft_answer`. It takes one integer, `steps`, and returns that same integer.
2. Replace the `NotImplementedError` with `return steps`.
3. Run the checks. Each one calls `draft_answer` with a different number and expects that number back.

## The exercise

You write `draft_answer`. The checks call it with several step counts and expect each call to return the count it was given. The starter fails those checks until you fill the function in.
"""


def _stub_starter() -> str:
    return '''"""Placeholder for a draft chapter.

Fill in `draft_answer`. It receives a step count and returns that count.
"""


def draft_answer(steps):
    """Return `steps`, the number of steps this placeholder receives.

    Args:
        steps: an integer step count.

    Returns:
        The same integer.
    """
    raise NotImplementedError("draft_answer")


if __name__ == "__main__":
    print(draft_answer(3))
'''


def _stub_solution() -> str:
    return '''"""Worked placeholder for a draft chapter."""


def draft_answer(steps):
    """Return `steps`, the number of steps this placeholder receives.

    Args:
        steps: an integer step count.

    Returns:
        The same integer.
    """
    return steps


if __name__ == "__main__":
    print(draft_answer(3))
'''


def _stub_tests() -> str:
    checks = []
    for steps in (0, 1, 2, 3, 5, 8):
        checks.append(
            f'''
@check("draft_answer returns {steps} when it is given {steps}")
def test_steps_{steps}(sub):
    got = has(sub, "draft_answer")({steps})
    if got != {steps}:
        fail(f"got {{got}}, expected {steps}")
'''
        )
    body = "\n".join(checks)
    return f'''"""Checks for the draft exercise."""

from rlcheck import check, fail, has
{body}'''


def _stub_meta(stub: Stub) -> str:
    payload = {
        "id": stub.slug,
        "title": stub.title,
        "brief": (
            "A placeholder exercise for a draft chapter. "
            "Return the number of steps you are given."
        ),
        "needs_gpu": False,
        "timeout": 30,
        "packages": [],
        "hints": ["draft_answer returns the integer it receives."],
    }
    return json.dumps(payload, indent=2) + "\n"


def _rewrite_meta_id(path: Path, new_id: str) -> None:
    text = path.read_text(encoding="utf-8")
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InsertError(f"{path} is not valid JSON") from exc
    old = meta.get("id")
    if old == new_id:
        return
    if isinstance(old, str):
        pattern = re.compile(r'("id"\s*:\s*")' + re.escape(old) + r'(")')
        updated, found = pattern.subn(rf"\g<1>{new_id}\g<2>", text, count=1)
        if found == 1:
            path.write_text(updated, encoding="utf-8")
            return
    meta["id"] = new_id
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _rename_upward(pairs: list[tuple[Path, Path]]) -> None:
    for source, dest in sorted(pairs, key=lambda item: item[1].name, reverse=True):
        if not source.exists():
            raise InsertError(f"missing {source.name} during the shift")
        if dest.exists():
            raise InsertError(f"refusing to overwrite {dest.name}")
        source.rename(dest)


def _insert_curriculum_rows(text: str, anchor: str, rows: list[str]) -> str:
    needle = f"| `{anchor}` |"
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if needle in line:
            addition = [row if row.endswith("\n") else row + "\n" for row in rows]
            return "".join([*lines[: index + 1], *addition, *lines[index + 1 :]])
    raise InsertError(f"docs/CURRICULUM.md has no row for {anchor}")


def _curriculum_row(stub: Stub) -> str:
    delivers = "A draft chapter with a placeholder exercise."
    return f"| `{stub.slug}` | {stub.title} | {delivers} |"


def _write(plan: Plan) -> None:
    content = plan.root / "content"
    lessons = content / "lessons"
    exercises = content / "exercises"
    mapping = dict(plan.shifts)

    lesson_pairs = []
    exercise_pairs = []
    for old, new in plan.shifts:
        lesson = lessons / f"{old}.md"
        exercise = exercises / old
        if lesson.is_file():
            lesson_pairs.append((lesson, lessons / f"{new}.md"))
        if exercise.is_dir():
            exercise_pairs.append((exercise, exercises / new))
    _rename_upward(lesson_pairs)
    _rename_upward(exercise_pairs)

    for path in sorted(lessons.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        updated = replace_slugs(text, mapping)
        if updated != text:
            path.write_text(updated, encoding="utf-8")

    new_ids = {new for _, new in plan.shifts}
    for directory in sorted(exercises.iterdir()):
        if not directory.is_dir() or directory.name not in new_ids:
            continue
        meta_path = directory / "meta.json"
        if meta_path.is_file():
            _rewrite_meta_id(meta_path, directory.name)

    renames_path = content / "renames.json"
    if renames_path.is_file():
        loaded = json.loads(renames_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise InsertError("renames.json is not an object")
        existing = {str(key): value for key, value in loaded.items()}
    else:
        existing = {}
    rewritten = apply_renames(existing, mapping)
    renames_path.write_text(
        json.dumps(rewritten, indent=1) + "\n", encoding="utf-8"
    )

    curriculum_path = plan.root / "docs" / "CURRICULUM.md"
    curriculum = replace_slugs(
        curriculum_path.read_text(encoding="utf-8"), mapping
    )
    curriculum = _insert_curriculum_rows(
        curriculum, plan.anchor, [_curriculum_row(stub) for stub in plan.stubs]
    )
    curriculum_path.write_text(curriculum, encoding="utf-8")

    for stub in plan.stubs:
        (lessons / f"{stub.slug}.md").write_text(_stub_lesson(stub), encoding="utf-8")
        directory = exercises / stub.slug
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(_stub_meta(stub), encoding="utf-8")
        (directory / "starter.py").write_text(_stub_starter(), encoding="utf-8")
        (directory / "solution.py").write_text(_stub_solution(), encoding="utf-8")
        (directory / "tests.py").write_text(_stub_tests(), encoding="utf-8")


def apply(
    root: Path,
    after: str,
    slug: str,
    count: int = 1,
    dry_run: bool = False,
) -> str:
    """Insert drafts under ``root``. A dry run returns the plan and writes nothing."""
    plan = build_plan(root, after, slug, count)
    if not dry_run:
        _write(plan)
    return render(plan, dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Insert one or more draft lessons after an existing lesson and "
            "shift every later lesson so reader progress still resolves."
        )
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Slug of the lesson that stays in place. Every later lesson shifts.",
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Name of the new lesson, without its number prefix.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help=(
            "How many draft lessons to insert. Later lessons shift by this "
            "many. The default is 1."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and leave every file unchanged.",
    )
    args = parser.parse_args(argv)
    try:
        report = apply(ROOT, args.after, args.slug, args.count, args.dry_run)
    except InsertError as exc:
        print(f"insert_lesson: {exc}", file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
