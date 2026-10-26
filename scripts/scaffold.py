"""Extract every source file from DESIGN.md into the working tree.

    python scripts/scaffold.py            # write files
    python scripts/scaffold.py --check    # report differences, write nothing

DESIGN.md is the single source of truth: each file lives under a heading naming
its path, followed by a fenced block. This reads exactly that and nothing else,
so the document and the tree cannot drift.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "DESIGN.md"

# "## 12. `app/models.py`"  /  "### `tests/conftest.py` — the important one"
HEADING = re.compile(
    r"^#{2,4}\s+(?:\d+\.\s*)?`([^`]+)`(?:\s+[-—].*)?\s*$", re.M)

# Only these become files. Everything else in the doc is prose or illustration.
SUFFIXES = {".py", ".toml", ".sql", ".yml", ".yaml", ".json", ".js", ".sh",
            ".md", ".example", ".gitignore"}
NAMED = {"Dockerfile", ".gitignore", ".env.example"}

# Some sections show a usage example before the code. Matching the fence
# language to the file type takes the RIGHT block regardless of order -- this
# script's own section is the case that found the bug.
LANGS = {
    ".py": {"python", "py"}, ".toml": {"toml"}, ".sql": {"sql"},
    ".yml": {"yaml", "yml"}, ".yaml": {"yaml", "yml"}, ".json": {"json"},
    ".js": {"javascript", "js"}, ".sh": {"bash", "sh"},
    ".md": {"markdown", "md"}, ".example": {"bash", "sh"},
}
NAMED_LANGS = {"Dockerfile": {"dockerfile"}, ".gitignore": {"gitignore"},
               ".env.example": {"bash", "sh"}}

# Written after extraction; they have no content worth putting in the document.
EMPTY_FILES = [
    "app/__init__.py",
    "app/providers/__init__.py",
    "app/cache/__init__.py",
    "bench/results/.gitkeep",
]


def is_file_heading(name: str) -> bool:
    return name in NAMED or Path(name).suffix in SUFFIXES


def extract(text: str) -> list[tuple[str, str]]:
    """Return [(path, body)] for the best fenced block under each file heading.

    "Best" = the first block whose fence language matches the file type, falling
    back to the first block if none is tagged. A section that shows a `bash`
    usage example before its code would otherwise write the example into the
    source file.
    """
    lines = text.splitlines()
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        m = HEADING.match(lines[i])
        if not m or not is_file_heading(m.group(1)):
            i += 1
            continue
        path = m.group(1)

        want = NAMED_LANGS.get(path) or LANGS.get(Path(path).suffix) or set()
        blocks: list[tuple[str, str]] = []          # (language, body)
        j = i + 1
        while j < len(lines):
            if HEADING.match(lines[j]):             # next file section
                break
            if not lines[j].startswith("```"):
                j += 1
                continue
            # README.md is fenced with four backticks because it contains three.
            fence = "````" if lines[j].startswith("````") else "```"
            lang = lines[j][len(fence):].strip().lower()
            k = j + 1
            body: list[str] = []
            while k < len(lines) and not lines[k].startswith(fence):
                body.append(lines[k])
                k += 1
            blocks.append((lang, chr(10).join(body).rstrip(chr(10)) + chr(10)))
            j = k + 1
            if lang in want:                        # right one, stop looking
                break

        if blocks:
            out.append((path, next((b for lang, b in blocks if lang in want),
                                   blocks[0][1])))
        i = j
    return out


def verify(path: str, body: str) -> str | None:
    """Cheap syntax check so a bad paste fails here rather than at runtime."""
    try:
        if path.endswith(".py"):
            ast.parse(body, filename=path)
        elif path.endswith(".json"):
            json.loads(body)
    except (SyntaxError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report differences without writing")
    args = ap.parse_args()

    if not DOC.exists():
        print(f"cannot find {DOC}", file=sys.stderr)
        return 1

    files = extract(DOC.read_text(encoding="utf-8"))
    if not files:
        print("no file blocks found -- has the heading format changed?", file=sys.stderr)
        return 1

    written = changed = same = 0
    problems: list[str] = []

    for path, body in files:
        err = verify(path, body)
        if err:
            problems.append(f"{path}: {err}")
            continue

        dest = ROOT / path
        existing = dest.read_text(encoding="utf-8") if dest.exists() else None

        if existing == body:
            same += 1
            continue
        if args.check:
            print(f"  DIFFERS  {path}" if existing is not None else f"  NEW      {path}")
            changed += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
        print(f"  {'updated' if existing is not None else 'wrote  '}  {path}")
        written += 1

    if not args.check:
        for rel in EMPTY_FILES:
            p = ROOT / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.touch()
                print(f"  wrote    {rel}")

    print(f"\n{len(files)} file blocks | {written} written | {same} unchanged"
          + (f" | {changed} would change" if args.check else ""))

    if problems:
        print("\nPROBLEMS -- these were NOT written:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    if not args.check:
        print("\nnext:  cp .env.example .env && docker compose up --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
