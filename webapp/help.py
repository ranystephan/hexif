"""Render the Workbench bundled help pages.

The Workbench ships three markdown pages — overview, controls, FAQ — that
the help panel renders in-place. This module turns the markdown sources
in ``webapp/help/*.md`` into the HTML the frontend injects.

Why a small dedicated module rather than inlining into
:mod:`webapp.workbench_app`:

* The list-pages helper introspects the on-disk markdown to read the
  page title from the first heading; it does not belong on the FastAPI
  route handler.
* The render path injects ``id=`` attributes for slug anchors and
  rewrites external links to ``target=_blank``; the policy is testable
  without spinning up a TestClient.
* Path-traversal protection is centralised here so every callsite
  (route handler, tests) gets the same guarantees.

Markdown source files are shipped with the package and are not user supplied.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown_it import MarkdownIt

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Where help pages live on disk. Resolved once at import so callers do not
# pay the path lookup on every render.
HELP_DIR: Path = Path(__file__).parent / "help"

# The three canonical pages and the order they appear in the tab strip.
# Kept module-level so tests can introspect without parsing the directory
# listing themselves (which would couple test ordering to filesystem
# enumeration).
PAGE_IDS: tuple[str, ...] = ("overview", "controls", "faq")

# Page-id regex per the API contract — bound to filename-safe chars so path
# traversal is impossible at the type system level. The route handler in
# :mod:`webapp.workbench_app` enforces the same regex for 422 vs 404
# distinction.
PAGE_ID_RE: re.Pattern[str] = re.compile(r"^[a-z0-9_-]+$")

# Default fallback title when the markdown source has no heading. Reads
# as a typo-trap rather than a silent ID echo so a docs author who forgets
# the # heading sees the issue when the page lands in the tab strip.
_DEFAULT_TITLE_PREFIX = "Untitled: "


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Slugify a heading into an anchor id.

    The scheme is:
      1. Lowercase.
      2. Drop ``&`` entirely so ``H&E`` collapses to ``HE`` (matches the
         discoverability deep-links specified in the task — e.g., the
         "Base layer (H&E ↔ CODEX)" heading must produce
         ``#base-layer-he-codex``).
      3. Replace every other non-alphanumeric run with a single ``-``.
      4. Strip leading / trailing hyphens.

    This is a deterministic, pure-Python slugifier with no plugin dependency.
    """
    s = text.lower().replace("&", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# ---------------------------------------------------------------------------
# Page metadata
# ---------------------------------------------------------------------------


def _read_title(md_path: Path) -> str:
    """Return the title to show in the tab strip.

    The convention: the first ``#`` (h1) heading wins; if none, the first
    ``##`` (h2) wins. If neither is present we fall back to a sentinel so
    docs authors notice the missing heading rather than seeing a silent
    page-id echo. The lookup reads at most the first ~200 lines to keep
    boot-time list-pages cheap even when a future help page is large.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"{_DEFAULT_TITLE_PREFIX}{md_path.stem}"
    h1: str | None = None
    h2: str | None = None
    # Iterate by line; markdown headings are always at the start of a
    # line per CommonMark.
    for raw in text.splitlines():
        line = raw.lstrip()
        if line.startswith("# ") and h1 is None:
            h1 = line[2:].strip()
            break
        if line.startswith("## ") and h2 is None:
            h2 = line[3:].strip()
    if h1:
        return h1
    if h2:
        return h2
    return f"{_DEFAULT_TITLE_PREFIX}{md_path.stem}"


def list_pages(help_dir: Path | None = None) -> list[dict[str, str]]:
    """Return the metadata for the help tab strip.

    Each page in :data:`PAGE_IDS` produces ``{"id": <id>, "title":
    <heading>}``. The title is read straight from the markdown source so
    docs edits propagate without code changes.

    Why an explicit page-id list rather than ``Path.glob("*.md")``:
    glob-order is filesystem-dependent (case-sensitive vs not, inode
    insertion order) and we want the tab strip to be reproducible.
    """
    base = help_dir if help_dir is not None else HELP_DIR
    out: list[dict[str, str]] = []
    for page_id in PAGE_IDS:
        md_path = base / f"{page_id}.md"
        if not md_path.is_file():
            # Skip absent pages rather than erroring — a partial install
            # (e.g., one page deleted during dev) should still render the
            # remaining tabs. Tests cover the full-list path explicitly.
            continue
        out.append({"id": page_id, "title": _read_title(md_path)})
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _new_renderer() -> MarkdownIt:
    """Build a MarkdownIt instance configured for the help pages.

    Why per-call rather than module-level: MarkdownIt's renderer carries
    mutable rule state and we install custom heading + link renderers
    below. Per-call construction keeps the patches isolated.

    The base preset is ``commonmark`` plus an explicit ``table`` enable —
    every help page uses CommonMark headings / lists / fenced blocks
    plus the controls.md / overview.md tables. We deliberately do *not*
    enable ``linkify`` (which would auto-detect bare URLs) because that
    rule pulls in the optional ``linkify-it-py`` dep; the help pages
    write explicit ``[text](url)`` links so the rule is unnecessary.

    ``html: False`` is set as defense-in-depth: the commonmark preset
    defaults to ``html: True``, which would pass raw HTML through
    untouched. The help pages are repo-controlled so there is no live
    XSS surface today, but explicitly disabling raw-HTML hardens us
    against a future docs author pasting a literal ``<script>`` tag
    into a .md page and having it execute inside the panel.
    """
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    _install_heading_anchors(md)
    _install_link_target_blank(md)
    return md


def _install_heading_anchors(md: MarkdownIt) -> None:
    """Inject ``id="<slug>"`` attributes on every heading open token.

    markdown-it-py does not slugify headings by default; the mdit-py-
    plugins ``anchors`` plugin would add a dep just to wrap this two-line
    patch. We do it inline by walking the heading_open token and reading
    the inline content of the immediately following inline token (per the
    standard markdown-it token layout: heading_open, inline, heading_close).
    """
    default_render = md.renderer.rules.get("heading_open")

    def render_heading_open(tokens, idx, options, env):  # type: ignore[no-untyped-def]
        # The inline token following heading_open carries the text content;
        # it always exists in well-formed CommonMark input.
        inline = tokens[idx + 1] if idx + 1 < len(tokens) else None
        text = inline.content if inline is not None else ""
        slug = _slugify(text)
        if slug:
            tokens[idx].attrSet("id", slug)
        if default_render is not None:
            return default_render(tokens, idx, options, env)
        return md.renderer.renderToken(tokens, idx, options, env)

    md.renderer.rules["heading_open"] = render_heading_open


def _install_link_target_blank(md: MarkdownIt) -> None:
    """Open ``http(s)://`` and ``mailto:`` links in a new tab.

    Internal anchors (``href="#..."``) keep their default behavior so
    they scroll within the help panel. This rule mutates only the token's
    attributes — the renderer's own escape / encode pipeline still runs.
    """
    default_render = md.renderer.rules.get("link_open")

    def render_link_open(tokens, idx, options, env):  # type: ignore[no-untyped-def]
        href = tokens[idx].attrGet("href") or ""
        if href.startswith(("http://", "https://", "mailto:")):
            tokens[idx].attrSet("target", "_blank")
            tokens[idx].attrSet("rel", "noopener noreferrer")
        if default_render is not None:
            return default_render(tokens, idx, options, env)
        return md.renderer.renderToken(tokens, idx, options, env)

    md.renderer.rules["link_open"] = render_link_open


def render_page(page_id: str, help_dir: Path | None = None) -> str:
    """Render ``webapp/help/<page_id>.md`` to an HTML body.

    Returns the body HTML (no ``<html>`` / ``<head>`` wrapper) so the
    frontend can inject it into a scrollable container. The route
    handler is responsible for the ``Content-Type`` header.

    :raises ValueError: ``page_id`` does not match :data:`PAGE_ID_RE`.
        Path-traversal attempts (``../etc_passwd``, absolute paths) are
        rejected here before any filesystem access.
    :raises FileNotFoundError: the regex-validated id has no on-disk
        markdown file. Distinct from the value error so the route can
        map them to 422 vs 404 respectively.
    """
    if not PAGE_ID_RE.fullmatch(page_id):
        raise ValueError(f"invalid page_id {page_id!r}; must match {PAGE_ID_RE.pattern}")
    base = help_dir if help_dir is not None else HELP_DIR
    md_path = base / f"{page_id}.md"
    # Belt-and-braces — even with the regex check, an explicit is_file()
    # surfaces a missing-page case as a 404-mapped FileNotFoundError
    # rather than a confusing IOError from open().
    if not md_path.is_file():
        raise FileNotFoundError(f"unknown help page: {page_id!r}")
    text = md_path.read_text(encoding="utf-8")
    return _new_renderer().render(text)
