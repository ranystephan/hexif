"""Tests for help system — in-site help system.

Three layers of coverage:

1. ``webapp.help`` unit tests — :func:`list_pages` / :func:`render_page`
   against synthetic markdown in a tmp dir so the helpers can be
   exercised in isolation from the shipped help/*.md sources.
2. ``/api/help`` + ``/api/help/{page_id}`` HTTP contract tests via
   FastAPI's :class:`TestClient`, using the real shipped pages so we
   verify the canonical macro AP numbers land in the rendered body.
3. Static-asset reachability — the help panel JS is served at the
   expected static path (delegated to
   :mod:`tests.test_workbench_frontend`).

The httpx URL layer normalizes raw ``..`` segments before they reach
the route (Starlette strips them in the path resolver). The
path-traversal test therefore probes a regex-rejection form that
*does* reach the handler (``..etc_passwd``, no slash) — covering the
regex layer that's the actual line of defense. The unit-level test
exercises the literal ``../etc_passwd`` against :func:`render_page`
directly to prove the function-side guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("markdown_it")


# ---------------------------------------------------------------------------
# Stand-alone help-route client (no bundle dependency)
# ---------------------------------------------------------------------------


def _help_client():
    """Build a minimal FastAPI client mounting only the help routes.

    The shipped ``webapp.workbench_app.create_workbench_app`` needs a
    real bundle on disk; the help system is bundle-independent (markdown
    sources ship in the repo), so we mount the routes directly to keep
    the test self-contained.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.testclient import TestClient
    from webapp import help as help_mod
    from webapp.schemas import HelpPageInfo, HelpPageList

    app = FastAPI()

    @app.get("/api/help", response_model=HelpPageList)
    def help_pages() -> HelpPageList:
        return HelpPageList(
            pages=[HelpPageInfo(id=p["id"], title=p["title"]) for p in help_mod.list_pages()]
        )

    @app.get("/api/help/{page_id}")
    def help_page(page_id: str) -> HTMLResponse:
        # Mirror the production route in webapp.workbench_app: the regex
        # in render_page runs against the raw page_id (API contract), so
        # uppercase reaches the regex gate and returns 422 rather than
        # being silently lowercased to a 404.
        try:
            html = help_mod.render_page(page_id)
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_page_id", "id": page_id, "message": str(e)},
            ) from e
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": "page_not_found", "id": page_id, "message": str(e)},
            ) from e
        return HTMLResponse(content=html, media_type="text/html; charset=utf-8")

    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests for webapp.help (synthetic pages in tmp_path)
# ---------------------------------------------------------------------------


def test_markdown_it_imports() -> None:
    """The new ``markdown-it-py`` dep is reachable from the worktree."""
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    assert "<h1>" in md.render("# x")


def test_list_pages_reads_title_from_h1(tmp_path: Path) -> None:
    """``list_pages`` picks the first ``#`` heading as the tab title."""
    from webapp.help import PAGE_IDS, list_pages

    # Synthesize an overview-shaped tmp dir so we exercise the helper
    # without depending on the shipped docs evolving.
    (tmp_path / "overview.md").write_text("# Hello world\n\nbody\n")
    (tmp_path / "controls.md").write_text("# Controls!\n\n## sub\n")
    pages = list_pages(help_dir=tmp_path)
    # PAGE_IDS is the canonical ordering; the function honours it.
    ids = [p["id"] for p in pages]
    assert ids == [pid for pid in PAGE_IDS if pid in {"overview", "controls"}]
    titles = {p["id"]: p["title"] for p in pages}
    assert titles["overview"] == "Hello world"
    assert titles["controls"] == "Controls!"


def test_list_pages_falls_back_to_h2(tmp_path: Path) -> None:
    """When no ``#`` is present, the first ``##`` becomes the title."""
    from webapp.help import list_pages

    (tmp_path / "overview.md").write_text("## Only an h2\n")
    pages = list_pages(help_dir=tmp_path)
    assert pages[0]["title"] == "Only an h2"


def test_render_page_injects_heading_ids(tmp_path: Path) -> None:
    """``render_page`` slugifies every heading and emits ``id=`` attrs."""
    from webapp.help import render_page

    (tmp_path / "overview.md").write_text(
        "# What is the HEXIF Workbench?\n\n"
        "## Base layer (H&E ↔ CODEX)\n\n"
        "## Calibrated confidence strata\n"
    )
    html = render_page("overview", help_dir=tmp_path)
    assert 'id="what-is-the-hexif-workbench"' in html
    assert 'id="base-layer-he-codex"' in html
    assert 'id="calibrated-confidence-strata"' in html


def test_render_page_external_links_open_in_new_tab(tmp_path: Path) -> None:
    """``http(s)://`` links get ``target="_blank" rel="noopener…"``."""
    from webapp.help import render_page

    (tmp_path / "overview.md").write_text(
        "# Heading\n\n[lab](https://ajglab.org/) and [scroll](#section).\n"
    )
    html = render_page("overview", help_dir=tmp_path)
    assert 'href="https://ajglab.org/"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html
    # Internal anchors keep their default — no target attribute.
    assert 'href="#section"' in html
    # The internal link's <a> tag is *not* annotated with target.
    # We do a soft check via substring presence + absence by inspecting
    # the bytes between the href and the closing tag.
    after = html.split('href="#section"', 1)[1]
    assert "target=" not in after.split(">", 1)[0]


def test_render_page_rejects_path_traversal(tmp_path: Path) -> None:
    """The strict regex in :func:`render_page` refuses traversal syntax."""
    from webapp.help import render_page

    for bad in ("../etc_passwd", "..", "/etc/passwd", "a\\b", "ABC", " "):
        with pytest.raises(ValueError):
            render_page(bad, help_dir=tmp_path)


def test_render_page_escapes_raw_html_tags(tmp_path: Path) -> None:
    """Raw HTML in source is escaped, not passed through.

    Defense-in-depth: the commonmark preset defaults ``html: True``,
    which would echo a literal ``<script>`` tag through to the rendered
    body. We configure ``MarkdownIt("commonmark", {"html": False})`` so
    a future docs author who pastes raw HTML into a .md page has it
    escaped (``&lt;script&gt;``) instead of executing inside the help
    panel. The shipped pages are repo-controlled so this is hardening,
    not a live fix — but the test pins the policy.
    """
    from webapp.help import render_page

    (tmp_path / "overview.md").write_text("# Heading\n\n<script>alert('x')</script>\n\nbody\n")
    html = render_page("overview", help_dir=tmp_path)
    # The literal <script> tag must NOT appear in the output — markdown-it
    # escapes it to its entity form when html: False is set.
    assert "<script>" not in html
    assert "<script" not in html  # also catches "<script " etc.
    # The escaped form should be present so the user still sees the text.
    assert "&lt;script&gt;" in html or "&lt;script" in html


def test_render_page_missing_id_is_file_not_found(tmp_path: Path) -> None:
    """A regex-valid id with no on-disk file raises ``FileNotFoundError``."""
    from webapp.help import render_page

    # An empty tmp dir lacks every page; the call resolves the regex
    # gate, hits the is_file() check, and raises FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        render_page("overview", help_dir=tmp_path)


# ---------------------------------------------------------------------------
# /api/help endpoint contract (uses the shipped pages)
# ---------------------------------------------------------------------------


def test_api_help_lists_three_pages() -> None:
    """``GET /api/help`` returns the 3 canonical pages with non-empty titles."""
    c = _help_client()
    r = c.get("/api/help")
    assert r.status_code == 200
    pages = r.json()["pages"]
    assert len(pages) == 3
    ids = [p["id"] for p in pages]
    assert ids == ["overview", "controls", "faq"]
    for p in pages:
        assert p["title"]
        # The fallback sentinel is "Untitled: <id>" — none of the shipped
        # pages should be missing a heading.
        assert not p["title"].startswith("Untitled:")


def test_api_help_overview_renders_html() -> None:
    """``GET /api/help/overview`` returns HTML starting with a heading."""
    c = _help_client()
    r = c.get("/api/help/overview")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text.lstrip()
    # The page leads with an h1 — slugified id attached.
    assert body.startswith("<h1") or body.startswith("<h2")


def test_api_help_overview_contains_no_historical_metric() -> None:
    """Lost-run metrics must not survive in the publication help page."""
    c = _help_client()
    body = c.get("/api/help/overview").text
    assert "0.6836" not in body
    assert "not a medical device" in body


def test_api_help_unknown_page_is_404() -> None:
    """A regex-valid id with no file returns a structured 404 detail."""
    c = _help_client()
    r = c.get("/api/help/notapage")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "page_not_found"
    assert detail["id"] == "notapage"


def test_api_help_mixed_case_is_422() -> None:
    """A mixed-case id is rejected by the regex gate as 422.

    Spec §3.5 contract: ``page_id`` must match ``^[a-z0-9_-]+$``.
    Uppercase fails the regex (before any filesystem lookup), so the
    route returns 422 ``invalid_page_id`` rather than 404 — the
    behavior matches the literal regex promise in the spec.
    """
    c = _help_client()
    r = c.get("/api/help/NotAPage")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "invalid_page_id"
    assert detail["id"] == "NotAPage"


def test_api_help_path_traversal_is_422() -> None:
    """A regex-rejecting id (path-traversal-shaped) returns 422.

    The literal ``../etc_passwd`` is normalized by Starlette's URL
    resolver before reaching the route — so this test probes a form
    that *does* land in the handler with the offending string intact:
    ``..etc_passwd`` (two leading dots, no slash). The unit test
    :func:`test_render_page_rejects_path_traversal` covers the literal
    ``../etc_passwd`` against the helper directly.
    """
    c = _help_client()
    r = c.get("/api/help/..etc_passwd")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "invalid_page_id"
    assert "[a-z0-9_-]+" in detail["message"]


def test_api_help_explicit_traversal_via_unit() -> None:
    """``../etc_passwd`` against ``render_page`` raises ``ValueError``.

    Paired with :func:`test_api_help_path_traversal_is_422`, this
    covers both sides of the contract — the route maps the exception
    to 422 (verified there) and the helper actually raises the
    exception (verified here, because URL normalization makes the
    raw form unreachable through HTTP).
    """
    from webapp.help import render_page

    with pytest.raises(ValueError):
        render_page("../etc_passwd")


# ---------------------------------------------------------------------------
# Tab-strip ordering + tables
# ---------------------------------------------------------------------------


def test_overview_describes_bundle_driven_panels() -> None:
    c = _help_client()
    body = c.get("/api/help/overview").text
    assert "bundle manifest" in body
    assert "never substituted" in body


def test_controls_has_url_parameters_section() -> None:
    """The canonical URL params are documented in controls.md."""
    c = _help_client()
    body = c.get("/api/help/controls").text
    # Spot-check on the key parameter names. We use ``<code>`` because
    # the markdown source wraps them in backticks; markdown-it emits
    # ``<code>core</code>`` etc.
    for param in ("core", "model", "color_by", "marker", "base", "compare"):
        assert f"<code>{param}</code>" in body, f"missing <code>{param}</code> in controls.md"
