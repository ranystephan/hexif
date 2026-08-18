"""Node-driven unit tests for ``webapp/static/workbench/state.js``.

Why a Python harness for JS code: we have no Jest / Vitest pipeline
(spec section 3.3 mandates vanilla JS, no build), and the URL ↔ State
contract is the workbench frontend's only critical invariant the
test gate can lock down without a browser. Spawning Node from pytest
keeps the test in one place; CI just needs ``node`` on PATH.

When Node is not available on the cluster we mark the test ``skip``
with the reason so the gate doesn't degrade silently.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_JS = REPO_ROOT / "webapp" / "static" / "workbench" / "state.js"


def _node() -> str | None:
    """Resolve the ``node`` binary; ``None`` when it isn't installed."""
    return shutil.which("node")


def _run_node_script(script: str) -> str:
    """Execute ``script`` via Node and return its stdout.

    Node 25 is on the cluster's module path; the test imports ``state.js``
    via dynamic ``import()`` from a tmp .mjs file so the ES-module
    syntax is honored. Any uncaught error inside the script bubbles up
    as a non-zero exit code and the test fails.
    """
    cmd = [_node() or "node", "--input-type=module", "-e", script]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"node script failed (exit {proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout.strip()


_HAVE_NODE = _node() is not None and STATE_JS.exists()


pytestmark = pytest.mark.skipif(
    not _HAVE_NODE,
    reason="node or state.js missing — skipping URL↔state contract tests",
)


# We pass STATE_JS as a file:// import so Node accepts the ES syntax
# without a package.json or build step.
def _state_js_url() -> str:
    return STATE_JS.resolve().as_uri()


def test_default_state_round_trip() -> None:
    """An empty URL round-trips to the canonical default state."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/');
    console.log(JSON.stringify({{
        core: s.core,
        mode: s.mode,
        color_by: s.color_by,
        marker: s.marker,
        phenotypes: s.phenotypes,
        selected: s.selected,
        compare: s.compare,
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["core"] is None
    assert state["mode"] == "expanded"
    assert state["color_by"] == "phenotype"
    assert state["marker"] is None
    assert state["phenotypes"] is None
    assert state["selected"] is None
    assert state["compare"] is False


def test_marker_pred_url_round_trip() -> None:
    """A URL with color_by=marker_pred preserves the marker through both directions."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const url = 'http://x/?core=A&color_by=marker_pred&marker=CD8&selected=42';
    const s = m.parseUrlToState(url);
    const back = m.stateToUrl(s);
    console.log(JSON.stringify({{
        core: s.core, color_by: s.color_by, marker: s.marker, selected: s.selected,
        round_trip: back,
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["core"] == "A"
    assert state["color_by"] == "marker_pred"
    assert state["marker"] == "CD8"
    assert state["selected"] == 42
    assert "color_by=marker_pred" in state["round_trip"]
    assert "marker=CD8" in state["round_trip"]
    assert "selected=42" in state["round_trip"]
    # Defaults must be omitted from the serialized URL.
    assert "mode=expanded" not in state["round_trip"]


def test_unknown_marker_falls_back_with_warning() -> None:
    """An unknown marker name resets to null (then re-defaults to Ki67)."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&color_by=marker_pred&marker=BOGUS');
    console.log(JSON.stringify({{ marker: s.marker, color_by: s.color_by }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    # The validator strips the unknown marker, then the cross-field
    # invariant re-fills the slot with DEFAULT_MARKER (Ki67).
    assert state["marker"] == "Ki67"
    assert state["color_by"] == "marker_pred"


def test_phenotype_csv_round_trip() -> None:
    """Phenotype CSV survives encode/decode and is preserved as a Set."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState(
        'http://x/?core=A&phenotypes=immune_cd45,t_cell_cd45_cd8,no_call'
    );
    console.log(JSON.stringify({{
        phenotypes: [...s.phenotypes].sort(),
        round_trip: m.stateToUrl(s),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["phenotypes"] == sorted(
        [
            "immune_cd45",
            "t_cell_cd45_cd8",
            "no_call",
        ]
    )
    assert "phenotypes=" in state["round_trip"]


def test_all_phenotypes_collapses_to_null() -> None:
    """If every chip is on, the phenotypes param drops out of the URL."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const all = [
        'immune_cd45','t_cell_cd45_cd8','tumor_ca9_or_panck',
        'macrophage_cd68_or_cd163','m2_like_cd68_cd163','caf_fap_or_asma',
        'proliferating_ki67','pdl1_positive','pdl1_tumor_like','no_call',
    ].join(',');
    const s = m.parseUrlToState('http://x/?core=A&phenotypes=' + all);
    const url = m.stateToUrl(s);
    console.log(JSON.stringify({{ url, hasPheno: url.includes('phenotypes=') }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["hasPheno"] is False


def test_compare_flag_round_trip() -> None:
    """The compare=1 flag survives the round trip."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&compare=1');
    const back = m.stateToUrl(s);
    console.log(JSON.stringify({{ compare: s.compare, url: back }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["compare"] is True
    assert "compare=1" in state["url"]


def test_state_equals_handles_set_phenotypes() -> None:
    """stateEquals correctly compares the phenotype Set."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const a = m.parseUrlToState('http://x/?core=A&phenotypes=immune_cd45,t_cell_cd45_cd8');
    const b = m.parseUrlToState('http://x/?core=A&phenotypes=t_cell_cd45_cd8,immune_cd45');
    const c = m.parseUrlToState('http://x/?core=A&phenotypes=immune_cd45');
    console.log(JSON.stringify({{
        ab: m.stateEquals(a, b),
        ac: m.stateEquals(a, c),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["ab"] is True  # set semantics, order-insensitive
    assert state["ac"] is False  # different sets


def test_fetch_key_invariant() -> None:
    """fetchKey is identical for two states that should share a cache slot."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const a = m.parseUrlToState('http://x/?core=A&color_by=marker_pred&marker=CD8&selected=10');
    const b = m.parseUrlToState('http://x/?core=A&color_by=marker_pred&marker=CD8&selected=999');
    const c = m.parseUrlToState('http://x/?core=A&color_by=marker_pred&marker=Ki67');
    console.log(JSON.stringify({{
        ab: m.fetchKey(a) === m.fetchKey(b),
        ac: m.fetchKey(a) === m.fetchKey(c),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["ab"] is True  # selection doesn't affect the GeoJSON fetch
    assert state["ac"] is False  # marker change does


def test_default_marker_when_url_omits_one() -> None:
    """When color_by != phenotype and URL omits marker, default to Ki67."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&color_by=marker_error');
    console.log(JSON.stringify({{
        marker: s.marker, color_by: s.color_by, defaultMarker: m.DEFAULT_MARKER,
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["marker"] == state["defaultMarker"] == "Ki67"


def test_base_split_round_trip() -> None:
    """commit 8 — base=split round-trips through the URL like 'codex' does."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&base=split');
    console.log(JSON.stringify({{
        base: s.base, url: m.stateToUrl(s),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["base"] == "split"
    assert "base=split" in state["url"]


def test_min_prob_round_trip_and_clamp() -> None:
    """commit 7 — min_prob round-trips and clamps to [0, 1]."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const a = m.parseUrlToState('http://x/?core=A&min_prob=0.75');
    const b = m.parseUrlToState('http://x/?core=A&min_prob=2.5');
    const c = m.parseUrlToState('http://x/?core=A&min_prob=-1');
    const d = m.parseUrlToState('http://x/?core=A');
    console.log(JSON.stringify({{
        a: a.min_prob, a_url: m.stateToUrl(a),
        b: b.min_prob,
        c: c.min_prob,
        d: d.min_prob, d_url: m.stateToUrl(d),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["a"] == 0.75
    assert "min_prob=0.75" in state["a_url"]
    assert state["b"] == 1.0
    assert state["c"] == 0
    assert state["d"] == 0
    assert "min_prob" not in state["d_url"]


def test_marker_preserved_in_phenotype_mode() -> None:
    """commit 3 — marker is preserved across color_by modes.

    The marker drives the CODEX base layer + side-panel marker bars even
    when the user is colouring cells by phenotype, so dropping it on a
    Focus toggle would force a repick. The URL round-trip therefore
    keeps the marker even when color_by=phenotype.
    """
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&color_by=phenotype&marker=Ki67');
    const back = m.stateToUrl(s);
    console.log(JSON.stringify({{
        marker: s.marker, color_by: s.color_by, round_trip: back,
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["color_by"] == "phenotype"
    assert state["marker"] == "Ki67"
    assert "marker=Ki67" in state["round_trip"]


# ---------------------------------------------------------------------------
# model selection: model axis
# ---------------------------------------------------------------------------


def test_model_round_trip_through_url() -> None:
    """A URL with ?model=miphei_vit preserves the model id and re-serializes it."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&model=miphei_vit');
    const back = m.stateToUrl(s);
    console.log(JSON.stringify({{ model: s.model, round_trip: back }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["model"] == "miphei_vit"
    assert "model=miphei_vit" in state["round_trip"]


def test_default_model_omitted_from_url() -> None:
    """When state.model equals the default (v1_1), it's omitted from the URL."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A');
    const back = m.stateToUrl(s);
    console.log(JSON.stringify({{ model: s.model, has_model: back.includes('model=') }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["model"] == "v1_1"
    assert state["has_model"] is False


def test_fetch_key_distinguishes_models() -> None:
    """Two states differing only by model id must have different fetchKeys."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const a = m.parseUrlToState('http://x/?core=A&model=v1');
    const b = m.parseUrlToState('http://x/?core=A&model=v1_1');
    const c = m.parseUrlToState('http://x/?core=A&model=v1');
    console.log(JSON.stringify({{
        ab_differ: m.fetchKey(a) !== m.fetchKey(b),
        ac_equal: m.fetchKey(a) === m.fetchKey(c),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["ab_differ"] is True
    assert state["ac_equal"] is True


def test_state_equals_includes_model() -> None:
    """stateEquals returns False when only model differs."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const a = m.parseUrlToState('http://x/?core=A&model=v1');
    const b = m.parseUrlToState('http://x/?core=A&model=v1_1');
    console.log(JSON.stringify({{ eq: m.stateEquals(a, b) }}));
    """
    out = _run_node_script(script)
    assert json.loads(out)["eq"] is False


# ---------------------------------------------------------------------------
# CODEX composite — base-layer URL state
# ---------------------------------------------------------------------------


def test_base_layer_round_trip() -> None:
    """`?base=codex` parses to state.base='codex' and re-serializes."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&base=codex');
    console.log(JSON.stringify({{
        base: s.base, round_trip: m.stateToUrl(s),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["base"] == "codex"
    assert "base=codex" in state["round_trip"]


def test_base_default_omitted_from_url() -> None:
    """The default `base=he` is dropped from the serialized URL."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A');
    console.log(JSON.stringify({{ base: s.base, url: m.stateToUrl(s) }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["base"] == "he"
    assert "base=" not in state["url"]


def test_base_switch_does_not_change_fetch_key() -> None:
    """fetchKey is identical for base=he and base=codex (API contract)."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const he = m.parseUrlToState('http://x/?core=A');
    const codex = m.parseUrlToState('http://x/?core=A&base=codex');
    console.log(JSON.stringify({{
        equal: m.fetchKey(he) === m.fetchKey(codex),
    }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["equal"] is True, "fetchKey changed with base — would force a wasted refetch"


def test_unknown_base_falls_back_to_he() -> None:
    """An unknown base value (typo, future extension) degrades to he."""
    script = f"""
    const m = await import('{_state_js_url()}');
    const s = m.parseUrlToState('http://x/?core=A&base=BOGUS');
    console.log(JSON.stringify({{ base: s.base }}));
    """
    out = _run_node_script(script)
    state = json.loads(out)
    assert state["base"] == "he"
