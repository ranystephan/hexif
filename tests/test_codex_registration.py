from __future__ import annotations

import json
import sys
import types

import pandas as pd

from hexif.codex_registration import (
    CodexRegistrationConfig,
    CodexRegistrationPipeline,
    _build_valis_registrar,
    core_sort_key,
    discover_cells,
    lattice_basis,
    summarize_manifest,
)


def test_build_valis_registrar_uses_explicit_rigid_configuration(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    def valis(**kwargs):
        captured.update(kwargs)
        return sentinel

    fake_module = types.SimpleNamespace(registration=types.SimpleNamespace(Valis=valis))
    monkeypatch.setitem(sys.modules, "valis", fake_module)
    he_path = tmp_path / "HE.tif"
    dapi_path = tmp_path / "DAPI.tif"

    result = _build_valis_registrar(
        he_path,
        dapi_path,
        tmp_path / "registration",
        max_processed_image_dim_px=1500,
    )

    assert result is sentinel
    assert captured == {
        "src_dir": str(tmp_path),
        "dst_dir": str(tmp_path / "registration"),
        "img_list": [str(he_path), str(dapi_path)],
        "imgs_ordered": True,
        "reference_img_f": str(he_path),
        "align_to_reference": True,
        "max_processed_image_dim_px": 1500,
        "create_masks": True,
        "non_rigid_registrar_cls": None,
    }


def test_discover_cells_and_lattice_basis(tmp_path):
    bf = tmp_path / "bestFocus"
    bf.mkdir()
    for name in ["B-2.tif", "A-1.tif", "A-3.tif", "not-a-core.tif", "C-x.tif"]:
        (bf / name).touch()

    cells = discover_cells(bf)

    assert cells == [("A", 1), ("A", 3), ("B", 2)]
    assert lattice_basis(cells) == (["A", "B"], [1, 2, 3])


def test_core_sort_key():
    assert sorted(["A-10", "A-2", "B-1"], key=core_sort_key) == ["A-2", "A-10", "B-1"]


def test_cached_row_marks_high_residual_bad(tmp_path):
    out_core = tmp_path / "ccRCC_TMA1" / "A-18"
    out_core.mkdir(parents=True)
    (out_core / "meta.json").write_text(json.dumps({"mean_rigid_D": 31.8}))

    config = CodexRegistrationConfig(
        codex_root=tmp_path / "CODEX",
        out_dir=tmp_path / "pairs",
        tmas={},
        bad_align_px=30.0,
    )
    pipeline = CodexRegistrationPipeline(config)

    row = pipeline.cached_row("ccRCC_TMA1", "A-18", out_core)

    assert row["status"] == "bad_alignment"
    assert row["rigid_D_px"] == 31.8
    assert "rigid_D=31.8px" in row["error"]


def test_purge_rejected_arrays_removes_only_rejected(tmp_path):
    out_dir = tmp_path / "pairs"
    good = out_dir / "TMA" / "A-1"
    bad = out_dir / "TMA" / "A-2"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    for path in [good, bad]:
        (path / "he.npy").write_bytes(b"he")
        (path / "codex.npy").write_bytes(b"cx")

    config = CodexRegistrationConfig(codex_root=tmp_path / "CODEX", out_dir=out_dir, tmas={})
    pipeline = CodexRegistrationPipeline(config)
    manifest = pd.DataFrame(
        [
            {"tma": "TMA", "core": "A-1", "status": "cached"},
            {"tma": "TMA", "core": "A-2", "status": "bad_alignment"},
        ]
    )

    removed = pipeline.purge_rejected_arrays(manifest)

    assert removed == 2
    assert (good / "he.npy").exists()
    assert (good / "codex.npy").exists()
    assert not (bad / "he.npy").exists()
    assert not (bad / "codex.npy").exists()


def test_rebuild_manifest_repairs_downstream_cutoff_status(tmp_path):
    out_dir = tmp_path / "pairs"
    core = out_dir / "TMA" / "A-1"
    core.mkdir(parents=True)
    (core / "he.npy").write_bytes(b"he")
    (core / "codex.npy").write_bytes(b"cx")
    (core / "meta.json").write_text(json.dumps({"mean_rigid_D": 18.0}))
    pd.DataFrame(
        [
            {
                "tma": "TMA",
                "core": "A-1",
                "status": "bad_alignment",
                "rigid_D_px": 18.0,
                "error": "rigid_D=18.0px > 5.0px training cutoff",
            }
        ]
    ).to_csv(out_dir / "TMA" / "manifest.csv", index=False)

    config = CodexRegistrationConfig(
        codex_root=tmp_path / "CODEX",
        out_dir=out_dir,
        tmas={},
        bad_align_px=30.0,
    )
    pipeline = CodexRegistrationPipeline(config)

    manifest = pipeline.rebuild_manifest()

    assert manifest.loc[0, "status"] == "cached"
    assert manifest.loc[0, "rigid_D_px"] == 18.0


def test_summarize_manifest_includes_usable_residuals():
    manifest = pd.DataFrame(
        [
            {"tma": "TMA", "core": "A-1", "status": "cached", "rigid_D_px": 2.0},
            {"tma": "TMA", "core": "A-2", "status": "bad_alignment", "rigid_D_px": 200.0},
        ]
    )

    summary = summarize_manifest(manifest)

    assert "bad_alignment" in summary
    assert "usable residuals: n=1" in summary
    assert "max=2.00px" in summary
