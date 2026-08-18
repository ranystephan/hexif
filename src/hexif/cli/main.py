"""Command-line interface for real-data preparation and the Workbench."""

from __future__ import annotations

import argparse
import sys

from hexif import __version__


def _cmd_build_cell_tables(args: argparse.Namespace) -> int:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from hexif.pipeline.bundle import build_bundle

    try:
        out = build_bundle(
            args.output_dir,
            pairs_dir=args.pairs_dir,
            cell_predictions=args.cell_predictions,
            consensus=args.consensus,
            manifest=args.manifest,
            thresholds_json=args.thresholds_json,
            model_metadata=args.model_metadata,
            model_ids=(args.model_id,),
            primary_model=args.model_id,
            split=args.split,
            max_cores=args.max_cores,
            render_thumbs=not args.no_thumbs,
            skip_codex=args.skip_codex,
            build_polygons=args.polygons,
            masks_dir=args.masks_dir,
            build_codex_composite=args.codex_composite,
            codex_scaler=args.codex_scaler,
        )
    except FileNotFoundError as e:
        # Reachable when --pairs-dir / --cell-predictions / --manifest /
        # --masks-dir point at a path that doesn't exist. Argparse's exit
        # convention is exit 2, but for runtime input errors a clean
        # stderr line + exit 1 is the matching shape for our CLI.
        print(f"hexif: {e}", file=sys.stderr)
        return 1
    print(f"bundle: {out}")
    return 0


def _cmd_rebuild_cell_tables(args: argparse.Namespace) -> int:
    """Extend an existing bundle's cells.parquet with new model columns.

    Each ``--add-models`` argument is a ``<model_id>=<csv_path>`` pair.
    The same CSV may be reused for several model ids; the parser pulls
    the matching ``ch??_pred_<model_id>`` columns out of the CSV for
    each merge. The manifest's ``models`` block is rewritten to
    include every model id present after the merge.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from hexif.pipeline.bundle import parse_model_csv_pair, rebuild_cell_tables

    bundle_arg = args.bundle
    if ".." in Path(bundle_arg).parts:
        print(
            f"hexif: --bundle must not contain '..' traversal: {bundle_arg}",
            file=sys.stderr,
        )
        return 1

    pairs: list[tuple[str, Path]] = []
    for spec in args.add_models or []:
        try:
            pairs.append(parse_model_csv_pair(spec))
        except (ValueError, FileNotFoundError) as e:
            print(f"hexif: {e}", file=sys.stderr)
            return 2

    try:
        out = rebuild_cell_tables(
            args.bundle,
            pairs,
            model_metadata=args.model_metadata,
            default_model_id=args.default_model_id,
        )
    except FileNotFoundError as e:
        print(f"hexif: {e}", file=sys.stderr)
        return 1
    print(f"cells.parquet: {out}")
    return 0


def _cmd_rebuild_codex_composite(args: argparse.Namespace) -> int:
    """Render `codex_composite.png` for every core in an existing bundle."""
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from hexif.pipeline.bundle import rebuild_codex_composite

    bundle_arg = args.bundle
    if ".." in Path(bundle_arg).parts:
        print(
            f"hexif: --bundle must not contain '..' traversal: {bundle_arg}",
            file=sys.stderr,
        )
        return 1

    try:
        composite = rebuild_codex_composite(
            args.bundle,
            pairs_dir=args.pairs_dir,
            codex_scaler=args.codex_scaler,
        )
    except FileNotFoundError as e:
        print(f"hexif: {e}", file=sys.stderr)
        return 1
    print(f"composite_percentiles: {len(composite)} cores")
    return 0


def _cmd_rebuild_codex_marker_thumbs(args: argparse.Namespace) -> int:
    """Render per-marker CODEX truth PNGs (`ch{NN}_codex.png`) for every core."""
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from hexif.pipeline.bundle import rebuild_codex_marker_thumbs

    bundle_arg = args.bundle
    if ".." in Path(bundle_arg).parts:
        print(
            f"hexif: --bundle must not contain '..' traversal: {bundle_arg}",
            file=sys.stderr,
        )
        return 1

    try:
        rendered = rebuild_codex_marker_thumbs(
            args.bundle,
            pairs_dir=args.pairs_dir,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as e:
        print(f"hexif: {e}", file=sys.stderr)
        return 1
    total = sum(len(v) for v in rendered.values())
    print(f"per-marker codex thumbs: {total} PNGs across {len(rendered)} cores")
    return 0


def _cmd_rebuild_polygons(args: argparse.Namespace) -> int:
    """Add ``cell_polygons.parquet`` to an existing bundle in place."""
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from hexif.pipeline.bundle import rebuild_polygons

    # Refuse traversal in --bundle (low concern locally, cheap polish).
    # We resolve the path and compare to its resolved form; argparse already
    # accepts the literal so we must do this manually here.
    bundle_arg = args.bundle
    if ".." in Path(bundle_arg).parts:
        print(f"hexif: --bundle must not contain '..' traversal: {bundle_arg}", file=sys.stderr)
        return 1

    try:
        out = rebuild_polygons(args.bundle, masks_dir=args.masks_dir)
    except FileNotFoundError as e:
        # Hit when --bundle points at a missing directory, the bundle is
        # missing cells.parquet, or the masks_dir resolved from the manifest
        # has been deleted. All three are user-actionable, not bugs.
        print(f"hexif: {e}", file=sys.stderr)
        return 1
    print(f"cell_polygons: {out}")
    return 0


def _cmd_spatial_summary(args: argparse.Namespace) -> int:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from hexif.spatial import build_spatial_summary

    edges, summary = build_spatial_summary(
        args.bundle,
        method=args.method,
        k=args.k,
        radius_px=args.radius_px,
        write_edges=args.write_edges,
        model_id=args.model_id,
    )
    print(f"spatial_summary: {summary}")
    if edges:
        print(f"spatial_edges:   {edges}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from pathlib import Path

    from hexif.report.pdf import render_core_report

    if args.all:
        import pandas as pd

        cores = pd.read_parquet(Path(args.bundle) / "core_summary.parquet")
        basenames = cores["basename"].tolist()
    else:
        basenames = args.basenames
    if not basenames:
        print(
            "hexif report: pass --all or one or more positional basenames",
            file=__import__("sys").stderr,
        )
        return 2
    Path(args.bundle) / "reports"
    for bn in basenames:
        try:
            p = render_core_report(
                args.bundle, bn, model_id=args.model_id, include_marker_maps=not args.no_maps
            )
            print(p)
        except Exception as e:
            print(f"FAIL {bn}: {e}", file=__import__("sys").stderr)
    return 0


def _cmd_serve_workbench(args: argparse.Namespace) -> int:
    """Launch the new Workbench FastAPI app reading a parquet bundle."""
    try:
        from webapp.cell_service import CellPhenotypeService
        from webapp.workbench_app import create_workbench_app
    except ImportError as e:
        print(
            f"hexif serve-workbench requires the [web] extra: pip install 'hexif[web]'\n  ({e})",
            file=sys.stderr,
        )
        return 2
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    service = CellPhenotypeService(args.bundle, model_id=args.model_id)
    app = create_workbench_app(service)
    import uvicorn

    print(f"\n  HEXIF Workbench at http://{args.host}:{args.port}/\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_audit_labels(args: argparse.Namespace) -> int:
    """Run the label audit label audit over a bundle's ``cells.parquet``.

    Writes ``cohort_audit.csv``, ``cohort_audit.md``, and
    ``cohort_audit.json`` under ``--out``. See
    ``docs/reproducibility.md`` for the algorithm and output
    schema.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("hexif.audit_labels")

    bundle = Path(args.bundle)
    thresholds = Path(args.thresholds)
    out_dir = Path(args.out)

    # 1. Validate input paths early — argparse already covered required-ness.
    if not bundle.exists():
        print(f"hexif audit-labels: --bundle does not exist: {bundle}", file=sys.stderr)
        return 1
    cells_path = bundle / "cells.parquet"
    if not cells_path.exists():
        print(
            f"hexif audit-labels: bundle is missing cells.parquet: {cells_path}",
            file=sys.stderr,
        )
        return 1
    if not thresholds.exists():
        print(
            f"hexif audit-labels: --thresholds does not exist: {thresholds}",
            file=sys.stderr,
        )
        return 1

    # 2. Make the output directory and prove it's writable before the long run.
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".hexif_audit_labels_write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        print(
            f"hexif audit-labels: --out is not writable ({out_dir}): {e}",
            file=sys.stderr,
        )
        return 1

    try:
        import pandas as pd

        from hexif.audit import (
            audit_cohort,
            load_cohort_thresholds,
            write_cohort_audit_csv,
            write_cohort_audit_json,
            write_cohort_audit_summary,
        )

        # 3. Load the cell table. fastparquet matches the rest of the
        #    bundle pipeline.
        cells = pd.read_parquet(cells_path, engine="fastparquet")

        # 4. Load the cohort thresholds JSON into the dict shape audit_cohort
        #    needs. We still pass the original path through for provenance.
        cohort_thresholds = load_cohort_thresholds(thresholds)

        # 5. Optional core filter.
        if args.cores:
            requested = [c.strip() for c in args.cores.split(",") if c.strip()]
            present = set(cells["basename"].unique())
            missing = [c for c in requested if c not in present]
            if missing:
                print(
                    "hexif audit-labels: --cores requested basenames not in bundle: "
                    + ", ".join(missing),
                    file=sys.stderr,
                )
                return 1
            cells = cells[cells["basename"].isin(requested)].reset_index(drop=True)

        # 6. Run the audit. Pass provenance paths through so the result
        #    records exactly what was audited.
        result = audit_cohort(
            cells,
            cohort_thresholds=cohort_thresholds,
            thresholds_path=thresholds,
            bundle_path=bundle,
            zscore_tau=args.zscore_tau,
            leiden_resolution=args.leiden_resolution,
            leiden_n_neighbors=args.leiden_n_neighbors,
            leiden_pca_n_comps=args.leiden_pca_n_comps,
            scale_max_value=args.scale_max_value,
            random_state=args.seed,
            show_progress=not args.no_progress,
        )

        # 6. Serialize all three outputs.
        csv_path = write_cohort_audit_csv(result, out_dir / "cohort_audit.csv")
        md_path = write_cohort_audit_summary(
            result,
            out_dir / "cohort_audit.md",
            top_n_outliers=args.top_n_outliers,
        )
        json_path = write_cohort_audit_json(result, out_dir / "cohort_audit.json")
    except SystemExit:
        raise
    except Exception:
        logger.exception("hexif audit-labels failed")
        return 1

    # 7. Summary lines to stderr — stdout stays clean for redirection.
    n_rows = sum(len(c.per_marker) for c in result.cores)
    print(
        f"[hexif audit-labels] wrote {n_rows} rows to {csv_path}",
        file=sys.stderr,
    )
    print(f"[hexif audit-labels]   summary: {md_path}", file=sys.stderr)
    print(f"[hexif audit-labels]   provenance: {json_path}", file=sys.stderr)

    n_suspicious = sum(
        1 for core in result.cores for m in core.per_marker.values() if m.suspicious_zero_consensus
    )
    # Find the marker with the highest mean disagreement across cores.
    marker_mean_disagreement: dict[str, float] = {}
    for marker_name, _ch in result.markers:
        gaps = [
            core.per_marker[marker_name].disagreement_score
            for core in result.cores
            if marker_name in core.per_marker
        ]
        if gaps:
            marker_mean_disagreement[marker_name] = sum(gaps) / len(gaps)
    if marker_mean_disagreement:
        top_marker, top_val = max(marker_mean_disagreement.items(), key=lambda kv: kv[1])
        top_str = f"{top_marker} (mean disagreement {top_val:.3f})"
    else:
        top_str = "n/a"
    print(
        f"[hexif audit-labels]   suspicious_zero_consensus rows: {n_suspicious}; "
        f"top marker by mean disagreement: {top_str}",
        file=sys.stderr,
    )
    return 0


def _cmd_build_spacec_labels(args: argparse.Namespace) -> int:
    """Rebuild training-set positivity labels via SPACEc-style cluster-then-label.

    Reads a cell table, runs Leiden clustering per core in 12-marker
    space, derives per-cluster positivity, and appends the new
    ``chXX_pos_spacec`` / ``chXX_zscore_spacec`` / ``cluster_id_spacec`` /
    ``phenotype_<name>_call_spacec`` columns. Optionally re-runs the
    label audit audit on the rebuilt labels to check the spec's acceptance
    gate (per-marker mean disagreement < 0.20).

    See ``docs/reproducibility.md`` for the algorithm.
    """
    import logging
    import os
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("hexif.build_spacec_labels")

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1. Validate input/output paths early.
    if not input_path.exists():
        print(
            f"hexif build-spacec-labels: --input does not exist: {input_path}",
            file=sys.stderr,
        )
        return 1
    suffix = input_path.suffix.lower()
    if suffix not in {".parquet", ".csv"}:
        print(
            f"hexif build-spacec-labels: --input must be .parquet or .csv (got {suffix!r})",
            file=sys.stderr,
        )
        return 1
    if output_path.suffix.lower() != suffix:
        print(
            f"hexif build-spacec-labels: --output suffix {output_path.suffix!r} "
            f"must match --input suffix {suffix!r} (no format conversion)",
            file=sys.stderr,
        )
        return 1

    output_parent = output_path.parent
    try:
        output_parent.mkdir(parents=True, exist_ok=True)
        probe = output_parent / ".hexif_build_spacec_labels_write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        print(
            f"hexif build-spacec-labels: --output parent is not writable ({output_parent}): {e}",
            file=sys.stderr,
        )
        return 1

    if args.reaudit and (not args.reaudit_thresholds or args.max_mean_disagreement is None):
        print(
            "hexif build-spacec-labels: --reaudit requires "
            "--reaudit-thresholds and --max-mean-disagreement.",
            file=sys.stderr,
        )
        return 1
    reaudit_thresholds_path: Path | None = None
    if args.reaudit_thresholds:
        reaudit_thresholds_path = Path(args.reaudit_thresholds)
        if not reaudit_thresholds_path.exists():
            print(
                f"hexif build-spacec-labels: --reaudit-thresholds does not exist: "
                f"{reaudit_thresholds_path}",
                file=sys.stderr,
            )
            return 1

    try:
        import pandas as pd

        from hexif.pipeline.spacec_gating import (
            MARKERS_DEFAULT,
            build_spacec_labels,
            emit_new_columns,
            write_spacec_provenance,
        )
        from hexif.pipeline.spacec_gating.types import col_pos_spacec

        # 3. Read the cell table — detect format from suffix.
        if suffix == ".parquet":
            try:
                cells = pd.read_parquet(input_path, engine="fastparquet")
            except (ImportError, ValueError):
                cells = pd.read_parquet(input_path)
        else:
            cells = pd.read_csv(input_path)

        if "basename" not in cells.columns:
            print(
                f"hexif build-spacec-labels: --input is missing the 'basename' column "
                f"({input_path})",
                file=sys.stderr,
            )
            return 1

        # 4. Optional core filter.
        if args.cores:
            requested = [c.strip() for c in args.cores.split(",") if c.strip()]
            present = set(cells["basename"].astype(str).unique())
            missing = [c for c in requested if c not in present]
            if missing:
                print(
                    "hexif build-spacec-labels: --cores requested basenames not in "
                    "--input: " + ", ".join(missing),
                    file=sys.stderr,
                )
                return 1
            cells = cells[cells["basename"].astype(str).isin(requested)].reset_index(drop=True)

        n_cells_input = len(cells)
        n_cores_input = int(cells["basename"].nunique()) if n_cells_input else 0

        # 5. Run the rebuild.
        result = build_spacec_labels(
            cells,
            markers=MARKERS_DEFAULT,
            zscore_tau=args.zscore_tau,
            leiden_resolution=args.leiden_resolution,
            leiden_n_neighbors=args.leiden_n_neighbors,
            leiden_pca_n_comps=args.leiden_pca_n_comps,
            scale_max_value=args.scale_max_value,
            random_state=args.seed,
            show_progress=not args.no_progress,
        )

        # 6. Augment the DataFrame.
        augmented = emit_new_columns(result, cells)

        # 7. Atomic write of the augmented table, preserving format.
        tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
        # Clean any leftover .tmp from a prior crashed run.
        if tmp_output.exists():
            tmp_output.unlink()
        if suffix == ".parquet":
            try:
                augmented.to_parquet(tmp_output, engine="fastparquet", index=False)
            except (ImportError, ValueError):
                augmented.to_parquet(tmp_output, index=False)
        else:
            augmented.to_csv(tmp_output, index=False)
        os.replace(tmp_output, output_path)

        # 8. Optional provenance JSON next to the output.
        provenance_path: Path | None = None
        if args.write_provenance:
            provenance_path = write_spacec_provenance(
                result, output_parent / "spacec_labels_provenance.json"
            )

        # 9. Optional --reaudit on the new labels.
        reaudit_dir: Path | None = None
        reaudit_disagreements: dict[str, float] = {}
        if args.reaudit:
            from hexif.audit import (
                audit_cohort,
                load_cohort_thresholds,
                write_cohort_audit_csv,
                write_cohort_audit_json,
                write_cohort_audit_summary,
            )

            assert reaudit_thresholds_path is not None  # narrowed above

            cohort_thresholds = load_cohort_thresholds(reaudit_thresholds_path)

            # Build a shallow copy of the augmented cells with each
            # `chXX_pos_consensus` column replaced by its `chXX_pos_spacec`
            # counterpart. Column views (no array copy) so this stays
            # cheap for the 123k × 280-col bundle frame. The audit reads
            # chXX_pos_consensus to populate n_pos_cohort_consensus; with
            # this swap that field reflects the rebuilt labels.
            reaudit_cells = augmented.copy(deep=False)
            for _name, channel in MARKERS_DEFAULT:
                cons_col = f"ch{channel:02d}_pos_consensus"
                spacec_col = col_pos_spacec(channel)
                if spacec_col not in reaudit_cells.columns:
                    raise RuntimeError(
                        f"build-spacec-labels: --reaudit expected column {spacec_col} "
                        "to be present after emit_new_columns — internal contract bug"
                    )
                reaudit_cells[cons_col] = reaudit_cells[spacec_col].to_numpy()

            reaudit_result = audit_cohort(
                reaudit_cells,
                cohort_thresholds=cohort_thresholds,
                thresholds_path=reaudit_thresholds_path,
                bundle_path=output_path,
                zscore_tau=args.zscore_tau,
                leiden_resolution=args.leiden_resolution,
                leiden_n_neighbors=args.leiden_n_neighbors,
                leiden_pca_n_comps=args.leiden_pca_n_comps,
                scale_max_value=args.scale_max_value,
                random_state=args.seed,
                show_progress=not args.no_progress,
            )

            reaudit_dir = output_parent / "cohort_audit_on_spacec_labels"
            reaudit_dir.mkdir(parents=True, exist_ok=True)
            write_cohort_audit_csv(reaudit_result, reaudit_dir / "cohort_audit.csv")
            write_cohort_audit_summary(reaudit_result, reaudit_dir / "cohort_audit.md")
            write_cohort_audit_json(reaudit_result, reaudit_dir / "cohort_audit.json")

            # Compute per-marker mean disagreement on the rebuilt labels.
            for marker_name, _ch in reaudit_result.markers:
                gaps = [
                    core.per_marker[marker_name].disagreement_score
                    for core in reaudit_result.cores
                    if marker_name in core.per_marker
                ]
                reaudit_disagreements[marker_name] = sum(gaps) / len(gaps) if gaps else float("inf")

    except SystemExit:
        raise
    except Exception:
        logger.exception("hexif build-spacec-labels failed")
        return 1

    # 10. Stderr-only summary.
    print(
        f"[hexif build-spacec-labels] processed {result.n_cores} cores "
        f"({result.n_cells_total:,} cells) — input had {n_cores_input} cores / "
        f"{n_cells_input:,} cells",
        file=sys.stderr,
    )
    print(
        f"[hexif build-spacec-labels]   wrote {output_path} "
        f"({len(augmented):,} rows × {len(augmented.columns)} cols)",
        file=sys.stderr,
    )
    if provenance_path is not None:
        print(
            f"[hexif build-spacec-labels]   provenance: {provenance_path}",
            file=sys.stderr,
        )

    if args.reaudit:
        assert reaudit_dir is not None  # set in the --reaudit branch above
        print(
            f"[hexif build-spacec-labels]   re-audit artifacts: {reaudit_dir}",
            file=sys.stderr,
        )
        gate = float(args.max_mean_disagreement)
        offending = {name: val for name, val in reaudit_disagreements.items() if val > gate}
        if offending:
            offenders_str = ", ".join(
                f"{name}={val:.3f}" for name, val in sorted(offending.items())
            )
            print(
                f"[hexif build-spacec-labels]   GATE FAIL: {len(offending)}/"
                f"{len(reaudit_disagreements)} markers exceed mean disagreement "
                f"{gate:.2f}: {offenders_str}",
                file=sys.stderr,
            )
            return 2
        max_marker, max_val = max(reaudit_disagreements.items(), key=lambda kv: kv[1])
        print(
            f"[hexif build-spacec-labels]   GATE PASS: all "
            f"{len(reaudit_disagreements)} markers have mean disagreement < "
            f"{gate:.2f} (worst: {max_marker}={max_val:.3f}).",
            file=sys.stderr,
        )
    return 0


def _cmd_codex_register(args: argparse.Namespace) -> int:
    from pathlib import Path

    from hexif.codex_registration import (
        CodexRegistrationConfig,
        CodexRegistrationPipeline,
        load_tma_specs,
        summarize_manifest,
    )

    config = CodexRegistrationConfig(
        codex_root=Path(args.codex_root),
        out_dir=Path(args.out_dir),
        tmas=load_tma_specs(args.tma_map),
        bad_align_px=args.bad_align_px,
        empty_he_sat=args.empty_he_sat,
        empty_dapi_mean=args.empty_dapi_mean,
    )
    pipeline = CodexRegistrationPipeline(config)

    if args.rebuild_manifest:
        manifest = pipeline.rebuild_manifest()
    elif args.all:
        manifest = pipeline.process_all(force=args.force, retry_bad=args.retry_bad)
    else:
        if not args.tma:
            raise SystemExit("codex-register requires --all, --tma, or --rebuild-manifest")
        manifest = pipeline.process_tma(args.tma, force=args.force, retry_bad=args.retry_bad)

    if args.purge_rejected_arrays:
        removed = pipeline.purge_rejected_arrays(manifest)
        print(f"purged {removed} stale arrays for rejected cores")

    print(summarize_manifest(manifest))
    print(f"output dir: {pipeline.config.out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hexif", description="HEXIF command-line interface.")
    p.add_argument("--version", action="version", version=f"hexif {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # Workbench bundle commands.
    s = sub.add_parser(
        "build-cell-tables", help="Build a Workbench parquet bundle from existing eval outputs"
    )
    s.add_argument("--output-dir", dest="output_dir", required=True)
    s.add_argument("--pairs-dir", dest="pairs_dir", required=True)
    s.add_argument(
        "--cell-predictions",
        dest="cell_predictions",
        required=True,
        help="Cell-level prediction CSV from the validated run",
    )
    s.add_argument("--consensus", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument(
        "--thresholds-json",
        required=True,
        help="Validated schema-version-1 operating-threshold artifact",
    )
    s.add_argument(
        "--model-metadata",
        required=True,
        help="Schema-version-1 JSON with model identity and run provenance",
    )
    s.add_argument("--model-id", required=True, help="Model id present in the predictions CSV")
    s.add_argument("--split", default="val")
    s.add_argument("--max-cores", dest="max_cores", type=int, default=None)
    s.add_argument("--no-thumbs", action="store_true")
    s.add_argument(
        "--skip-codex",
        action="store_true",
        help="Don't render CODEX-truth thumbs (faster, smaller bundle)",
    )
    s.add_argument(
        "--polygons",
        dest="polygons",
        action="store_true",
        default=True,
        help="Emit cell_polygons.parquet from masks_v1/ (default; required by the workbench overlay)",
    )
    s.add_argument(
        "--no-polygons",
        dest="polygons",
        action="store_false",
        help="Skip polygon extraction (faster build; the workbench will degrade to centroid dots)",
    )
    s.add_argument(
        "--masks-dir",
        dest="masks_dir",
        required=True,
        help="Directory containing <basename>_nuclei.npy and _expanded8.npy",
    )
    # CODEX composite — CODEX composite (DAPI/panCK/CD45) PNGs +
    # per-core percentile cache.
    s.add_argument(
        "--codex-composite",
        dest="codex_composite",
        action="store_true",
        default=True,
        help="Render the CODEX composite (DAPI+CD45+panCK) PNG per core (default)",
    )
    s.add_argument(
        "--no-codex-composite",
        dest="codex_composite",
        action="store_false",
        help="Skip CODEX composite rendering (e.g., HE-only bundles)",
    )
    s.add_argument(
        "--codex-scaler",
        dest="codex_scaler",
        default=None,
        help="Fitted CODEX scaler JSON; required unless CODEX rendering is disabled",
    )
    s.set_defaults(func=_cmd_build_cell_tables)

    # model selection: in-place migration of an existing bundle's
    # cells.parquet to carry multi-model prediction columns.
    s = sub.add_parser(
        "rebuild-cell-tables",
        help=(
            "Add multi-model prediction columns to an existing bundle's "
            "cells.parquet in place for Workbench model selection"
        ),
        description=(
            "Each --add-models argument is '<model_id>=<csv_path>'. "
            "The same CSV may be reused for multiple model ids; the "
            "loader picks ch??_pred_<model_id> columns out of the CSV "
            "for each merge. The manifest's `models` block is rewritten "
            "to include every model id present after the merge, with "
            "macro / per-marker / per-phenotype AP computed from the "
            "joined cells.parquet so the numbers and the columns can "
            "never drift out of sync."
        ),
    )
    s.add_argument("--bundle", required=True, help="Path to an existing bundle directory")
    s.add_argument(
        "--model-metadata",
        required=True,
        help="Schema-version-1 JSON covering every model retained in the bundle",
    )
    s.add_argument(
        "--add-models",
        dest="add_models",
        action="append",
        default=[],
        metavar="MODEL_ID=CSV_PATH",
        help=(
            "Model-id + predictions-CSV pair (repeatable). "
            "E.g., --add-models v1=preds.csv --add-models miphei_vit=preds.csv"
        ),
    )
    s.add_argument(
        "--default-model-id",
        dest="default_model_id",
        default=None,
        help=(
            "Which model id should be manifest.models[0] (the bundle's default). "
            "Defaults to the bundle's prior default, or v1_1 if none is set."
        ),
    )
    s.set_defaults(func=_cmd_rebuild_cell_tables)

    s = sub.add_parser(
        "rebuild-polygons",
        help="Add cell_polygons.parquet to an existing bundle in place",
    )
    s.add_argument("--bundle", required=True, help="Path to an existing bundle directory")
    s.add_argument(
        "--masks-dir",
        dest="masks_dir",
        default=None,
        help="Directory containing <basename>_nuclei.npy and _expanded8.npy "
        "(default: read from the bundle's manifest.json:sources.masks_dir)",
    )
    s.set_defaults(func=_cmd_rebuild_polygons)

    s = sub.add_parser(
        "rebuild-codex-composite",
        help="Render codex_composite.png + percentile cache for every core in a bundle",
    )
    s.add_argument("--bundle", required=True, help="Path to an existing bundle directory")
    s.add_argument(
        "--pairs-dir",
        dest="pairs_dir",
        required=True,
        help="Directory containing <basename>_CODEX.npy files",
    )
    s.add_argument(
        "--codex-scaler",
        dest="codex_scaler",
        required=True,
        help="Path to the fitted CODEX quantile-scaler JSON",
    )
    s.set_defaults(func=_cmd_rebuild_codex_composite)

    # Per-marker CODEX truth PNGs (sources for the per-marker DZI pyramids).
    s = sub.add_parser(
        "rebuild-codex-marker-thumbs",
        help=(
            "Render per-marker CODEX truth PNGs (ch{NN}_codex.png) for every "
            "core. These back the workbench's per-marker CODEX base layer."
        ),
    )
    s.add_argument("--bundle", required=True, help="Path to an existing bundle directory")
    s.add_argument(
        "--pairs-dir",
        dest="pairs_dir",
        default=None,
        help="Directory containing <basename>_CODEX.npy files "
        "(default: read from manifest.json:sources.pairs_dir)",
    )
    s.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render PNGs that already exist on disk (default: skip them)",
    )
    s.set_defaults(func=_cmd_rebuild_codex_marker_thumbs)

    s = sub.add_parser(
        "spatial-summary", help="Compute kNN graphs + COZI + hotspot stats for a bundle"
    )
    s.add_argument("--bundle", required=True)
    s.add_argument("--method", default="knn", choices=["knn", "radius"])
    s.add_argument("--k", type=int, default=16)
    s.add_argument("--radius-px", dest="radius_px", type=float, default=98.0)
    s.add_argument("--write-edges", dest="write_edges", action="store_true")
    s.add_argument("--model-id", dest="model_id", default="v1_1")
    s.set_defaults(func=_cmd_spatial_summary)

    s = sub.add_parser("report", help="Render per-core PDF report(s) from a bundle")
    s.add_argument("--bundle", required=True)
    s.add_argument("basenames", nargs="*", help="One or more core basenames; or pass --all")
    s.add_argument("--all", action="store_true", help="Render reports for every core in the bundle")
    s.add_argument("--model-id", dest="model_id", default="v1_1")
    s.add_argument(
        "--no-maps", action="store_true", help="Skip per-marker spatial maps section (smaller PDF)"
    )
    s.set_defaults(func=_cmd_report)

    s = sub.add_parser(
        "serve-workbench",
        help=(
            "Launch the HEXIF Spatial Cytometry Workbench: an interactive "
            "cell-level viewer over the parquet bundle, served at '/'."
        ),
    )
    s.add_argument("--bundle", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--model-id", dest="model_id", default="v1_1")
    s.set_defaults(func=_cmd_serve_workbench)

    # label audit label audit — diagnostic only, does not modify training labels.
    # See ``docs/reproducibility.md`` for the full spec.
    s = sub.add_parser(
        "audit-labels",
        help=(
            "Run the label audit cohort label audit: per-(core, marker) "
            "disagreement between the cohort-fit consensus pipeline, "
            "per-core Otsu, and Leiden-cluster z-score lenses."
        ),
        description=(
            "Diagnoses where the current cohort-fit threshold pipeline "
            "disagrees with two alternative cell-typing lenses (per-core "
            "Otsu refit, SPACEc-style Leiden clustering). Writes "
            "cohort_audit.{csv,md,json} under --out. Read-only: does not "
            "modify training labels. Spec: "
            "docs/reproducibility.md"
        ),
    )
    s.add_argument(
        "--bundle",
        required=True,
        help="Path to a bundle directory containing cells.parquet",
    )
    s.add_argument(
        "--thresholds",
        required=True,
        help="Path to thresholds.json (cohort-fit per-channel cutoffs in log1p space)",
    )
    s.add_argument(
        "--out",
        required=True,
        help="Output directory (created if missing); writes cohort_audit.{csv,md,json}",
    )
    # Hyperparameter overrides — defaults are pinned to the notebook so the
    # CD68/A-10 finding reproduces out of the box.
    from hexif.audit.types import (
        DEFAULT_LEIDEN_N_NEIGHBORS,
        DEFAULT_LEIDEN_PCA_N_COMPS,
        DEFAULT_LEIDEN_RESOLUTION,
        DEFAULT_RANDOM_STATE,
        DEFAULT_SCALE_MAX_VALUE,
        DEFAULT_ZSCORE_TAU,
    )

    s.add_argument(
        "--zscore-tau",
        dest="zscore_tau",
        type=float,
        default=DEFAULT_ZSCORE_TAU,
        help=f"Leiden cluster z-score positivity threshold (default: {DEFAULT_ZSCORE_TAU})",
    )
    s.add_argument(
        "--leiden-resolution",
        dest="leiden_resolution",
        type=float,
        default=DEFAULT_LEIDEN_RESOLUTION,
        help=f"Leiden resolution (default: {DEFAULT_LEIDEN_RESOLUTION})",
    )
    s.add_argument(
        "--leiden-n-neighbors",
        dest="leiden_n_neighbors",
        type=int,
        default=DEFAULT_LEIDEN_N_NEIGHBORS,
        help=f"Leiden n_neighbors (default: {DEFAULT_LEIDEN_N_NEIGHBORS})",
    )
    s.add_argument(
        "--leiden-pca-n-comps",
        dest="leiden_pca_n_comps",
        type=int,
        default=DEFAULT_LEIDEN_PCA_N_COMPS,
        help=f"PCA n_comps before Leiden (default: {DEFAULT_LEIDEN_PCA_N_COMPS})",
    )
    s.add_argument(
        "--scale-max-value",
        dest="scale_max_value",
        type=float,
        default=DEFAULT_SCALE_MAX_VALUE,
        help=f"sc.pp.scale max_value clip (default: {DEFAULT_SCALE_MAX_VALUE})",
    )
    s.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"random_state for sc.pp.neighbors and sc.tl.leiden (default: {DEFAULT_RANDOM_STATE})",
    )
    s.add_argument(
        "--cores",
        dest="cores",
        default=None,
        help="Optional CSV of basenames to restrict the audit to (e.g. 'ccRCC_TMA1__A-10,ccRCC_TMA1__B-3')",
    )
    s.add_argument(
        "--top-n-outliers",
        dest="top_n_outliers",
        type=int,
        default=20,
        help="N for the markdown top-N outlier tables (default: 20)",
    )
    s.add_argument(
        "--no-progress",
        dest="no_progress",
        action="store_true",
        help="Disable the per-core tqdm progress bar",
    )
    s.set_defaults(func=_cmd_audit_labels)

    # SPACEc gating SPACEc gating — rebuild training-set labels via per-core
    # cluster-then-label. See ``docs/reproducibility.md``.
    s = sub.add_parser(
        "build-spacec-labels",
        help=(
            "Rebuild cell-table positivity columns via SPACEc-style "
            "cluster-then-label (SPACEc gating fix for the cohort-threshold "
            "bidirectional disagreement diagnosed in label audit)."
        ),
        description=(
            "Iterates every core in --input, runs Leiden clustering in "
            "12-marker space per-core, derives per-cluster positivity "
            "(cluster mean z > tau), and appends chXX_pos_spacec / "
            "chXX_zscore_spacec / cluster_id_spacec / "
            "phenotype_<name>_call_spacec columns to the cell table. "
            "Existing chXX_pos / chXX_pos_consensus columns are "
            "preserved. Optional --reaudit re-runs the label audit audit on "
            "the rebuilt labels and checks that mean disagreement is < "
            "0.20 on every marker (the spec's acceptance gate). Spec: "
            "docs/reproducibility.md"
        ),
    )
    s.add_argument(
        "--input",
        required=True,
        help="Path to the cell table (parquet or CSV) to rebuild labels from",
    )
    s.add_argument(
        "--output",
        required=True,
        help=(
            "Path to write the augmented cell table. Suffix must match "
            "--input (no format conversion). Written atomically."
        ),
    )
    from hexif.audit.types import (
        DEFAULT_LEIDEN_N_NEIGHBORS,
        DEFAULT_LEIDEN_PCA_N_COMPS,
        DEFAULT_LEIDEN_RESOLUTION,
        DEFAULT_RANDOM_STATE,
        DEFAULT_SCALE_MAX_VALUE,
        DEFAULT_ZSCORE_TAU,
    )

    s.add_argument(
        "--zscore-tau",
        dest="zscore_tau",
        type=float,
        default=DEFAULT_ZSCORE_TAU,
        help=f"Cluster mean z-score positivity threshold (default: {DEFAULT_ZSCORE_TAU})",
    )
    s.add_argument(
        "--leiden-resolution",
        dest="leiden_resolution",
        type=float,
        default=DEFAULT_LEIDEN_RESOLUTION,
        help=f"Leiden resolution (default: {DEFAULT_LEIDEN_RESOLUTION})",
    )
    s.add_argument(
        "--leiden-n-neighbors",
        dest="leiden_n_neighbors",
        type=int,
        default=DEFAULT_LEIDEN_N_NEIGHBORS,
        help=f"Leiden n_neighbors (default: {DEFAULT_LEIDEN_N_NEIGHBORS})",
    )
    s.add_argument(
        "--leiden-pca-n-comps",
        dest="leiden_pca_n_comps",
        type=int,
        default=DEFAULT_LEIDEN_PCA_N_COMPS,
        help=f"PCA n_comps before Leiden (default: {DEFAULT_LEIDEN_PCA_N_COMPS})",
    )
    s.add_argument(
        "--scale-max-value",
        dest="scale_max_value",
        type=float,
        default=DEFAULT_SCALE_MAX_VALUE,
        help=f"sc.pp.scale max_value clip (default: {DEFAULT_SCALE_MAX_VALUE})",
    )
    s.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"random_state for sc.pp.neighbors and sc.tl.leiden (default: {DEFAULT_RANDOM_STATE})",
    )
    s.add_argument(
        "--cores",
        dest="cores",
        default=None,
        help=(
            "Optional CSV of basenames to restrict the rebuild to "
            "(e.g. 'ccRCC_TMA1__A-10,ccRCC_TMA1__B-3')"
        ),
    )
    s.add_argument(
        "--no-progress",
        dest="no_progress",
        action="store_true",
        help="Disable the per-core tqdm progress bar",
    )
    s.add_argument(
        "--write-provenance",
        dest="write_provenance",
        action="store_true",
        help=(
            "Also write spacec_labels_provenance.json under the output's "
            "parent directory (parameter dict, git rev, timestamp, "
            "scanpy+igraph versions, per-marker positive counts)."
        ),
    )
    s.add_argument(
        "--reaudit",
        dest="reaudit",
        action="store_true",
        help=(
            "After rebuilding labels, run the label audit audit (audit_cohort) "
            "with chXX_pos_consensus replaced by the new chXX_pos_spacec "
            "columns. Writes cohort_audit.{csv,md,json} under "
            "<output_parent>/cohort_audit_on_spacec_labels/."
        ),
    )
    s.add_argument(
        "--reaudit-thresholds",
        dest="reaudit_thresholds",
        default=None,
        help=(
            "Path to the cohort thresholds JSON (same file the original "
            "label audit audit consumed). Required when --reaudit is set."
        ),
    )
    s.add_argument(
        "--max-mean-disagreement",
        type=float,
        default=None,
        help="Predeclared acceptance limit; required with --reaudit.",
    )
    s.set_defaults(func=_cmd_build_spacec_labels)

    s = sub.add_parser("codex-register", help="Register CODEX TMA cores to H&E crops")
    target = s.add_mutually_exclusive_group()
    target.add_argument("--all", action="store_true", help="Process all configured TMAs")
    target.add_argument("--tma", help="Process one configured TMA, e.g. ccOC_TMA1")
    s.add_argument(
        "--codex-root",
        required=True,
        help="Root containing HandE/ and CODEX scan directories",
    )
    s.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for paired arrays and manifests",
    )
    s.add_argument(
        "--tma-map",
        required=True,
        help="Private YAML mapping TMA IDs to H&E slide and CODEX subdirectory",
    )
    s.add_argument(
        "--force", action="store_true", help="Recompute even if he.npy/codex.npy are cached"
    )
    s.add_argument(
        "--retry-bad",
        action="store_true",
        help="Delete prior bad/empty/high-residual rows and retry them",
    )
    s.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Only rebuild manifest_all.csv from per-TMA manifests",
    )
    s.add_argument(
        "--purge-rejected-arrays",
        action="store_true",
        help="Delete stale he.npy/codex.npy for rejected rows",
    )
    s.add_argument(
        "--bad-align-px",
        type=float,
        default=30.0,
        help="Reject VALIS rigid residuals above this value",
    )
    s.add_argument(
        "--empty-he-sat",
        type=float,
        default=2.0,
        help="Mean HSV saturation below this is empty H&E",
    )
    s.add_argument(
        "--empty-dapi-mean", type=float, default=30.0, help="Mean DAPI below this is empty CODEX"
    )
    s.set_defaults(func=_cmd_codex_register)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
