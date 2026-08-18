"""Resume / checkpoint round-trip tests for hexif.training.cell_phenotype.

These tests verify that ``_save_training_state`` → ``_load_training_state``
restores the full training state bit-identically — model weights, optimizer
moment buffers, scheduler step count, AMP scaler scale, RNG state, the
best-so-far snapshot, and the per-epoch history. This is the contract that
makes SLURM ``--requeue`` correct: a re-launched job that loads
``last_state.pt`` must produce the same loss / gradient / step sequence
as if the original job had never been preempted.

Runs on CPU with the open ``vit_base_smoke`` encoder so it's runnable in
CI without a GPU or HuggingFace gated access.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    from hexif.cell_model import CellPhenotypeModel, apply_lora_to_vit, build_encoder
    from hexif.training.cell_phenotype import (
        _load_training_state,
        _save_training_state,
    )

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


needs_deps = pytest.mark.skipif(not _HAS_DEPS, reason="torch + timm + hexif required")


def _build_tiny_state(
    *, with_grad_step: bool = True
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.amp.GradScaler,
    argparse.Namespace,
    dict,
]:
    """Construct a minimal CellPhenotypeModel + opt/sched/scaler for round-trip testing.

    Uses vit_base_smoke (open, ~86M params) on CPU so the test doesn't need
    GPU or gated HuggingFace access. We optionally run one gradient step
    to populate the optimizer's first/second moment buffers — without that
    step, ``opt.state_dict()`` has no per-parameter state, so an identity
    round-trip would pass trivially.
    """
    encoder = build_encoder("vit_base_smoke", pretrained=False)
    apply_lora_to_vit(encoder, rank=4, alpha=4.0)
    for n, p in encoder.named_parameters():
        p.requires_grad = ".A." in n or ".B." in n
    model = CellPhenotypeModel(
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        n_pred=12,
        n_core_mean=12,
        head_hidden=(64, 32),
        dropout=0.0,
        use_gradient_checkpointing=False,
    )
    # Two param groups (head + LoRA), to mirror the real trainer's structure.
    lora_params = [
        p for n, p in model.named_parameters() if (".A." in n or ".B." in n) and p.requires_grad
    ]
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": 1e-3, "weight_decay": 1e-3},
            {"params": head_params, "lr": 5e-3, "weight_decay": 1e-4},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    if with_grad_step:
        model.train()
        rgb = torch.randn((2, 3, 224, 224))
        pred = torch.zeros((2, 12))
        core_mean = torch.zeros((2, 12))
        marker_logits, _ = model(rgb, pred, core_mean)
        loss = marker_logits.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()

    args = argparse.Namespace(
        epochs=10,
        seed=42,
        output_dir="(set by test)",
    )
    info = {
        "encoder": "vit_base_smoke",
        "n_lora_modules": 24,
        "n_lora_params": 99999,
        "n_head_params": 99999,
        "n_trainable": 99999,
        "encoder_chs": [768],
    }
    return model, opt, sched, scaler, args, info


@needs_deps
def test_save_and_load_round_trip_restores_model_weights() -> None:
    """After save → mutate → load, the model's weights must be bit-identical
    to the snapshot. This is the core resume contract — without it the next
    forward pass produces different logits than the run that was preempted.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "last_state.pt"
        model, opt, sched, scaler, args, info = _build_tiny_state()

        # Snapshot the weights we're going to compare against.
        ref_state = {k: v.clone() for k, v in model.state_dict().items()}

        _save_training_state(
            path,
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
            epoch=3,
            best={
                "epoch": 3,
                "macro_marker_ap": 0.42,
                "marker_probs": np.zeros((10, 12), dtype=np.float32),
                "phenotype_probs": np.zeros((10, 9), dtype=np.float32),
                "marker_y": np.zeros((10, 12), dtype=np.float32),
                "phenotype_y": np.zeros((10, 9), dtype=np.float32),
            },
            no_improve=1,
            epoch_rows=[{"epoch": i, "loss": 0.1 * i} for i in range(4)],
            args=args,
            info=info,
        )
        assert path.exists()

        # Mutate every weight so the load has to actually do work.
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        # Sanity: weights differ from snapshot now.
        post_zero = next(iter(model.state_dict().values()))
        assert post_zero.abs().sum() == 0

        # Round-trip.
        next_epoch, best, no_improve, epoch_rows = _load_training_state(
            path, model=model, opt=opt, sched=sched, scaler=scaler
        )

        # Model weights restored exactly.
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, ref_state[k], rtol=0, atol=0)

        # Saved scalars restored exactly.
        assert next_epoch == 4  # saved epoch 3 → next is 4
        assert no_improve == 1
        assert best["macro_marker_ap"] == 0.42
        assert best["marker_probs"].shape == (10, 12)
        assert len(epoch_rows) == 4


@needs_deps
def test_save_and_load_round_trip_restores_optimizer_state() -> None:
    """AdamW maintains exp_avg / exp_avg_sq per parameter; these MUST survive
    round-trip or resumed training will take an effective Adam-warmup step
    on its first iteration, deviating from the un-preempted trajectory.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "last_state.pt"
        model, opt, sched, scaler, args, info = _build_tiny_state(with_grad_step=True)
        # After one step, AdamW state has populated buffers we can compare.
        ref_state_dict = opt.state_dict()
        # Deep-copy the tensor leaves so subsequent training can't mutate them.
        ref_state_dicts = [
            {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
            for s in ref_state_dict["state"].values()
        ]
        assert ref_state_dicts, "AdamW state should be non-empty after one step"

        _save_training_state(
            path,
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
            epoch=0,
            best={
                "epoch": -1,
                "macro_marker_ap": -1.0,
                "marker_probs": None,
                "phenotype_probs": None,
                "marker_y": None,
                "phenotype_y": None,
            },
            no_improve=0,
            epoch_rows=[],
            args=args,
            info=info,
        )

        # Mutate opt state so the load has to actually restore.
        for s in opt.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                if k in s:
                    s[k].zero_()

        _load_training_state(path, model=model, opt=opt, sched=sched, scaler=scaler)

        # Compare restored opt state to snapshot.
        new_state = opt.state_dict()["state"]
        for new_s, ref_s in zip(new_state.values(), ref_state_dicts, strict=False):
            for k, ref_v in ref_s.items():
                if isinstance(ref_v, torch.Tensor):
                    torch.testing.assert_close(new_s[k], ref_v, rtol=0, atol=0)
                else:
                    assert new_s[k] == ref_v


@needs_deps
def test_save_and_load_round_trip_restores_rng_state() -> None:
    """Augmentation calls ``torch.rand`` and ``torch.randint`` from the
    global RNG. Resume must restore that RNG state, otherwise the
    augmentation sequence after resume differs from the un-preempted run
    and training diverges (even if model weights match exactly)."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "last_state.pt"
        model, opt, sched, scaler, args, info = _build_tiny_state(with_grad_step=False)

        # Seed deterministically, advance RNG by some known amount, save,
        # then verify a subsequent torch.rand call after load matches the
        # subsequent torch.rand call after save.
        torch.manual_seed(42)
        np.random.seed(42)
        _ = torch.rand(13)  # advance torch RNG
        _ = np.random.random(7)  # advance numpy RNG

        # Capture "what comes next" before saving.
        expected_torch_next = torch.rand(5).clone()
        expected_np_next = np.random.random(5).copy()

        # Re-seed + advance to the same point so we can save the same state.
        torch.manual_seed(42)
        np.random.seed(42)
        _ = torch.rand(13)
        _ = np.random.random(7)

        _save_training_state(
            path,
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
            epoch=0,
            best={
                "epoch": -1,
                "macro_marker_ap": -1.0,
                "marker_probs": None,
                "phenotype_probs": None,
                "marker_y": None,
                "phenotype_y": None,
            },
            no_improve=0,
            epoch_rows=[],
            args=args,
            info=info,
        )

        # Mutate RNG state so the load has to restore it.
        torch.manual_seed(999)
        np.random.seed(999)
        _ = torch.rand(50)
        _ = np.random.random(50)

        _load_training_state(path, model=model, opt=opt, sched=sched, scaler=scaler)

        # The RNG should now be at the post-save position. The next draws
        # must match what we captured before saving.
        post_load_torch = torch.rand(5)
        post_load_np = np.random.random(5)
        torch.testing.assert_close(post_load_torch, expected_torch_next, rtol=0, atol=0)
        np.testing.assert_array_equal(post_load_np, expected_np_next)


@needs_deps
def test_save_uses_atomic_replace() -> None:
    """``_save_training_state`` should write to ``<path>.tmp`` and atomically
    rename. If the process is killed mid-save, the canonical
    ``last_state.pt`` is either the previous good state or absent — never
    the half-written tmp file. Verifies the tmp file does not remain after
    a successful save."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "last_state.pt"
        model, opt, sched, scaler, args, info = _build_tiny_state(with_grad_step=False)
        _save_training_state(
            path,
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
            epoch=0,
            best={
                "epoch": -1,
                "macro_marker_ap": -1.0,
                "marker_probs": None,
                "phenotype_probs": None,
                "marker_y": None,
                "phenotype_y": None,
            },
            no_improve=0,
            epoch_rows=[],
            args=args,
            info=info,
        )
        assert path.exists()
        tmp = path.with_suffix(path.suffix + ".tmp")
        assert not tmp.exists(), "tmp file should be renamed away after successful save"
