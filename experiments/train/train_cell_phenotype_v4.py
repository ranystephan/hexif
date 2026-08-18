#!/usr/bin/env python3
"""Train the focal-loss HEXIF cell-phenotype configuration.

Swaps CTransPath (Swin-Tiny, 28M) for a pathology foundation model
(UNI2 ViT-Giant-d1536-d24 with SwiGLU + 8 register tokens, 681M params,
or H-optimus-0 ViT-Giant, ~1.1B). LoRA on every attention block (rank 8).
Per-block gradient checkpointing keeps activations within a 12-24 GB
VRAM budget. Gradient accumulation preserves the effective batch size.

The shared training loop (dataset construction, focal-BCE + hierarchy
loss, optimizer setup, AMP scaler, validation, early stop, metric
tables) lives in :mod:`hexif.training.cell_phenotype`. This script only
owns argparse defaults + the v4 model build.

UNI2 / H-optimus-0 weights are gated on HuggingFace; run
``hf auth login`` with a token that has accepted the model card terms
before the first run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn

from hexif.cell_model import (
    ENCODER_CHOICES,
    ENCODER_REVISIONS,
    CellPhenotypeModel,
    apply_lora_to_vit,
    build_encoder,
)
from hexif.training.cell_phenotype import (
    add_shared_args,
    apply_smoke_overrides,
    parse_int_list,
    set_seeds,
    train,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # The smaller micro-batch and gradient accumulation fit large encoders
    # while preserving the intended effective batch size.
    add_shared_args(
        parser,
        default_batch=32,
        default_grad_accum=4,
    )
    parser.add_argument(
        "--encoder",
        choices=ENCODER_CHOICES,
        default="uni2",
        help=(
            "Which v4 encoder to use. 'uni2' (Mahmood lab ViT-G/14, gated) "
            "or 'h_optimus_0' (Bioptimus ViT-Giant, gated) for production; "
            "'vit_base_smoke' (open) for tests + CI."
        ),
    )
    parser.add_argument(
        "--no_grad_checkpointing",
        action="store_true",
        help=(
            "Disable per-block gradient checkpointing on the encoder. Default "
            "is enabled — required to fit ViT-Huge / Giant on a 12-24 GB GPU "
            "at micro-batch 32. Pass this only if you have plenty of VRAM "
            "(80+ GB) and want max throughput."
        ),
    )
    return parser.parse_args()


def _build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """v4 path — UNI2 / H-optimus-0 / open-substitute ViT + LoRA + CellPhenotypeModel.

    LoRA matches every ``attn.qkv`` and ``attn.proj`` Linear (= depth × 2
    modules) and freezes the base; only the LoRA A/B matrices and the head
    are trainable. Per-block gradient checkpointing is configured at
    construction time on the encoder via ``set_grad_checkpointing(True)``.
    """
    encoder = build_encoder(args.encoder, pretrained=True)
    n_lora = apply_lora_to_vit(encoder, rank=int(args.lora_rank), alpha=float(args.lora_alpha))
    if n_lora == 0:
        raise RuntimeError(
            f"LoRA matched 0 modules on encoder {args.encoder!r} — expected "
            "ViT blocks with .attn.qkv and .attn.proj names."
        )
    for n, p in encoder.named_parameters():
        p.requires_grad = ".A." in n or ".B." in n

    use_ckpt = not bool(args.no_grad_checkpointing)
    model = CellPhenotypeModel(
        encoder=encoder,
        embed_dim=int(encoder.embed_dim),
        n_pred=12,
        n_core_mean=12,
        head_hidden=tuple(parse_int_list(args.mlp_hidden)),
        dropout=float(args.dropout),
        use_gradient_checkpointing=use_ckpt,
    ).to(device)

    n_lora_params = sum(p.numel() for n, p in model.named_parameters() if ".A." in n or ".B." in n)
    n_head_params = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info: dict[str, Any] = {
        "encoder": args.encoder,
        "encoder_revision": ENCODER_REVISIONS[args.encoder],
        "encoder_embed_dim": int(encoder.embed_dim),
        "encoder_chs": [int(encoder.embed_dim)],
        "use_gradient_checkpointing": use_ckpt,
        "n_lora_modules": n_lora,
        "n_lora_params": n_lora_params,
        "n_head_params": n_head_params,
        "n_trainable": n_total,
    }
    return model, info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    args = _parse_args()
    if args.smoke:
        apply_smoke_overrides(args)
    # Seed all RNGs BEFORE model construction — LoRA A/B matrices are
    # randomly initialized, so seeding after construction silently gives
    # non-reproducible weights across runs.
    set_seeds(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    marker_channels = parse_int_list(args.marker_channels)

    model, info = _build_model(args, device)
    train(
        model,
        args=args,
        info=info,
        marker_channels=marker_channels,
        device=device,
    )


if __name__ == "__main__":
    main()
