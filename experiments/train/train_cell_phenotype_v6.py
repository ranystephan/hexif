#!/usr/bin/env python3
"""Train the asymmetric-loss HEXIF cell-phenotype configuration.

Identical architecture as v4 (UNI2 / H-optimus-0 encoder + LoRA +
cell-token pool + fusion features + ``CellPhenotypeV1Head`` parallel
marker / phenotype heads). The only change is the marker loss:
class-balanced focal BCE → Asymmetric Loss (Ridnik et al. 2021,
arXiv 2009.14119).

The loss follows Ridnik et al., 2021 (arXiv:2009.14119). The focal and
asymmetric configurations share the same architecture so their next
real-data rerun can isolate the loss choice.
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
    # Defaults mirror v4 (UNI2 / H-opt, SPACEc labels, owners-partition-
    # friendly output dir). Only the marker-loss family changes:
    # ``--marker_loss asl`` is the v6-specific override below.
    add_shared_args(
        parser,
        default_batch=32,
        default_grad_accum=4,
    )
    parser.add_argument(
        "--encoder",
        choices=ENCODER_CHOICES,
        default="h_optimus_0",
        help=(
            "Encoder backbone. 'h_optimus_0' and 'uni2' are research "
            "configurations; 'vit_base_smoke' is restricted to tests."
        ),
    )
    parser.add_argument(
        "--no_grad_checkpointing",
        action="store_true",
        help=(
            "Disable per-block gradient checkpointing on the encoder. "
            "Default enabled — required to fit ViT-Huge / Giant on a "
            "12-24 GB GPU at micro-batch 32."
        ),
    )
    return parser.parse_args()


def _build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """v6 path — CellPhenotypeModel verbatim (loss change only is loss-side, not architecture)."""
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
        "marker_loss": args.marker_loss,
        "asl_gamma_pos": float(args.asl_gamma_pos),
        "asl_gamma_neg": float(args.asl_gamma_neg),
        "asl_clip": float(args.asl_clip),
    }
    return model, info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    args = _parse_args()
    if args.smoke:
        apply_smoke_overrides(args)
    # v6-specific defaults: force ``--marker_loss asl`` unless explicitly
    # overridden on the CLI. (Useful: ``--marker_loss focal`` lets v6
    # reproduce v4 H-opt byte-for-byte as a regression check.)
    if not any(arg.startswith("--marker_loss") for arg in sys.argv[1:]):
        args.marker_loss = "asl"
    set_seeds(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    marker_channels = parse_int_list(args.marker_channels)

    model, info = _build_model(args, device)
    logging.info(
        "v6 marker_loss=%s (γ⁺=%.2f, γ⁻=%.2f, clip=%.3f)",
        args.marker_loss,
        float(args.asl_gamma_pos),
        float(args.asl_gamma_neg),
        float(args.asl_clip),
    )
    train(
        model,
        args=args,
        info=info,
        marker_channels=marker_channels,
        device=device,
    )


if __name__ == "__main__":
    main()
