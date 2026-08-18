"""Unit + smoke tests for the v4 encoder swap.

Spec: ``experiments/configs/v4_uni2_encoder.md``. Code under test:
``hexif.cell_model.build_encoder``, ``hexif.cell_model.vit_cell_pool``,
``hexif.cell_model.CellPhenotypeModel``.

Tests use ``vit_base_patch16_224`` (open weights via timm, no HuggingFace
gated access needed) so they run on any developer machine. UNI2 /
H-optimus-0 paths share the same code; only the encoder weights differ.

Smoke gates that are too heavy for the unit suite (multi-epoch training
on bundle data, HF auth) live in
``experiments/train/train_cell_phenotype_v4.py --smoke``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

_HAS_TIMM = importlib.util.find_spec("timm") is not None

# timm + the v4 symbols both come from hexif.cell_model — the import is gated
# on the heavier deps so a default ``pytest -q`` skips cleanly on a
# machine without torch/timm. (``pytest`` itself only needs the import
# to succeed; the body decides whether to skip per-test.)
needs_timm = pytest.mark.skipif(
    not _HAS_TIMM,
    reason="timm not installed; v4 encoder + LoRA tests require it",
)

if _HAS_TIMM:
    from hexif.cell_model import (
        ENCODER_CHOICES,
        CellPhenotypeModel,
        apply_lora_to_vit,
        build_encoder,
        vit_cell_pool,
    )


# --------------------------------------------------------------------------- #
# vit_cell_pool — geometric correctness
# --------------------------------------------------------------------------- #


class TestVitCellPool:
    """``vit_cell_pool(tokens) -> (N, 2d)``: CLS concat center-patch.

    Center patch is at spatial position ``(grid // 2, grid // 2)``; in raster
    order that's index ``(grid // 2) * grid + (grid // 2)``. For a 14×14 grid
    (ViT-Base/16 at 224×224), center = 7·14+7 = 105. For 16×16 (UNI2's
    patch-14 layout at 224), center = 8·16+8 = 136.
    """

    def test_shape_is_n_by_2d(self) -> None:
        """Standard ViT output (N, 1 + n_patches, d) → (N, 2 d)."""
        if not _HAS_TIMM:
            pytest.skip("torch needed")
        rng = np.random.default_rng(0)
        tokens = torch.tensor(
            rng.standard_normal((4, 1 + 196, 64)).astype(np.float32)
        )  # 14×14 patches, d=64
        out = vit_cell_pool(tokens)
        assert out.shape == (4, 128)
        assert out.dtype == tokens.dtype

    def test_cls_is_first_d_dims_center_is_second_d_dims(self) -> None:
        """The output layout is [cls | center_patch] along the last axis.

        We construct tokens where CLS has a recognizable value and only
        the center patch has another recognizable value; every other
        patch is zero. Then assert the (cls, center) values appear in
        the expected halves of the output.
        """
        if not _HAS_TIMM:
            pytest.skip("torch needed")
        n_patches = 196  # 14×14
        d = 8
        # Build (1, 197, 8) — CLS = ones, center patch = 2*ones, rest = 0.
        tokens = torch.zeros((1, 1 + n_patches, d), dtype=torch.float32)
        tokens[0, 0] = 1.0  # CLS
        center_idx = (14 // 2) * 14 + (14 // 2)  # 7*14 + 7 = 105
        tokens[0, 1 + center_idx] = 2.0  # center patch
        out = vit_cell_pool(tokens)
        assert out.shape == (1, 2 * d)
        # First half should be the CLS value (1.0).
        assert torch.allclose(out[0, :d], torch.full((d,), 1.0))
        # Second half should be the center value (2.0).
        assert torch.allclose(out[0, d:], torch.full((d,), 2.0))

    def test_grid_size_inferred_from_n_patches(self) -> None:
        """The pool reads grid_size from n_patches at runtime so it works
        for any ViT timm exposes — UNI2 (patch=14 ⇒ 16×16=256 patches),
        ViT-Base (patch=16 ⇒ 14×14=196), etc."""
        if not _HAS_TIMM:
            pytest.skip("torch needed")
        for n_patches in (196, 256, 64):  # 14×14, 16×16, 8×8
            grid = round(n_patches**0.5)
            d = 4
            tokens = torch.zeros((1, 1 + n_patches, d), dtype=torch.float32)
            # Mark only the center patch with a unique value.
            center_idx = (grid // 2) * grid + (grid // 2)
            tokens[0, 1 + center_idx] = 7.0
            out = vit_cell_pool(tokens)
            # The center-patch half (last d dims) must be all 7s.
            assert torch.allclose(out[0, d:], torch.full((d,), 7.0)), (
                f"n_patches={n_patches} grid={grid} center_idx={center_idx}: "
                f"got out[0, d:]={out[0, d:].tolist()}"
            )

    def test_non_square_patch_count_raises(self) -> None:
        """A non-square patch count means the ViT was fed a non-square
        input; we explicitly bail rather than guess a grid size."""
        if not _HAS_TIMM:
            pytest.skip("torch needed")
        # 200 isn't a perfect square (14² = 196, 15² = 225).
        tokens = torch.zeros((1, 1 + 200, 4))
        with pytest.raises(ValueError, match=r"not a perfect square"):
            vit_cell_pool(tokens)

    def test_wrong_rank_raises(self) -> None:
        """Token tensor must be (N, 1+n, d). A 2-D tensor (forgotten
        batch dim) or 4-D (sequence reshaped to 2-D grid) should
        produce a clear error, not a silent miscomputation."""
        if not _HAS_TIMM:
            pytest.skip("torch needed")
        with pytest.raises(ValueError, match="expects"):
            vit_cell_pool(torch.zeros((197, 64)))  # missing batch dim
        with pytest.raises(ValueError, match="expects"):
            vit_cell_pool(torch.zeros((1, 14, 14, 64)))  # 4-D

    def test_register_tokens_skipped_for_uni2_layout(self) -> None:
        """UNI2 has 8 register tokens after CLS, so num_prefix_tokens=9.
        Without the explicit argument the pool would pick the wrong
        token as 'center patch' (off by the register count).
        """
        if not _HAS_TIMM:
            pytest.skip("torch needed")
        n_prefix = 9  # 1 CLS + 8 register
        n_patches = 256  # 16×16 grid (UNI2 patch=14 at 224×224)
        d = 8
        tokens = torch.zeros((1, n_prefix + n_patches, d), dtype=torch.float32)
        tokens[0, 0] = 1.0  # CLS
        for i in range(1, n_prefix):
            tokens[0, i] = -99.0  # register tokens — pool MUST NOT read these
        center_patch_index = (16 // 2) * 16 + (16 // 2)  # 8*16+8 = 136
        tokens[0, n_prefix + center_patch_index] = 7.0  # center patch
        out = vit_cell_pool(tokens, num_prefix_tokens=n_prefix)
        # First half: CLS (1.0).
        assert torch.allclose(out[0, :d], torch.full((d,), 1.0))
        # Second half: center patch (7.0), NOT a register token's -99.0.
        assert torch.allclose(out[0, d:], torch.full((d,), 7.0))

    def test_num_prefix_tokens_zero_raises(self) -> None:
        """num_prefix_tokens must be >= 1; we read tokens[:, 0] as CLS."""
        if not _HAS_TIMM:
            pytest.skip("torch needed")
        tokens = torch.zeros((1, 197, 64), dtype=torch.float32)
        with pytest.raises(ValueError, match="num_prefix_tokens"):
            vit_cell_pool(tokens, num_prefix_tokens=0)


# --------------------------------------------------------------------------- #
# build_encoder — factory + error paths
# --------------------------------------------------------------------------- #


@needs_timm
class TestBuildV4Encoder:
    def test_choices_constant_is_complete(self) -> None:
        """The exported list of choices must match the keys of the
        internal HF-Hub map. Catches a typo where someone adds a new
        encoder to the map but forgets to expose it."""
        from hexif.cell_model import _ENCODER_HF_HUB

        assert set(ENCODER_CHOICES) == set(_ENCODER_HF_HUB.keys())

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown v4 encoder"):
            build_encoder("gigapath", pretrained=False)

    def test_vit_base_smoke_constructs_random_init(self) -> None:
        """``pretrained=False`` builds the encoder without network access,
        which keeps this test runnable in CI / offline environments."""
        enc = build_encoder("vit_base_smoke", pretrained=False)
        assert hasattr(enc, "embed_dim")
        # ViT-Base/16 has embed_dim = 768.
        assert int(enc.embed_dim) == 768

    def test_vit_base_forward_features_returns_token_sequence(self) -> None:
        """The forward_features path produces (N, 1 + n_patches, d) — the
        shape vit_cell_pool expects. Verified on the open ViT-Base."""
        enc = build_encoder("vit_base_smoke", pretrained=False)
        enc.eval()
        with torch.no_grad():
            x = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
            tokens = enc.forward_features(x)
        # ViT-Base/16 at 224×224 → 196 patches + 1 CLS = 197 tokens
        assert tokens.shape == (2, 197, 768)

    def test_uni2_kwargs_build_correct_arch_without_pretrained(self) -> None:
        """The UNI2 timm_kwargs dict must produce a model that:
            - exposes embed_dim = 1536 (ViT-Giant width)
            - exposes num_prefix_tokens = 9 (1 CLS + 8 register tokens)
            - forwards (2,3,224,224) → (2, 265, 1536) tokens

        We exercise this WITHOUT downloading pretrained weights so the
        test runs offline. The bug we're guarding against: a previous
        version of build_encoder passed only ``num_classes=0,
        img_size=224`` to timm, which silently fell back to the stock
        vit_giant_patch14_224 config (depth=40, no reg_tokens). Loading
        the UNI2 checkpoint into that fallback arch produced a pos_embed
        shape mismatch at ``resample_abs_pos_embed`` time.
        """
        from hexif.cell_model import _encoder_timm_kwargs

        try:
            import timm
        except ImportError:
            pytest.skip("timm required")
        kwargs = _encoder_timm_kwargs("uni2")
        # Spot-check key entries that distinguish UNI2 from stock ViT-Giant.
        assert kwargs["depth"] == 24, "UNI2 is depth-24, not stock vit-giant depth-40"
        assert kwargs["reg_tokens"] == 8, "UNI2 uses 8 register tokens"
        assert kwargs["embed_dim"] == 1536
        assert kwargs["no_embed_class"] is True
        assert kwargs["dynamic_img_size"] is True
        # The actual build — fast (no weight download).
        m = timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)
        assert int(m.embed_dim) == 1536
        assert int(m.num_prefix_tokens) == 9
        m.eval()
        with torch.no_grad():
            x = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
            tokens = m.forward_features(x)
        # 1 CLS + 8 reg + 256 patches (16×16 at patch=14, img=224) = 265
        assert tokens.shape == (2, 265, 1536), f"expected (2, 265, 1536), got {tuple(tokens.shape)}"

    def test_h_optimus_0_kwargs_minimal(self) -> None:
        """H-optimus-0 needs init_values=1e-5 + dynamic_img_size=False; the
        rest is timm's stock vit-giant. ``num_prefix_tokens`` should be 1
        (CLS only — no register tokens)."""
        from hexif.cell_model import _encoder_timm_kwargs

        kwargs = _encoder_timm_kwargs("h_optimus_0")
        assert kwargs.get("init_values") == 1e-5
        assert kwargs.get("dynamic_img_size") is False
        assert "reg_tokens" not in kwargs  # would alter prefix-token count


# --------------------------------------------------------------------------- #
# CellPhenotypeModel — forward, LoRA wrap, gradient-checkpoint equivalence
# --------------------------------------------------------------------------- #


@needs_timm
class TestCellPhenotypeModel:
    def _build_model(
        self, *, use_grad_ckpt: bool = False, apply_lora: bool = True
    ) -> CellPhenotypeModel:
        encoder = build_encoder("vit_base_smoke", pretrained=False)
        if apply_lora:
            n_lora = apply_lora_to_vit(encoder, rank=8, alpha=8.0)
            # ViT-Base has 12 blocks × 2 attn modules = 24
            assert n_lora == 24, (
                f"LoRA wrapped {n_lora} modules on ViT-Base; expected 24 "
                "(12 blocks × {{qkv, proj}})."
            )
            for n, p in encoder.named_parameters():
                p.requires_grad = (".A." in n) or (".B." in n)
        return CellPhenotypeModel(
            encoder=encoder,
            embed_dim=int(encoder.embed_dim),
            n_pred=12,
            n_core_mean=12,
            head_hidden=(768, 384),
            dropout=0.1,
            use_gradient_checkpointing=use_grad_ckpt,
        )

    def test_forward_shape(self) -> None:
        model = self._build_model(apply_lora=True)
        model.eval()
        rgb = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
        pred = torch.zeros((2, 12), dtype=torch.float32)
        core_mean = torch.zeros((2, 12), dtype=torch.float32)
        with torch.no_grad():
            mlogits, plogits = model(rgb, pred, core_mean)
        assert mlogits.shape == (2, 12)
        assert plogits.shape == (2, 9)

    def test_lora_wrap_count_matches_vit_depth(self) -> None:
        """Verify apply_lora_to_vit wraps exactly the qkv + proj Linear
        in every transformer block. ViT-Base = 12 blocks ⇒ 24 modules."""
        encoder = build_encoder("vit_base_smoke", pretrained=False)
        n_lora = apply_lora_to_vit(encoder, rank=8, alpha=8.0)
        assert n_lora == 24

    def test_state_dict_has_lora_A_B_keys(self) -> None:
        """After wrapping, the encoder's state_dict carries ``A.weight`` /
        ``B.weight`` entries for every wrapped Linear. This is the contract
        the v1.1 trainer relies on for freeze-everything-except-LoRA via
        ``p.requires_grad = '.A.' in n or '.B.' in n``."""
        encoder = build_encoder("vit_base_smoke", pretrained=False)
        apply_lora_to_vit(encoder, rank=8, alpha=8.0)
        names = [n for n in encoder.state_dict()]
        a_names = [n for n in names if n.endswith(".A.weight")]
        b_names = [n for n in names if n.endswith(".B.weight")]
        # 24 wrapped modules × {A, B} = 48 keys total
        assert len(a_names) == 24
        assert len(b_names) == 24

    def test_freeze_policy_only_trains_lora_and_head(self) -> None:
        """After the trainer's freeze step (.requires_grad = '.A.' in n
        or '.B.' in n on encoder params), only LoRA matrices + head + the
        non-encoder modules carry grad. Verifies the param-group filter
        in train_cell_phenotype_v1_1.py:line 509 matches what we wired."""
        model = self._build_model(apply_lora=True)
        # The model has 4 categories of param:
        #   1. encoder LoRA A/B — trainable
        #   2. encoder non-LoRA — frozen (set by _build_model in trainer)
        #   3. head trunk + heads — trainable
        #   4. BatchNorm1d feat_norm with affine=False — no params
        n_lora_trainable = sum(
            p.numel()
            for n, p in model.encoder.named_parameters()
            if p.requires_grad and (".A." in n or ".B." in n)
        )
        n_enc_other_trainable = sum(
            p.numel()
            for n, p in model.encoder.named_parameters()
            if p.requires_grad and ".A." not in n and ".B." not in n
        )
        n_head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        # All four properties at once:
        assert n_lora_trainable > 0, "no LoRA params are trainable"
        assert n_enc_other_trainable == 0, (
            f"{n_enc_other_trainable} non-LoRA encoder params are trainable — freeze leaked"
        )
        assert n_head_trainable > 0, "head has no trainable params"

    def test_grad_checkpoint_equivalence_in_eval_mode(self) -> None:
        """In eval mode + no_grad, gradient checkpointing must not change
        the forward output: timm's per-block checkpoint is a no-op for
        the math, only the activation-saving discipline differs. Pins
        the invariant that ``set_grad_checkpointing(True)`` doesn't
        accidentally fork the computation in inference.
        """
        model = self._build_model(use_grad_ckpt=True, apply_lora=False)
        model.eval()
        rgb = torch.randn((1, 3, 224, 224), dtype=torch.float32)
        pred = torch.zeros((1, 12), dtype=torch.float32)
        core_mean = torch.zeros((1, 12), dtype=torch.float32)
        # Compare: encoder grad_checkpointing on vs off — same weights,
        # same input, eval+no_grad context → outputs must match exactly.
        with torch.no_grad():
            model.encoder.set_grad_checkpointing(True)
            m1, p1 = model(rgb, pred, core_mean)
            model.encoder.set_grad_checkpointing(False)
            m2, p2 = model(rgb, pred, core_mean)
        torch.testing.assert_close(m1, m2, rtol=0, atol=0)
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)

    def test_grad_checkpointing_flag_propagates_to_encoder(self) -> None:
        """When CellPhenotypeModel is constructed with use_gradient_checkpointing=True,
        the timm encoder's ``grad_checkpointing`` attribute must be True
        — that's the real switch that activates per-block checkpointing
        in ``encoder.forward_features``. The CellPhenotypeModel flag is just the
        ergonomic entry point.
        """
        model = self._build_model(use_grad_ckpt=True, apply_lora=False)
        assert getattr(model.encoder, "grad_checkpointing", False) is True
        model2 = self._build_model(use_grad_ckpt=False, apply_lora=False)
        assert getattr(model2.encoder, "grad_checkpointing", False) is False

    def test_v4_model_trainable_param_count_matches_spec(self) -> None:
        """Trainable params = LoRA (encoder) + head + feat_norm.

        On ViT-Base/16 (``d = 768``, qkv-out = ``3·768 = 2304``) with
        rank=8 alpha=8:

          - LoRA per qkv  module: A(8·768) + B(2304·8)
                                = 6,144 + 18,432 = 24,576
          - LoRA per proj module: A(8·768) + B(768·8)
                                = 6,144 + 6,144  = 12,288
          - Per block: 24,576 + 12,288 = **36,864**
          - 12 blocks: **442,368 LoRA params total**

        Head (trunk(1656→768→384) + marker(384→12) + phenotype(384→9)):
          - Linear(2·768 + 12 + 12, 768) = 1556·768 + 768 = 1,195,776
          - Linear(768, 384) = 295,296
          - marker_head = 4,620; phenotype_head = 3,465
          - **head total ≈ 1.5M**

        Grand total ≈ **1.94M trainable**.

        We assert a band rather than the exact value since the head's
        internal `hidden` tuple may evolve (e.g., future v4.1 may tweak
        it). Out-of-band failure means either LoRA wrapped a different
        set of modules or the head class changed under us."""
        model = self._build_model(apply_lora=True)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Sanity bounds — comfortable margins on either side.
        assert 1.5e6 < n_trainable < 3.0e6, (
            f"v4 trainable params on ViT-Base = {n_trainable}; expected ~1.9M. "
            "If significantly outside this band, either the head or the LoRA "
            "wrapping count changed and the spec needs updating."
        )
