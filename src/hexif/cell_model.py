"""Pathology-encoder model used by the HEXIF cell-phenotyping runs."""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with a low-rank LoRA adapter.

    Forward computes ``F.linear(x, W, b) + (alpha / rank) * B(A(x))`` where
    ``A: in_features -> rank`` and ``B: rank -> out_features`` are bias-less
    linear maps. ``A`` is kaiming-uniform initialised (the nn.Linear default,
    a=sqrt(5)); ``B`` is initialised to zero so the LoRA path contributes
    nothing at the start of training. The base ``weight`` / ``bias`` of the
    wrapped Linear are exposed via ``self.weight`` / ``self.bias`` so callers
    can freeze them externally.

    The submodule names ``A`` and ``B`` are part of the checkpoint
    contract and are used to select trainable adapter parameters.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        in_features = int(base.in_features)
        out_features = int(base.out_features)
        # Keep the original Linear's parameters as our own so the wrapper is a
        # drop-in replacement for the base Linear. They live on this module
        # under the names ``weight`` and ``bias`` -- identical to nn.Linear --
        # which keeps every existing state_dict key stable and lets the
        # caller freeze them with ``param.requires_grad = False``.
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.weight = nn.Parameter(base.weight.detach().clone())
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        # LoRA submodules. Keeping them as nn.Linear sub-modules means their
        # parameters appear in state_dict at "<wrapper>.A.weight" and
        # "<wrapper>.B.weight", matching the v1.1 checkpoint layout.
        self.A = nn.Linear(in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, out_features, bias=False)
        # Match nn.Linear's default A init (kaiming_uniform_ with a=sqrt(5))
        # so the LoRA path starts from a reasonable distribution.
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        # Zero B so the adapter starts as a no-op (LoRA paper convention).
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(x, self.A.weight), self.B.weight)
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.rank}, alpha={self.alpha}"
        )

    def __repr__(self) -> str:
        # Use the default nn.Module __repr__ but flatten it to a single line so
        # very deep nesting does not blow the stack during pretty-printing.
        # nn.Module.__repr__ is already recursion-safe in PyTorch, but we keep
        # this explicit override defensive against future custom traversal.
        return f"_LoRALinear({self.extra_repr()})"


def apply_lora_to_vit(encoder: nn.Module, rank: int, alpha: float) -> int:
    """Inject LoRA adapters into every attention ``qkv`` / ``proj`` Linear.

    Walks ``encoder.named_modules()`` and replaces every module whose name
    ends in ``.attn.qkv`` or ``.attn.proj`` (the timm Swin and ViT attention
    naming convention) with a :class:`_LoRALinear` wrapper of equivalent
    base weight + bias. Returns the integer count of wrapped modules.

    The wrapper's submodules are named ``A`` and ``B``, producing stable
    state-dict keys such as ``<...>.attn.qkv.A.weight``.
    """
    # Collect targets first so we don't mutate the module tree while iterating.
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for module_name, module in encoder.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{module_name}.{child_name}" if module_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if full.endswith(".attn.qkv") or full.endswith(".attn.proj"):
                targets.append((module, child_name, child))

    n_wrapped = 0
    for parent, attr_name, base_linear in targets:
        wrapped = _LoRALinear(base_linear, rank=rank, alpha=alpha)
        # Preserve the dtype / device of the original Linear.
        wrapped.to(dtype=base_linear.weight.dtype, device=base_linear.weight.device)
        setattr(parent, attr_name, wrapped)
        n_wrapped += 1
    return n_wrapped


ENCODER_CHOICES: tuple[str, ...] = ("uni2", "h_optimus_0", "vit_base_smoke")
ENCODER_REVISIONS: dict[str, str | None] = {
    "uni2": "d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
    "h_optimus_0": "b145cc1e6c6b30d3251aa8b1f844e6974188a743",
    "vit_base_smoke": None,
}
_ENCODER_HF_HUB: dict[str, str] = {
    "uni2": f"hf-hub:MahmoodLab/UNI2-h@{ENCODER_REVISIONS['uni2']}",
    "h_optimus_0": f"hf-hub:bioptimus/H-optimus-0@{ENCODER_REVISIONS['h_optimus_0']}",
    # Open ViT-Base for smoke + tests. Not a substitute for UNI2 — exists
    # only so the v4 pipeline can be exercised end-to-end without gated
    # HuggingFace access.
    "vit_base_smoke": "vit_base_patch16_224",
}


def _uni2_timm_kwargs() -> dict[str, object]:
    """Return the canonical ``timm.create_model`` kwargs for UNI2.

    Sourced verbatim from UNI2's HuggingFace model card. Without these
    kwargs ``timm.create_model`` falls back to the stock
    ``vit_giant_patch14_224`` config (depth=40, embed_dim=1536, no
    register tokens), which does not match UNI2's saved checkpoint
    (depth=24, SwiGLU MLP, 8 register tokens, no class-embed in pos_embed).
    The mismatch surfaces as a ``RuntimeError`` in
    ``resample_abs_pos_embed`` at load time.
    """
    from timm.layers import SwiGLUPacked

    return {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


def _encoder_timm_kwargs(name: str) -> dict[str, object]:
    """Return the per-encoder ``timm.create_model`` kwargs.

    See each model's HuggingFace README for provenance:
        UNI2: https://huggingface.co/MahmoodLab/UNI2-h
        H-optimus-0: https://huggingface.co/bioptimus/H-optimus-0

    For ``vit_base_smoke`` (the open substitute used in tests + smoke
    runs) we only need ``num_classes=0`` plus a defensive ``img_size=224``
    (which is also the timm default for vit_base_patch16_224, so the
    kwarg is a no-op in practice).
    """
    if name == "uni2":
        return _uni2_timm_kwargs()
    if name == "h_optimus_0":
        # Per Bioptimus's README — only ``init_values`` and an explicit
        # ``dynamic_img_size=False`` are required; the rest of the arch
        # (depth=40, embed_dim=1536, no reg_tokens) is timm's default
        # vit_giant_patch14_224.
        return {
            "num_classes": 0,
            "init_values": 1e-5,
            "dynamic_img_size": False,
            "img_size": 224,
        }
    if name == "vit_base_smoke":
        return {"num_classes": 0, "img_size": 224}
    raise ValueError(f"unknown v4 encoder {name!r}; expected one of {sorted(ENCODER_CHOICES)}")


def build_encoder(name: str, *, pretrained: bool = True) -> nn.Module:
    """Construct a timm ViT-style encoder for v4 training.

    Args:
        name: one of ``ENCODER_CHOICES``.

            * ``'uni2'`` — Mahmood lab's UNI2 ViT-Huge/14 (683M params).
              Pretrained on 200M H&E + IHC tiles via DINOv2 objective.
              Weights gated on HuggingFace; requires acceptance of the
              clinical-research agreement.
            * ``'h_optimus_0'`` — Bioptimus H-optimus-0 ViT-Giant/14 (~1.1B
              params). MIPHEI-ViT's reference encoder. Weights gated on
              HuggingFace; non-commercial license.
            * ``'vit_base_smoke'`` — open ViT-Base/16 (86M). For tests +
              smoke runs only; not used for production training.

        pretrained: when True, load the foundation-model weights from
            HuggingFace Hub. When False, the encoder is randomly
            initialized (useful for forward-shape tests that need no
            network access).

    Returns:
        A timm ViT module with a populated ``.embed_dim`` attribute. Gated
        production encoders are loaded at the immutable revisions in
        :data:`ENCODER_REVISIONS`. The
        module is in evaluation mode by default; the caller switches it
        to train mode and applies LoRA via :func:`apply_lora_to_vit`.

    Raises:
        ValueError: if ``name`` is not in :data:`ENCODER_CHOICES`.
        RuntimeError: from timm when the HF Hub returns 401/403 (gated
            access not granted) or the network is unreachable.
    """
    if name not in ENCODER_CHOICES:
        raise ValueError(f"unknown v4 encoder {name!r}; expected one of {sorted(ENCODER_CHOICES)}")
    hub_id = _ENCODER_HF_HUB[name]
    # Build the model with each encoder's canonical ``timm_kwargs`` (see
    # :func:`_encoder_timm_kwargs`). UNI2 in particular MUST be built
    # with its README's kwargs — without them timm falls back to the stock
    # ``vit_giant_patch14_224`` config (depth=40, no reg_tokens) and the
    # pos_embed shape mismatches the saved checkpoint at load time.
    kwargs = _encoder_timm_kwargs(name)
    enc = timm.create_model(hub_id, pretrained=pretrained, **kwargs)
    # timm exposes the patch / token width as `embed_dim` and the count of
    # prefix tokens (CLS + register tokens) as `num_prefix_tokens`. The
    # cell pool needs both: it splits the token sequence as
    # ``[prefix, patches]`` and reads the center patch by raster index.
    # For UNI2 ``num_prefix_tokens == 9`` (1 CLS + 8 register tokens);
    # for H-optimus-0 / vit_base_smoke it is 1 (CLS only).
    if not hasattr(enc, "embed_dim"):
        raise RuntimeError(
            f"timm model {hub_id!r} has no .embed_dim attribute — unexpected "
            "ViT layout, the cell pool code path would fail."
        )
    if not hasattr(enc, "num_prefix_tokens"):
        raise RuntimeError(
            f"timm model {hub_id!r} has no .num_prefix_tokens attribute — "
            "vit_cell_pool needs this to skip CLS + register tokens."
        )
    return enc


def vit_cell_pool(tokens: torch.Tensor, num_prefix_tokens: int = 1) -> torch.Tensor:
    """Pool a ViT token sequence into a ``(N, 2d)`` per-cell feature vector.

    Layout assumed (standard timm ViT ``forward_features`` output):
        ``tokens[:, 0]`` is the CLS token.
        ``tokens[:, 1:num_prefix_tokens]`` are register tokens (if any).
        ``tokens[:, num_prefix_tokens:]`` are patch tokens in raster
        order on a ``√n_patches × √n_patches`` grid (input image was square).

    For a 224×224 input with patch_size=16 (vit_base_smoke) we get a 14×14
    patch grid (= 196 patch tokens) and ``num_prefix_tokens = 1`` (CLS only).
    For UNI2 (patch_size=14, 8 register tokens at 224×224 input) we get
    16×16 = 256 patch tokens and ``num_prefix_tokens = 9`` (CLS + 8 reg).
    For H-optimus-0 (patch_size=14, no register tokens) we get 16×16 = 256
    patches and ``num_prefix_tokens = 1``.

    The cell-of-interest is at the *image center* by construction (the
    upstream dataset crops 224×224 patches centered on the cell centroid).
    The center patch token is therefore at spatial position
    ``(grid_size // 2, grid_size // 2)`` which in raster order is index
    ``(grid_size // 2) * grid_size + (grid_size // 2)`` *within the patch
    block* (i.e., after skipping the prefix tokens).

    Args:
        tokens: ``(N, num_prefix_tokens + n_patches, d)`` ViT token sequence
            as returned by ``model.forward_features``.
        num_prefix_tokens: count of non-patch tokens at the start of the
            sequence. Pass ``model.num_prefix_tokens`` from the timm
            encoder. Must be ``>= 1`` since the CLS token is at index 0
            and we read it as the global feature.

    Returns:
        A tensor of shape ``(N, 2 * d)`` formed by concatenating
        ``[cls_token, center_patch_token]``. ``d`` is read from the input
        tensor's last dimension.

    Raises:
        ValueError: if ``tokens.ndim != 3``, if ``num_prefix_tokens < 1``,
            or if the inferred ``n_patches`` is not a perfect square (which
            would mean the ViT was fed a non-square input — unsupported
            by this pool).
    """
    if tokens.ndim != 3:
        raise ValueError(
            f"vit_cell_pool expects (N, 1+n_patches, d); got shape {tuple(tokens.shape)}"
        )
    if num_prefix_tokens < 1:
        raise ValueError(
            f"vit_cell_pool: num_prefix_tokens must be >= 1 (CLS at index 0); "
            f"got {num_prefix_tokens}."
        )
    cls = tokens[:, 0]
    patches = tokens[:, num_prefix_tokens:]
    n_patches = patches.shape[1]
    grid = round(n_patches**0.5)
    if grid * grid != n_patches:
        raise ValueError(
            f"vit_cell_pool: patch count {n_patches} is not a perfect square; input was not square."
        )
    center = (grid // 2) * grid + (grid // 2)
    cell = patches[:, center]
    return torch.cat([cls, cell], dim=1)


class CellPhenotypeModel(nn.Module):
    """ViT pathology encoder, LoRA adapters, cell-token pool, and task heads.

    Encoder + pool are encoder-arch-aware (ViT pool ≠ Swin pool); everything
    Downstream processing uses batch normalization and parallel marker and
    phenotype heads. The training entry point selects the loss.

    Construction sequence used by ``train_cell_phenotype_v4.py``::

        encoder = build_encoder("uni2", pretrained=True)
        apply_lora_to_vit(encoder, rank=8, alpha=8.0)
        for n, p in encoder.named_parameters():
            p.requires_grad = (".A." in n) or (".B." in n)   # freeze base
        model = CellPhenotypeModel(
            encoder=encoder,
            embed_dim=encoder.embed_dim,
            n_pred=12,
            n_core_mean=12,
            head_hidden=(768, 384),
            dropout=0.1,
            use_gradient_checkpointing=True,
        )

    The head is imported lazily to keep model construction isolated.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        n_pred: int = 12,
        n_core_mean: int = 12,
        head_hidden: tuple[int, ...] = (768, 384),
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        # Imported lazily to avoid a top-level cycle:
        # Avoid importing the task head until model construction.
        from hexif.cell_phenotype import CellPhenotypeV1Head

        self.encoder = encoder
        self.embed_dim = int(embed_dim)
        # Read num_prefix_tokens from the encoder so the pool skips CLS +
        # register tokens correctly. UNI2 has 9 (1 CLS + 8 reg); H-optimus-0
        # and vit_base_smoke have 1 (CLS only). Falling back to 1 keeps
        # backwards compatibility for unit tests that pass a stub encoder.
        self.num_prefix_tokens = int(getattr(encoder, "num_prefix_tokens", 1))
        feat_dim = 2 * self.embed_dim
        in_dim = feat_dim + int(n_pred) + int(n_core_mean)
        # Standardize encoder features at runtime via running batchnorm
        # statistics — exactly v1.1 / v2's `feat_norm`. ``affine=False``
        # keeps the layer parameter-free (no learnable scale/shift) so the
        # downstream MLP is the only place gradients flow.
        self.feat_norm = nn.BatchNorm1d(feat_dim, affine=False)
        self.head = CellPhenotypeV1Head(
            in_dim=in_dim,
            hidden=head_hidden,
            dropout=dropout,
            n_markers=12,
            n_phenotypes=9,
        )
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        # Per-block gradient checkpointing via timm's built-in mechanism.
        # This is required for ViT-Giant / ViT-Huge to fit on a 12 GB GPU
        # at batch 32: wrapping the entire encoder in one outer
        # ``torch.utils.checkpoint.checkpoint`` saves activations only at
        # the encoder boundary, so the recompute-on-backward still has to
        # hold all 24 blocks' activations in memory and OOMs at the first
        # LoRA-augmented attention. ``set_grad_checkpointing(True)`` tells
        # timm to checkpoint each block individually — only one block's
        # activations live in memory at a time. Forward path is unchanged;
        # we just call ``encoder.forward_features(rgb)`` as normal.
        if self.use_gradient_checkpointing and hasattr(encoder, "set_grad_checkpointing"):
            encoder.set_grad_checkpointing(True)

    def _encoder_forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return its (N, 1+n_patches, d) token output."""
        return self.encoder.forward_features(rgb)

    def forward(
        self, rgb: torch.Tensor, pred: torch.Tensor, core_mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(marker_logits, phenotype_logits)``.

        Args:
            rgb: ``(N, 3, H, W)`` H&E patch batch, ImageNet-normalized.
                For UNI2 / H-optimus-0, H = W = 224.
            pred: ``(N, n_pred)`` cached marker predictions from the dense
                ORION model — used as side-channel fusion features.
            core_mean: ``(N, n_pred)`` per-core mean of those predictions.

        The encoder's per-block gradient checkpointing is configured in
        :meth:`__init__` via ``encoder.set_grad_checkpointing(True)`` when
        ``use_gradient_checkpointing`` is True. timm then wraps each
        attention block in ``torch.utils.checkpoint`` internally during
        ``forward_features``. This is much more memory-efficient than
        wrapping the entire encoder in one outer checkpoint (which would
        OOM in the backward pass for ViT-Huge / Giant at batch 32 on a
        12 GB GPU).
        """
        tokens = self._encoder_forward(rgb)
        feat = vit_cell_pool(tokens, self.num_prefix_tokens)  # (N, 2 * embed_dim)
        feat = self.feat_norm(feat)
        x = torch.cat([feat, pred, core_mean], dim=1)
        return self.head(x)
