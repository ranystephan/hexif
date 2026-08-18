"""Unit tests for :func:`hexif.cell_phenotype.asymmetric_loss_with_logits`.

Paper: Ridnik et al. 2021, "Asymmetric Loss for Multi-Label Classification"
(arXiv 2009.14119, ICCV 2021). Reference implementation:
https://github.com/Alibaba-MIIL/ASL.

Tests target the load-bearing mathematical properties — these are what
differentiate ASL from a vanilla BCE variant and what we need to be
sure of before switching the v6 marker loss:

1. **BCE limit.** With ``γ⁺ = γ⁻ = 0, clip = 0`` the loss must reduce
   exactly to ``F.binary_cross_entropy_with_logits`` (mean reduction).
   This is the "do no harm" sanity check.
2. **Probability shift suppresses easy negatives to zero.** For a true
   negative (y=0) where the model predicts ``p < clip``, the
   contribution to the per-element loss must be **exactly zero** — this
   is the headline trick that differentiates ASL from focal-BCE.
3. **Asymmetric focusing direction.** Increasing ``γ⁻`` must monotonically
   reduce the loss contribution of confidently-classified negatives;
   increasing ``γ⁺`` does NOT reduce positives the same way unless
   they're confidently correct.
4. **Focusing weights are detached.** Gradient flows through the log
   terms but not through the focusing weights — verified by checking
   ``∂L/∂logits`` against a hand-computed expression that treats the
   focusing weights as constants.
5. **Numerical stability under AMP.** Loss is finite for extreme logits
   (±50) in both fp32 and fp16.
6. **Ignore-mask semantics.** Matches ``focal_bce_with_logits``: True =
   ignore, masked entries don't contribute.
7. **Input validation.** Bad γ / clip values raise ValueError.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hexif.cell_phenotype import (
    asymmetric_loss_with_logits,
    focal_bce_with_logits,
)

# --------------------------------------------------------------------------- #
# 1. BCE limit
# --------------------------------------------------------------------------- #


class TestBCELimit:
    def test_zero_gamma_zero_clip_equals_bce(self) -> None:
        """γ⁺=γ⁻=0, clip=0 → exact BCE mean."""
        torch.manual_seed(0)
        logits = torch.randn(8, 12)
        target = torch.randint(0, 2, (8, 12)).float()
        asl = asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=0.0, clip=0.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        torch.testing.assert_close(asl, bce, rtol=1e-6, atol=1e-7)

    def test_zero_gamma_zero_clip_with_ignore_mask_equals_masked_bce(self) -> None:
        """Same identity but with an ignore mask."""
        torch.manual_seed(0)
        logits = torch.randn(8, 12)
        target = torch.randint(0, 2, (8, 12)).float()
        mask = torch.rand(8, 12) > 0.5
        asl = asymmetric_loss_with_logits(
            logits, target, gamma_pos=0.0, gamma_neg=0.0, clip=0.0, ignore_mask=mask
        )
        bce_unreduced = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        keep = (~mask).float()
        expected = (bce_unreduced * keep).sum() / keep.sum().clamp_min(1.0)
        torch.testing.assert_close(asl, expected, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------- #
# 2. Probability shift suppresses easy negatives to zero
# --------------------------------------------------------------------------- #


class TestProbabilityShift:
    def test_easy_negative_loss_is_exactly_zero(self) -> None:
        """For y=0 and p < clip, the per-element loss must be EXACTLY 0.

        Reason: ``p_m = max(p - clip, 0) = 0`` and ``log(1 - 0) = 0``,
        and the focusing weight is also 0. Both factors vanish, so the
        contribution is identically zero, not just small — this is what
        the paper means by "hard-thresholding easy negatives."
        """
        # logits very negative → p ≈ 0 << clip = 0.05.
        logits = torch.tensor([[-10.0]])
        target = torch.tensor([[0.0]])
        loss = asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        assert loss.item() == 0.0, (
            f"easy negative should have exactly zero loss; got {loss.item():.3e}"
        )

    def test_zero_clip_does_not_suppress_easy_negatives(self) -> None:
        """With clip=0, the easy-negative still has a tiny but nonzero loss
        (this is the focal-BCE regime — ASL's `m` is the unique difference)."""
        logits = torch.tensor([[-10.0]])
        target = torch.tensor([[0.0]])
        loss = asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.0)
        # p^4 * log(1-p) with p = sigmoid(-10) ≈ 4.5e-5 → loss ≈ 4.1e-18, not 0.
        assert 0.0 < loss.item() < 1e-10

    def test_hard_negative_still_has_substantial_loss(self) -> None:
        """For y=0 and p ≈ 0.9 (hard negative — model is confidently wrong),
        the loss must still be large enough to drive gradient updates.
        Specifically, larger than the easy-negative loss by many orders."""
        easy = asymmetric_loss_with_logits(
            torch.tensor([[-10.0]]),
            torch.tensor([[0.0]]),
            gamma_pos=0.0,
            gamma_neg=4.0,
            clip=0.05,
        )
        hard = asymmetric_loss_with_logits(
            torch.tensor([[+2.2]]),
            torch.tensor([[0.0]]),  # σ(2.2) ≈ 0.9
            gamma_pos=0.0,
            gamma_neg=4.0,
            clip=0.05,
        )
        assert hard.item() > 0.5, f"hard negative loss should be substantial; got {hard.item():.3e}"
        assert hard.item() > 1e6 * (easy.item() + 1e-30), (
            "hard negative should massively exceed easy negative loss"
        )


# --------------------------------------------------------------------------- #
# 3. Asymmetric focusing direction
# --------------------------------------------------------------------------- #


class TestAsymmetricFocusing:
    def test_gamma_neg_monotonically_reduces_easy_negative_contribution(self) -> None:
        """For a moderately-easy negative (p ≈ 0.1, y=0, clip=0), the
        loss should monotonically decrease as γ⁻ increases — focal-style.
        We use clip=0 to isolate the γ⁻ effect from the probability shift."""
        logits = torch.tensor([[-2.2]])  # σ ≈ 0.1
        target = torch.tensor([[0.0]])
        losses = [
            asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=g, clip=0.0).item()
            for g in (0.0, 1.0, 2.0, 4.0, 8.0)
        ]
        # Monotone non-increasing.
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1] - 1e-9, (
                f"γ⁻ ablation not monotone non-increasing: {losses}"
            )
        # And not just constant (real focusing happens).
        assert losses[0] > losses[-1] + 1e-6, f"γ⁻=8 should reduce loss below γ⁻=0; got {losses}"

    def test_gamma_pos_zero_preserves_positive_gradient(self) -> None:
        """With γ⁺=0, the positive loss term is just `-log(p)` — the
        full BCE positive contribution. This is the paper's headline
        choice: γ⁺=0 keeps maximum gradient on rare positives."""
        # Hard positive: model predicts p ≈ 0.1, true label 1.
        logits = torch.tensor([[-2.2]])
        target = torch.tensor([[1.0]])
        asl = asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        # Equals -log(sigmoid(-2.2)) (no focusing on positives).
        expected = -F.logsigmoid(logits).mean()
        torch.testing.assert_close(asl, expected, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------- #
# 4. Focusing weights are detached (gradient sanity)
# --------------------------------------------------------------------------- #


class TestGradientFlow:
    def test_grad_does_not_flow_through_focusing_weights(self) -> None:
        """If focusing weights were NOT detached, the gradient w.r.t. logits
        would have extra terms from d(w_neg)/d(logits) and d(w_pos)/d(logits).
        With detach, the gradient is simply ``w_neg · d(log(1-p_m))/d(logits)``
        for negatives and ``w_pos · d(log(p))/d(logits)`` for positives.

        We verify by computing the analytic gradient under the
        focusing-weights-detached convention and comparing to autograd.
        """
        torch.manual_seed(0)
        logits = torch.tensor([[1.5, -0.7, 0.0]], requires_grad=True)
        target = torch.tensor([[1.0, 0.0, 1.0]])
        gamma_pos, gamma_neg, clip = 0.0, 4.0, 0.05

        loss = asymmetric_loss_with_logits(
            logits, target, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip
        )
        loss.backward()
        autograd_grad = logits.grad.detach().clone()

        # Analytic gradient with detached focusing weights:
        # For y=1: ∂L/∂logit = - w_pos · ∂log(p)/∂logit = - w_pos · (1 - p).
        # For y=0: ∂L/∂logit = - w_neg · ∂log(1 - p_m)/∂logit.
        #   With p_m = p - clip (when p > clip), 1-p_m = 1-p+clip,
        #   so ∂log(1-p_m)/∂logit = -p(1-p) / (1-p+clip).
        #   Therefore: ∂L/∂logit = w_neg · p(1-p) / (1-p+clip).
        # All divided by the number of elements (mean reduction).
        n = logits.numel()
        p = torch.sigmoid(logits.detach())
        p_m = (p - clip).clamp_min(0.0)
        w_pos = (1 - p).pow(gamma_pos) if gamma_pos > 0 else torch.ones_like(p)
        w_neg = p_m.pow(gamma_neg) if gamma_neg > 0 else torch.ones_like(p_m)

        # Build expected gradient piecewise.
        grad_pos = -w_pos * (1 - p)  # positives
        # Negatives: case-split on whether p > clip.
        denom = 1 - p + clip
        # When p <= clip, p_m = 0 → log(1-p_m) = 0, focusing weight = 0; gradient is 0.
        grad_neg = torch.where(p > clip, w_neg * p * (1 - p) / denom, torch.zeros_like(p))
        expected = (target * grad_pos + (1 - target) * grad_neg) / n

        torch.testing.assert_close(autograd_grad, expected, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- #
# 5. Numerical stability under extreme logits
# --------------------------------------------------------------------------- #


class TestNumericalStability:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_finite_on_extreme_logits(self, dtype: torch.dtype) -> None:
        """Logits at ±50 should still produce finite, non-NaN losses
        in both fp32 and fp16 (AMP regime). Catches a bug class where
        log(0) or 0^0 sneak in for very confident predictions."""
        logits = torch.tensor([[-50.0, 50.0, -50.0, 50.0]], dtype=dtype)
        target = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=dtype)
        loss = asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        assert torch.isfinite(loss), f"non-finite loss at extreme logits, dtype={dtype}"
        # Correctly classified extremes (entries 0 and 1) should each
        # contribute approximately 0; confidently-wrong extremes
        # (entries 2 and 3) should contribute substantial loss.
        assert loss.item() > 1.0


# --------------------------------------------------------------------------- #
# 6. Ignore mask semantics
# --------------------------------------------------------------------------- #


class TestIgnoreMask:
    def test_ignore_mask_excludes_those_entries_from_mean(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(4, 5)
        target = torch.randint(0, 2, (4, 5)).float()
        mask = torch.zeros(4, 5, dtype=torch.bool)
        mask[:, :2] = True  # ignore first 2 columns
        full = asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        masked = asymmetric_loss_with_logits(
            logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, ignore_mask=mask
        )
        # Manually compute the masked version: only columns 2..4 contribute.
        partial = asymmetric_loss_with_logits(
            logits[:, 2:], target[:, 2:], gamma_pos=0.0, gamma_neg=4.0, clip=0.05
        )
        torch.testing.assert_close(masked, partial, rtol=1e-6, atol=1e-7)
        # Full ≠ masked unless the masked entries happened to be exactly 0.
        assert (full - masked).abs().item() > 0


# --------------------------------------------------------------------------- #
# 7. Input validation
# --------------------------------------------------------------------------- #


class TestInputValidation:
    def test_negative_gamma_raises(self) -> None:
        logits = torch.zeros(1, 1)
        target = torch.zeros(1, 1)
        with pytest.raises(ValueError, match="γ"):
            asymmetric_loss_with_logits(logits, target, gamma_pos=-0.5, gamma_neg=4.0)
        with pytest.raises(ValueError, match="γ"):
            asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=-1.0)

    def test_bad_clip_raises(self) -> None:
        logits = torch.zeros(1, 1)
        target = torch.zeros(1, 1)
        with pytest.raises(ValueError, match="clip"):
            asymmetric_loss_with_logits(logits, target, clip=-0.1)
        with pytest.raises(ValueError, match="clip"):
            asymmetric_loss_with_logits(logits, target, clip=1.0)


# --------------------------------------------------------------------------- #
# 8. Real-data-ish sanity: ASL pushes harder on rare positives than focal-BCE
# --------------------------------------------------------------------------- #


def test_asl_assigns_more_relative_weight_to_rare_positives_than_focal() -> None:
    """For a synthetic mini-batch with a single rare positive among many
    easy negatives, the rare positive should contribute a *larger fraction*
    of total ASL loss than of total focal-BCE loss.

    Intuition: ASL zeros the easy negatives via the clip shift; focal-BCE
    only down-weights them. So the rare positive's share of total loss
    grows under ASL.

    This test is the "this is why we're switching" sanity check.
    """
    # 100 cells, 1 marker. One positive (cell 0) hard to predict (p ≈ 0.3);
    # 99 negatives all easy (p ≈ 0.02).
    logits = torch.full((100, 1), -3.9)  # σ ≈ 0.02
    logits[0, 0] = -0.85  # σ ≈ 0.30 — hard positive
    target = torch.zeros(100, 1)
    target[0, 0] = 1.0

    alpha = torch.tensor([0.5])  # neutral alpha for focal

    focal = focal_bce_with_logits(logits, target, alpha=alpha, gamma=2.0)
    asl = asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)

    # Per-element loss values for the positive (entry 0) under each loss.
    # Focal: -alpha · (1 - 0.30)^2 · log(0.30) ≈ 0.5 · 0.49 · 1.20 = 0.296.
    # ASL  : -1.0 · log(0.30) ≈ 1.20. ASL puts 4× more weight on this entry.
    # Mean-reduction across 100 entries: focal mean is dragged up by 99
    # easy-negative contributions (~0.02^2 · log(0.98) each ≈ 4e-6); ASL
    # mean is NOT — those entries are zeroed by the clip shift.

    # The asserted property: ASL's per-batch mean loss > focal's per-batch
    # mean loss in this setup (the rare positive is the dominant term).
    assert asl.item() > focal.item(), (
        f"ASL mean ({asl.item():.4f}) should exceed focal mean "
        f"({focal.item():.4f}) when rare positive dominates an all-easy-neg batch"
    )

    # And the ASL loss should equal -log(σ(-0.85)) / 100 to within float,
    # since the 99 easy-neg entries contribute exactly zero (clip).
    expected_asl = -F.logsigmoid(torch.tensor(-0.85)) / 100
    torch.testing.assert_close(asl, expected_asl, rtol=1e-5, atol=1e-7)
