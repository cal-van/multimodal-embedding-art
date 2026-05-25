"""
Tests for the diffusion-prior losses (M2b): SDS + VSD.

Uses small synthetic score functions so the tests run instantly without
loading a real diffusion model.
"""

from __future__ import annotations

import torch

from embedding_art.diffusion_priors import (
    DualTrackConfig,
    SDSLoss,
    VSDLoss,
    dual_track_loss,
    make_dual_track_tracks,
)


def _alpha(t: torch.Tensor) -> torch.Tensor:
    """Toy schedule α_t = √(1 - t²). Reshapes to broadcast over [B, C, H, W]."""
    return torch.sqrt(1 - t.pow(2)).view(-1, 1, 1, 1)


def _sigma(t: torch.Tensor) -> torch.Tensor:
    """Toy schedule σ_t = t. Reshapes to broadcast over [B, C, H, W]."""
    return t.view(-1, 1, 1, 1)


def _weight(t: torch.Tensor) -> torch.Tensor:
    """Per-step weighting. Reshapes to broadcast over [B, C, H, W]."""
    return torch.ones_like(t).view(-1, 1, 1, 1)


def _identity_score(x_t: torch.Tensor, t: torch.Tensor, conditioning) -> torch.Tensor:
    """Score function that just returns x_t. Lets us verify gradient flow."""
    return x_t


class TestSDSLoss:
    """SDSLoss basics."""

    def test_returns_scalar(self) -> None:
        loss = SDSLoss(score_fn=_identity_score, sigma_schedule=_sigma, alpha_schedule=_alpha)
        loss.seed(42)
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        value = loss(x, conditioning=None, timestep=0.5)
        assert value.dim() == 0  # scalar

    def test_gradient_flows_to_x(self) -> None:
        loss = SDSLoss(score_fn=_identity_score, sigma_schedule=_sigma, alpha_schedule=_alpha)
        loss.seed(42)
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        value = loss(x, conditioning=None, timestep=0.5)
        value.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.shape == x.shape

    def test_raises_when_x_not_requires_grad(self) -> None:
        loss = SDSLoss(score_fn=_identity_score, sigma_schedule=_sigma, alpha_schedule=_alpha)
        x = torch.randn(2, 3, 4, 4, requires_grad=False)
        try:
            loss(x, conditioning=None)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_deterministic_with_seed(self) -> None:
        loss1 = SDSLoss(score_fn=_identity_score, sigma_schedule=_sigma, alpha_schedule=_alpha)
        loss1.seed(42)
        loss2 = SDSLoss(score_fn=_identity_score, sigma_schedule=_sigma, alpha_schedule=_alpha)
        loss2.seed(42)
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        v1 = loss1(x, conditioning=None)
        x.grad = None
        v2 = loss2(x, conditioning=None)
        assert torch.allclose(v1, v2)


class TestVSDLoss:
    """VSDLoss basics."""

    def test_returns_scalar(self) -> None:
        loss = VSDLoss(
            teacher_score_fn=_identity_score,
            student_score_fn=_identity_score,
            sigma_schedule=_sigma,
            alpha_schedule=_alpha,
        )
        loss.seed(42)
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        value = loss(x, conditioning=None, timestep=0.5)
        assert value.dim() == 0

    def test_zero_when_teacher_and_student_agree(self) -> None:
        """If teacher == student, the VSD gradient is zero — that's the point."""
        loss = VSDLoss(
            teacher_score_fn=_identity_score,
            student_score_fn=_identity_score,
            sigma_schedule=_sigma,
            alpha_schedule=_alpha,
        )
        loss.seed(7)
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        value = loss(x, conditioning=None, timestep=0.5)
        value.backward()
        # Gradient should be ~zero everywhere.
        assert torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_nonzero_when_teacher_and_student_disagree(self) -> None:
        """When student returns a different prediction, the gradient is nonzero."""

        def teacher(x_t, t, c):
            return x_t

        def student(x_t, t, c):
            return -x_t  # very different

        loss = VSDLoss(
            teacher_score_fn=teacher,
            student_score_fn=student,
            sigma_schedule=_sigma,
            alpha_schedule=_alpha,
        )
        loss.seed(7)
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        value = loss(x, conditioning=None, timestep=0.5)
        value.backward()
        assert (x.grad.abs() > 0).any()

    def test_student_loss_is_real_mse(self) -> None:
        """student_loss is an MSE that backprops into the student's params."""
        # Make a trainable student (a tiny linear layer) so we can verify
        # that its loss has gradient.
        student = torch.nn.Linear(48, 48, bias=False)

        def student_fn(x_t, t, c):
            flat = x_t.flatten(start_dim=1)
            return student(flat).reshape_as(x_t)

        loss = VSDLoss(
            teacher_score_fn=_identity_score,
            student_score_fn=student_fn,
            sigma_schedule=_sigma,
            alpha_schedule=_alpha,
        )
        loss.seed(11)
        x = torch.randn(2, 3, 4, 4)
        student_l = loss.student_loss(x, conditioning=None, timestep=0.5)
        student_l.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


class TestDualTrackConfig:
    """Validation rules for DualTrackConfig."""

    def test_default_is_pure_honest(self) -> None:
        config = DualTrackConfig()
        assert config.alpha == 1.0
        config.validate()

    def test_alpha_out_of_range_rejected(self) -> None:
        for bad in (-0.1, 1.5, 2.0):
            try:
                DualTrackConfig(alpha=bad).validate()
                raise AssertionError(f"Expected ValueError for alpha={bad}")
            except ValueError:
                pass

    def test_negative_weights_rejected(self) -> None:
        try:
            DualTrackConfig(honest_weight=-1.0).validate()
            raise AssertionError("Expected ValueError for honest_weight=-1.0")
        except ValueError:
            pass


class TestDualTrackLoss:
    """Loss blending semantics."""

    def test_pure_honest_skips_natural(self) -> None:
        honest_called = {"count": 0}
        natural_called = {"count": 0}

        def honest_fn() -> torch.Tensor:
            honest_called["count"] += 1
            return torch.tensor(1.0)

        def natural_fn() -> torch.Tensor:
            natural_called["count"] += 1
            return torch.tensor(10.0)

        value = dual_track_loss(
            honest_loss_fn=honest_fn,
            natural_loss_fn=natural_fn,
            config=DualTrackConfig(alpha=1.0),
        )
        assert value.item() == 1.0
        assert honest_called["count"] == 1
        assert natural_called["count"] == 0

    def test_pure_natural(self) -> None:
        value = dual_track_loss(
            honest_loss_fn=lambda: torch.tensor(1.0),
            natural_loss_fn=lambda: torch.tensor(10.0),
            config=DualTrackConfig(alpha=0.0),
        )
        assert value.item() == 10.0

    def test_balanced_blend(self) -> None:
        value = dual_track_loss(
            honest_loss_fn=lambda: torch.tensor(2.0),
            natural_loss_fn=lambda: torch.tensor(10.0),
            config=DualTrackConfig(alpha=0.5),
        )
        assert value.item() == 6.0

    def test_honest_weight_applied_before_blending(self) -> None:
        value = dual_track_loss(
            honest_loss_fn=lambda: torch.tensor(2.0),
            natural_loss_fn=lambda: torch.tensor(10.0),
            config=DualTrackConfig(alpha=1.0, honest_weight=3.0),
        )
        assert value.item() == 6.0

    def test_natural_loss_none_falls_back_to_honest_only(self) -> None:
        value = dual_track_loss(
            honest_loss_fn=lambda: torch.tensor(4.0),
            natural_loss_fn=None,
            config=DualTrackConfig(alpha=0.3),
        )
        assert abs(value.item() - 0.3 * 4.0) < 1e-5


class TestMakeDualTrackTracks:
    def test_returns_two_tracks(self) -> None:
        tracks = make_dual_track_tracks(alpha_honest=1.0, alpha_natural=0.0)
        assert len(tracks) == 2
        assert tracks[0].alpha == 1.0
        assert tracks[1].alpha == 0.0
