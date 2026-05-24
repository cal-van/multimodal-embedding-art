"""
Tests for ``PatchAlignmentLoss`` (M2c).

Uses an in-test fake encoder that returns synthetic per-layer patch tokens so
the loss can be exercised without loading real LanguageBind weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from embedding_art.core.patch_alignment import PatchAlignmentLoss
from embedding_art.encoders.features import LayerFeatures

# ---------------------------------------------------------------------------
# Fake encoder for unit tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeViTEncoder:
    """A toy encoder that returns per-image patch tokens at two layers.

    The first token is treated as a CLS token (per ViT convention). Token
    values are deterministic functions of the input image so the loss has a
    well-defined gradient through the encoder.
    """

    n_patches: int = 16
    embed_dim: int = 32
    has_layer_features: bool = True

    def get_layer_features(self, tensor: torch.Tensor) -> dict[int, LayerFeatures]:
        if not self.has_layer_features:
            raise AttributeError("disabled for test")
        # tensor: [B, 3, H, W]; flatten to embed
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        b = tensor.shape[0]
        feat = tensor.flatten(start_dim=1)
        # Project to (n_patches + 1) * embed_dim via a fixed-but-image-dependent map.
        proj_dim = (self.n_patches + 1) * self.embed_dim
        # Use the *first* proj_dim elements; pad/repeat as needed.
        n = feat.shape[1]
        if n < proj_dim:
            feat = feat.repeat(1, (proj_dim // n) + 1)
        proj = feat[:, :proj_dim].reshape(b, self.n_patches + 1, self.embed_dim)

        return {
            0: LayerFeatures(
                tensor=proj,
                spatial=True,
                shape_semantic="batch_tokens_dim",
                layer_name="layer_0",
            ),
            4: LayerFeatures(
                tensor=proj * 0.5,
                spatial=True,
                shape_semantic="batch_tokens_dim",
                layer_name="layer_4",
            ),
        }


class _EncoderWithoutPatchTokens:
    """Encoder without ``get_layer_features`` — patch loss should be disabled."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPatchAlignmentLossInactive:
    """When the encoder lacks ``get_layer_features``, the loss disables itself."""

    def test_encoder_without_patch_tokens_disables_loss(self) -> None:
        loss = PatchAlignmentLoss()
        encoder = _EncoderWithoutPatchTokens()
        target = torch.rand(1, 3, 32, 32)
        loss.calibrate(target, encoder)
        assert loss.is_active is False

    def test_inactive_loss_returns_zero(self) -> None:
        loss = PatchAlignmentLoss()
        encoder = _EncoderWithoutPatchTokens()
        target = torch.rand(1, 3, 32, 32)
        loss.calibrate(target, encoder)
        current = torch.rand(1, 3, 32, 32)
        value = loss(current, encoder)
        assert value.item() == 0.0


class TestPatchAlignmentLossActive:
    """When the encoder exposes patch tokens, the loss has expected properties."""

    def test_calibration_populates_reference_tokens(self) -> None:
        loss = PatchAlignmentLoss()
        encoder = _FakeViTEncoder()
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        assert loss.is_active is True
        assert set(loss.reference_tokens.keys()) == {0, 4}

    def test_loss_zero_when_current_equals_target(self) -> None:
        loss = PatchAlignmentLoss()
        encoder = _FakeViTEncoder()
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        value = loss(target, encoder)
        # When current image == target image, patch tokens are identical
        # and cosine similarity = 1, so loss = 0.
        assert value.item() == 0.0 or value.item() < 1e-5

    def test_loss_nonzero_for_different_image(self) -> None:
        loss = PatchAlignmentLoss()
        encoder = _FakeViTEncoder()
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        # Use a contrasting random image
        torch.manual_seed(0)
        current = torch.rand(1, 3, 16, 16)
        value = loss(current, encoder)
        assert value.item() > 1e-3

    def test_loss_is_differentiable(self) -> None:
        loss = PatchAlignmentLoss()
        encoder = _FakeViTEncoder()
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        current = torch.rand(1, 3, 16, 16, requires_grad=True)
        value = loss(current, encoder)
        value.backward()
        assert current.grad is not None
        assert current.grad.shape == current.shape

    def test_drop_cls_token_default(self) -> None:
        """By default the first token (CLS) is dropped from the per-token sum."""
        loss = PatchAlignmentLoss()
        encoder = _FakeViTEncoder(n_patches=4, embed_dim=8)
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        ref = loss.reference_tokens[0]
        # n_patches=4 + 1 CLS = 5 total tokens, drop CLS = 4 patches.
        assert ref.shape == (1, 4, 8)

    def test_drop_cls_token_disabled(self) -> None:
        loss = PatchAlignmentLoss(drop_cls_token=False)
        encoder = _FakeViTEncoder(n_patches=4, embed_dim=8)
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        ref = loss.reference_tokens[0]
        # n_patches=4 + 1 CLS = 5 total tokens, drop_cls_token=False keeps all 5.
        assert ref.shape == (1, 5, 8)


class TestPatchAlignmentLossLayerSubset:
    """``layer_subset`` controls which layers participate."""

    def test_layer_subset_restricts_calibration(self) -> None:
        loss = PatchAlignmentLoss(layer_subset=[4])
        encoder = _FakeViTEncoder()
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        assert set(loss.reference_tokens.keys()) == {4}

    def test_empty_layer_subset_disables_loss(self) -> None:
        loss = PatchAlignmentLoss(layer_subset=[99])
        encoder = _FakeViTEncoder()
        target = torch.rand(1, 3, 16, 16)
        loss.calibrate(target, encoder)
        assert loss.is_active is False
