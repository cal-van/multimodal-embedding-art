"""Tests for the linear-probe training pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embedding_art.evaluation.linear_probes import (
    LinearProbe,
    ProbeDataset,
    evaluate_probes,
    load_linear_probe,
    load_probe_manifest,
    save_linear_probe,
    save_probe_manifest,
    train_linear_probe,
)


def _make_separable_binary_dataset(
    *, n_pos: int = 60, n_neg: int = 60, dim: int = 32, seed: int = 0
) -> ProbeDataset:
    """Two linearly-separable Gaussians along the first axis."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n_pos, dim, generator=g) * 0.5
    pos[:, 0] += 2.0  # shift class 1 along axis 0
    neg = torch.randn(n_neg, dim, generator=g) * 0.5
    neg[:, 0] -= 2.0
    embeddings = torch.cat([pos, neg], dim=0)
    # Multi-label binary with a single "is_pos" column.
    labels = torch.cat([torch.ones(n_pos, 1), torch.zeros(n_neg, 1)], dim=0)
    return ProbeDataset(embeddings=embeddings, labels=labels, label_names=["is_pos"])


def _make_separable_multiclass_dataset(
    *, n_per_class: int = 40, dim: int = 32, num_classes: int = 4, seed: int = 0
) -> ProbeDataset:
    """One Gaussian per class along disjoint axes."""
    g = torch.Generator().manual_seed(seed)
    parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for c in range(num_classes):
        x = torch.randn(n_per_class, dim, generator=g) * 0.3
        x[:, c % dim] += 2.5
        parts.append(x)
        label_parts.append(torch.full((n_per_class,), c, dtype=torch.long))
    embeddings = torch.cat(parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    label_names = [f"c{c}" for c in range(num_classes)]
    return ProbeDataset(embeddings=embeddings, labels=labels, label_names=label_names)


class TestProbeDataset:
    def test_multilabel_flag_set_from_label_shape(self) -> None:
        ds = _make_separable_binary_dataset()
        assert ds.is_multilabel is True
        assert ds.num_classes == 1

    def test_multiclass_flag_set_from_label_shape(self) -> None:
        ds = _make_separable_multiclass_dataset()
        assert ds.is_multilabel is False
        assert ds.num_classes == 4


class TestLinearProbeForward:
    def test_multilabel_returns_sigmoid_outputs_in_zero_one(self) -> None:
        probe = LinearProbe(in_features=16, num_classes=3, multilabel=True)
        x = torch.randn(4, 16)
        out = probe(x)
        assert out.shape == (4, 3)
        assert (out >= 0).all() and (out <= 1).all()

    def test_multiclass_returns_softmax_distribution(self) -> None:
        probe = LinearProbe(
            in_features=16, num_classes=4, multilabel=False, label_names=list("abcd")
        )
        x = torch.randn(2, 16)
        out = probe(x)
        assert out.shape == (2, 4)
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_invalid_label_names_length_raises(self) -> None:
        with pytest.raises(ValueError):
            LinearProbe(in_features=4, num_classes=3, multilabel=False, label_names=["a", "b"])


class TestTrainLinearProbe:
    def test_binary_probe_separates_two_gaussians(self) -> None:
        dataset = _make_separable_binary_dataset(n_pos=80, n_neg=80)
        probe, metrics = train_linear_probe(dataset, epochs=200, seed=0)
        assert isinstance(probe, LinearProbe)
        # Separable data → high AUROC (the rank statistic is robust to
        # threshold; the small val split makes raw accuracy noisier).
        assert metrics.val_auroc is not None
        assert metrics.val_auroc > 0.9
        # Loss curve is mostly decreasing.
        assert metrics.train_loss[-1] < metrics.train_loss[0]

    def test_multiclass_probe_separates_four_gaussians(self) -> None:
        dataset = _make_separable_multiclass_dataset()
        probe, metrics = train_linear_probe(dataset, epochs=120, seed=0)
        assert isinstance(probe, LinearProbe)
        assert metrics.val_accuracy > 0.9
        # Multi-class AUROC is not currently computed; expect None.
        assert metrics.val_auroc is None

    def test_rejects_tiny_dataset(self) -> None:
        with pytest.raises(ValueError):
            train_linear_probe(
                ProbeDataset(
                    embeddings=torch.zeros(1, 4),
                    labels=torch.tensor([0]),
                    label_names=["x"],
                ),
                epochs=5,
            )

    def test_batched_training_runs(self) -> None:
        dataset = _make_separable_binary_dataset(n_pos=20, n_neg=20)
        probe, metrics = train_linear_probe(dataset, epochs=20, batch_size=8, seed=0)
        assert metrics.num_epochs == 20
        assert isinstance(probe, LinearProbe)


class TestSaveLoadRoundtrip:
    def test_single_probe_roundtrip(self, tmp_path: Path) -> None:
        dataset = _make_separable_binary_dataset()
        probe, _ = train_linear_probe(dataset, epochs=20, seed=0)
        path = tmp_path / "probe.pt"
        save_linear_probe(probe, path)
        reloaded = load_linear_probe(path)
        x = torch.randn(4, dataset.num_features)
        torch.testing.assert_close(probe(x), reloaded(x), atol=1e-6, rtol=1e-6)
        assert reloaded.label_names == probe.label_names
        assert reloaded.multilabel == probe.multilabel

    def test_probe_manifest_roundtrip(self, tmp_path: Path) -> None:
        binary_ds = _make_separable_binary_dataset()
        multi_ds = _make_separable_multiclass_dataset()
        p1, _ = train_linear_probe(binary_ds, epochs=20, seed=0)
        p2, _ = train_linear_probe(multi_ds, epochs=20, seed=0)
        probes = {"is_animal": p1, "category": p2}
        save_probe_manifest(probes, tmp_path / "probes")
        reloaded = load_probe_manifest(tmp_path / "probes")
        assert set(reloaded.keys()) == {"is_animal", "category"}
        x = torch.randn(2, binary_ds.num_features)
        torch.testing.assert_close(reloaded["is_animal"](x), p1(x), atol=1e-6, rtol=1e-6)


class TestEvaluateProbes:
    def test_binary_probe_returns_single_float(self) -> None:
        dataset = _make_separable_binary_dataset()
        probe, _ = train_linear_probe(dataset, epochs=80, seed=0)
        # Positive-class direction should yield > 0.5.
        positive_dir = torch.zeros(dataset.num_features)
        positive_dir[0] = 3.0
        results = evaluate_probes(positive_dir, {"is_pos": probe})
        assert isinstance(results["is_pos"], float)
        assert results["is_pos"] > 0.5

    def test_multiclass_probe_returns_distribution(self) -> None:
        dataset = _make_separable_multiclass_dataset(num_classes=3)
        probe, _ = train_linear_probe(dataset, epochs=120, seed=0)
        # Class-1 direction in feature 1 should win.
        focus = torch.zeros(dataset.num_features)
        focus[1] = 3.0
        out = evaluate_probes(focus, {"category": probe})
        assert isinstance(out["category"], dict)
        dist = out["category"]
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-5)
        # Argmax should be class 1.
        assert max(dist, key=dist.get) == "c1"
