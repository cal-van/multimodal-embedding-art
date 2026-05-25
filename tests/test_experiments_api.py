"""Tests for /experiments/anchor-compare.

Mocks the encoder + text_anchor_readout so the test runs in milliseconds
without loading LanguageBind weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from embedding_art.web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _mock_registry_with_encoder(embeddings: dict[str, torch.Tensor]) -> MagicMock:
    """Build a mock create_default_registry()/encoder that returns the
    requested embedding for each text input."""

    def encode_text(text: str) -> torch.Tensor:
        if text not in embeddings:
            raise KeyError(f"unmocked text: {text}")
        return embeddings[text].unsqueeze(0)

    encoder = MagicMock()
    encoder.encode_text.side_effect = encode_text
    registry = MagicMock()
    registry.load.return_value = encoder
    return registry


class TestAnchorCompareEndpoint:
    """POST /experiments/anchor-compare returns text encodings + cosines."""

    def test_returns_pairwise_cosines_for_three_texts(self, client: TestClient) -> None:
        # Construct three deterministic embeddings.
        e_a = torch.tensor([1.0, 0.0, 0.0])
        e_b = torch.tensor([0.0, 1.0, 0.0])
        # 'c' deliberately close to 'a' (should yield highest similarity vs a).
        e_c = torch.tensor([0.9, 0.1, 0.0])
        registry = _mock_registry_with_encoder({"a": e_a, "b": e_b, "c": e_c})

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=registry,
        ):
            with patch(
                "embedding_art.interpretation.text_anchor.text_anchor_readout",
                return_value=[("anchor1", 0.9), ("anchor2", 0.8)],
            ):
                response = client.post(
                    "/experiments/anchor-compare",
                    json={
                        "concept_label": "test",
                        "texts": ["a", "b", "c"],
                    },
                )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["concept_label"] == "test"
        assert data["encoder"] == "languagebind"
        assert len(data["entries"]) == 3

        # Cosine matrix is symmetric.
        matrix = data["cosine_matrix"]
        labels = ["a", "b", "c"]
        for li in labels:
            for lj in labels:
                assert li in matrix
                assert lj in matrix[li]
                assert matrix[li][lj] == pytest.approx(matrix[lj][li])
            assert matrix[li][li] == pytest.approx(1.0)
        # a-c should be highly similar (constructed to be).
        assert matrix["a"]["c"] > matrix["a"]["b"]

        # Text-anchor records present per entry.
        for entry in data["entries"]:
            assert isinstance(entry["text_anchor"], list)
            if entry["text_anchor"]:
                assert "word" in entry["text_anchor"][0]
                assert "similarity" in entry["text_anchor"][0]

    def test_at_least_two_texts_required(self, client: TestClient) -> None:
        response = client.post(
            "/experiments/anchor-compare",
            json={"concept_label": "test", "texts": ["only one"]},
        )
        assert response.status_code == 422

    def test_blank_texts_filtered_out(self, client: TestClient) -> None:
        """Blank entries are stripped before the at-least-two check."""
        response = client.post(
            "/experiments/anchor-compare",
            json={"concept_label": "test", "texts": ["one", "   ", ""]},
        )
        assert response.status_code == 422

    def test_anchor_readout_failure_is_non_fatal(self, client: TestClient) -> None:
        """If text_anchor_readout raises, the endpoint still returns the
        embeddings + cosines and just omits the anchor records."""
        registry = _mock_registry_with_encoder(
            {"foo": torch.tensor([1.0, 0.0]), "bar": torch.tensor([0.0, 1.0])}
        )

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=registry,
        ):
            with patch(
                "embedding_art.interpretation.text_anchor.text_anchor_readout",
                side_effect=RuntimeError("boom"),
            ):
                response = client.post(
                    "/experiments/anchor-compare",
                    json={"concept_label": "x", "texts": ["foo", "bar"]},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["entries"][0]["text_anchor"] == []
        assert data["entries"][1]["text_anchor"] == []
