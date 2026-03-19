"""
Tests for ConceptSpec — encoder-agnostic concept specification.

ConceptSpec describes *what* to render without binding to any encoder.
It is intentionally decoupled from the Concept class (which holds an
encoder-specific embedding).

Test coverage:
- Construction with each modality individually
- Multimodal construction (multiple modalities at once)
- Weight field default and custom value
- describe() output for every combination
- __repr__ delegates to describe()
- Frozen (immutable) dataclass behaviour
"""

import pytest

from embedding_art.core.concept_spec import ConceptSpec

# ---------------------------------------------------------------------------
# Construction — individual modalities
# ---------------------------------------------------------------------------


class TestConceptSpecConstruction:
    def test_text_only(self):
        spec = ConceptSpec(text="a golden retriever")
        assert spec.text == "a golden retriever"
        assert spec.image is None
        assert spec.audio is None
        assert spec.video is None

    def test_image_only(self, tmp_path):
        img = tmp_path / "cat.png"
        spec = ConceptSpec(image=img)
        assert spec.image == img
        assert spec.text is None
        assert spec.audio is None
        assert spec.video is None

    def test_audio_only(self, tmp_path):
        audio = tmp_path / "bark.wav"
        spec = ConceptSpec(audio=audio)
        assert spec.audio == audio
        assert spec.text is None
        assert spec.image is None
        assert spec.video is None

    def test_video_only(self, tmp_path):
        video = tmp_path / "ocean.mp4"
        spec = ConceptSpec(video=video)
        assert spec.video == video
        assert spec.text is None
        assert spec.image is None
        assert spec.audio is None

    def test_empty_spec(self):
        """All fields optional — empty spec is valid."""
        spec = ConceptSpec()
        assert spec.text is None
        assert spec.image is None
        assert spec.audio is None
        assert spec.video is None


# ---------------------------------------------------------------------------
# Weight field
# ---------------------------------------------------------------------------


class TestConceptSpecWeight:
    def test_default_weight_is_one(self):
        spec = ConceptSpec(text="sunset")
        assert spec.weight == 1.0

    def test_custom_weight(self):
        spec = ConceptSpec(text="sunset", weight=0.5)
        assert spec.weight == 0.5

    def test_weight_zero(self):
        spec = ConceptSpec(text="silence", weight=0.0)
        assert spec.weight == 0.0

    def test_weight_negative(self):
        """Negative weights are structurally valid (subtraction in blending)."""
        spec = ConceptSpec(text="noise", weight=-1.0)
        assert spec.weight == -1.0

    def test_weight_above_one(self):
        spec = ConceptSpec(text="intense", weight=2.5)
        assert spec.weight == 2.5


# ---------------------------------------------------------------------------
# Multimodal construction
# ---------------------------------------------------------------------------


class TestConceptSpecMultimodal:
    def test_text_and_image(self, tmp_path):
        img = tmp_path / "reference.jpg"
        spec = ConceptSpec(text="majestic", image=img)
        assert spec.text == "majestic"
        assert spec.image == img

    def test_all_four_modalities(self, tmp_path):
        img = tmp_path / "photo.png"
        audio = tmp_path / "sound.wav"
        video = tmp_path / "clip.mp4"
        spec = ConceptSpec(text="vivid", image=img, audio=audio, video=video)
        assert spec.text == "vivid"
        assert spec.image == img
        assert spec.audio == audio
        assert spec.video == video


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


class TestConceptSpecDescribe:
    def test_text_describe(self):
        spec = ConceptSpec(text="ocean waves")
        assert spec.describe() == 'text:"ocean waves"'

    def test_image_describe(self, tmp_path):
        img = tmp_path / "cat.png"
        spec = ConceptSpec(image=img)
        assert spec.describe() == "image:cat.png"

    def test_audio_describe(self, tmp_path):
        audio = tmp_path / "bark.wav"
        spec = ConceptSpec(audio=audio)
        assert spec.describe() == "audio:bark.wav"

    def test_video_describe(self, tmp_path):
        video = tmp_path / "ocean.mp4"
        spec = ConceptSpec(video=video)
        assert spec.describe() == "video:ocean.mp4"

    def test_empty_describe(self):
        spec = ConceptSpec()
        assert spec.describe() == "empty"

    def test_multimodal_describe_order(self, tmp_path):
        """Parts appear in text → image → audio → video order."""
        img = tmp_path / "photo.jpg"
        audio = tmp_path / "sound.wav"
        spec = ConceptSpec(text="serene", image=img, audio=audio)
        assert spec.describe() == 'text:"serene" + image:photo.jpg + audio:sound.wav'

    def test_custom_weight_wraps_description(self):
        spec = ConceptSpec(text="dream", weight=0.3)
        assert spec.describe() == '0.3*(text:"dream")'

    def test_weight_one_does_not_wrap(self):
        """Weight of exactly 1.0 should not add the wrapping prefix."""
        spec = ConceptSpec(text="dream", weight=1.0)
        assert spec.describe() == 'text:"dream"'

    def test_empty_with_custom_weight(self):
        spec = ConceptSpec(weight=2.0)
        assert spec.describe() == "2.0*(empty)"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestConceptSpecRepr:
    def test_repr_wraps_describe(self):
        spec = ConceptSpec(text="fire")
        assert repr(spec) == 'ConceptSpec(text:"fire")'

    def test_repr_empty(self):
        spec = ConceptSpec()
        assert repr(spec) == "ConceptSpec(empty)"

    def test_repr_with_weight(self):
        spec = ConceptSpec(text="ice", weight=0.7)
        assert repr(spec) == 'ConceptSpec(0.7*(text:"ice"))'


# ---------------------------------------------------------------------------
# Immutability (frozen dataclass)
# ---------------------------------------------------------------------------


class TestConceptSpecImmutability:
    def test_text_field_immutable(self):
        spec = ConceptSpec(text="hello")
        with pytest.raises((AttributeError, TypeError)):
            spec.text = "world"  # type: ignore[misc]

    def test_weight_field_immutable(self):
        spec = ConceptSpec(weight=0.5)
        with pytest.raises((AttributeError, TypeError)):
            spec.weight = 1.0  # type: ignore[misc]

    def test_image_field_immutable(self, tmp_path):
        spec = ConceptSpec(image=tmp_path / "a.png")
        with pytest.raises((AttributeError, TypeError)):
            spec.image = tmp_path / "b.png"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Equality and hashing (frozen dataclass provides these automatically)
# ---------------------------------------------------------------------------


class TestConceptSpecEquality:
    def test_equal_text_specs(self):
        assert ConceptSpec(text="hello") == ConceptSpec(text="hello")

    def test_unequal_text_specs(self):
        assert ConceptSpec(text="hello") != ConceptSpec(text="world")

    def test_equal_with_weight(self):
        assert ConceptSpec(text="x", weight=0.5) == ConceptSpec(text="x", weight=0.5)

    def test_unequal_weight(self):
        assert ConceptSpec(text="x", weight=0.5) != ConceptSpec(text="x", weight=1.0)

    def test_hashable(self):
        """Frozen dataclasses must be hashable for use in sets/dicts."""
        spec = ConceptSpec(text="alpha")
        assert hash(spec) is not None
        s = {spec}
        assert spec in s

    def test_hashable_with_path(self, tmp_path):
        img = tmp_path / "img.png"
        spec = ConceptSpec(image=img)
        assert hash(spec) is not None
