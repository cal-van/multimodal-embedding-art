"""Implicit Neural Representation generator for direct pixel optimization.

Instead of optimizing a VAE latent, this generator directly optimizes a
neural network that maps (x, y) coordinates to RGB values. The network
uses Fourier features for natural frequency decomposition.

The key insight is that Fourier feature embeddings of coordinates give the
network a natural inductive bias toward smooth, multi-scale image structure —
high-frequency components only activate when the loss demands them, acting as
implicit regularization.
"""

import torch
import torch.nn as nn


class FourierFeatureNetwork(nn.Module):
    """MLP with Fourier feature input encoding.

    Maps (x, y) coordinates to RGB via learned Fourier features.
    This provides natural coarse-to-fine frequency decomposition,
    acting as implicit regularization.

    Args:
        height: Output image height in pixels.
        width: Output image width in pixels.
        hidden_dim: Number of hidden units in each MLP layer.
        n_layers: Total number of layers in the MLP (including input/output).
            Must be >= 2.
        n_frequencies: Number of Fourier frequency bands.  The input to the
            MLP is ``2 * n_frequencies`` (sin + cos of each projection).
        device: PyTorch device string.
    """

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_frequencies: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self.height = height
        self.width = width
        self._device_str = device

        # Fourier feature frequency matrix — learnable so the network can
        # adapt which frequencies are most useful for the target concept.
        self.B = nn.Parameter(torch.randn(n_frequencies, 2, device=device) * 10.0)

        # MLP: 2*n_frequencies → hidden → ... → 3 (RGB)
        # We need at least an input layer and an output layer.
        input_dim = n_frequencies * 2  # sin + cos concatenated
        n_layers = max(n_layers, 2)

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 3))
        self.mlp = nn.Sequential(*layers).to(device)

        # Pre-compute the coordinate grid once and register as a non-parameter
        # buffer so it moves with the module and is included in state_dict.
        ys = torch.linspace(-1, 1, height, device=device)
        xs = torch.linspace(-1, 1, width, device=device)
        # meshgrid returns (H, W) grids; stack → [H, W, 2], reshape → [H*W, 2]
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
        self.register_buffer("coords", grid.reshape(-1, 2))

    def forward(self) -> torch.Tensor:
        """Render the full image.

        Returns:
            Float tensor of shape ``[1, 3, H, W]`` with values in ``[0, 1]``.
        """
        # Project coordinates into Fourier feature space.
        # coords: [H*W, 2], B: [n_freq, 2] → projected: [H*W, n_freq]
        projected = self.coords @ self.B.T
        features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

        # MLP forward pass → RGB values in [0, 1].
        rgb = self.mlp(features)  # [H*W, 3]
        rgb = torch.sigmoid(rgb)

        # Reshape to [1, 3, H, W] image layout.
        return rgb.reshape(1, self.height, self.width, 3).permute(0, 3, 1, 2)


class INRGenerator:
    """DirectGenerator that optimizes an Implicit Neural Representation.

    Unlike VAE-based generators that decode from a latent space, this
    directly optimizes the network weights to produce the target image.
    The Fourier feature encoding provides natural regularization.

    Conforms to the ``DirectGenerator`` protocol defined in
    ``embedding_art.generators.base``.

    Args:
        height: Output image height in pixels.
        width: Output image width in pixels.
        hidden_dim: Number of hidden units in each MLP layer.
        n_layers: Total number of layers in the MLP.
        n_frequencies: Number of Fourier frequency bands.
        device: PyTorch device string.
    """

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_frequencies: int = 128,
        device: str = "cpu",
    ):
        self._device = torch.device(device)
        self._output_modality = "image"
        self._network = FourierFeatureNetwork(
            height=height,
            width=width,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_frequencies=n_frequencies,
            device=device,
        )

    @property
    def output_modality(self) -> str:
        """Output modality — always ``"image"`` for this generator."""
        return self._output_modality

    @property
    def device(self) -> torch.device:
        """Device the generator's network is on."""
        return self._device

    def get_optimizable_parameters(self) -> list[nn.Parameter]:
        """Return all network parameters for the optimizer.

        Returns:
            List of all ``torch.nn.Parameter`` objects in the network,
            including the Fourier frequency matrix ``B`` and all MLP weights.
        """
        return list(self._network.parameters())

    def render(self) -> torch.Tensor:
        """Render the current state of the network to an image.

        Returns:
            Float tensor of shape ``[1, 3, H, W]`` with values in ``[0, 1]``.
        """
        return self._network()
