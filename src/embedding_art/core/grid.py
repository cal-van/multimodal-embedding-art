"""
Grid generation utilities for concept interpolation.

Provides functions for creating 1D (row) and 2D (grid) interpolations
between concepts, and stitching the resulting images into a composite.
"""

from __future__ import annotations

from PIL import Image

from embedding_art.core.concept import Concept


def interpolate_1d_grid(
    concept_a: Concept,
    concept_b: Concept,
    cols: int,
) -> list[Concept]:
    """
    Create a 1D interpolation between two concepts.

    Uses spherical linear interpolation (slerp) to generate
    intermediate concepts along the geodesic path on the embedding
    hypersphere.

    Args:
        concept_a: Starting concept (t=0)
        concept_b: Ending concept (t=1)
        cols: Number of interpolation points (columns)

    Returns:
        List of concepts from concept_a to concept_b
    """
    if cols < 2:
        raise ValueError("cols must be at least 2 for interpolation")

    concepts = []
    for i in range(cols):
        t = i / (cols - 1) if cols > 1 else 0.5
        interpolated = Concept.slerp(concept_a, concept_b, t)
        concepts.append(interpolated)

    return concepts


def interpolate_2d_grid(
    corners: list[Concept],
    size: int,
) -> list[list[Concept]]:
    """
    Create a 2D interpolation grid from four corner concepts.

    Uses bilinear interpolation via slerp: first interpolates along
    the top and bottom edges, then interpolates vertically between
    each corresponding pair.

    Corner order: [top-left, top-right, bottom-left, bottom-right]

    Args:
        corners: List of 4 corner concepts in order:
                 [top-left, top-right, bottom-left, bottom-right]
        size: Grid dimension (creates size x size grid)

    Returns:
        List of lists (rows) of concepts, indexed as grid[row][col]
    """
    if len(corners) != 4:
        raise ValueError("corners must contain exactly 4 concepts")

    if size < 2:
        raise ValueError("size must be at least 2 for interpolation")

    top_left, top_right, bottom_left, bottom_right = corners

    grid = []
    for row in range(size):
        row_t = row / (size - 1) if size > 1 else 0.5

        # Interpolate left edge at this row
        left_concept = Concept.slerp(top_left, bottom_left, row_t)
        # Interpolate right edge at this row
        right_concept = Concept.slerp(top_right, bottom_right, row_t)

        # Interpolate across the row
        row_concepts = []
        for col in range(size):
            col_t = col / (size - 1) if size > 1 else 0.5
            cell_concept = Concept.slerp(left_concept, right_concept, col_t)
            row_concepts.append(cell_concept)

        grid.append(row_concepts)

    return grid


def stitch_grid(images: list[list[Image.Image]]) -> Image.Image:
    """
    Stitch a 2D grid of images into a single composite image.

    All images are assumed to be the same size. The first image's
    dimensions are used to calculate the output size.

    Args:
        images: 2D list of PIL Images, indexed as images[row][col]

    Returns:
        Single PIL Image containing all input images arranged in a grid
    """
    if not images or not images[0]:
        raise ValueError("images cannot be empty")

    # Get dimensions from first image
    first_img = images[0][0]
    img_width, img_height = first_img.size

    rows = len(images)
    cols = len(images[0])

    # Create output image
    output = Image.new("RGB", (cols * img_width, rows * img_height))

    # Paste each image into position
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            x = col_idx * img_width
            y = row_idx * img_height
            output.paste(img, (x, y))

    return output
