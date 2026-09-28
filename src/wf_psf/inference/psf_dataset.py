"""PSF dataset.

Defines the dataset object containing the data required for PSF model generation.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from dataclasses import dataclass
from numpy.typing import ArrayLike


@dataclass
class PSFDataset:
    """Dataset for PSF model generation.

    Attributes
    ----------
    positions: ArrayLike
        PSF positions. Shape: (n_sources, 2), where the last dimension
        contains [x, y] coordinates.
    seds : ArrayLike
        Spectral energy distributions for each source. Each SED contains
        wavelength positions and corresponding SED values.
        Shape: (n_sources, n_wavelengths, 2), where the last dimension
        contains [wavelength, SED value].
    sources : ArrayLike, optional
        Postage stamps of sources, e.g. star images.
        Shape: (n_sources, n_pixel, n_pixel).
    masks : ArrayLike, optional
        Masks corresponding to the source postage stamps.
        Shape: (n_sources, n_pixel, n_pixel). Defaults to None.
    """

    positions: ArrayLike
    seds: ArrayLike
    sources: ArrayLike | None = None
    masks: ArrayLike | None = None
