"""
Utility and convenience functions for plotting.
"""
from __future__ import annotations

import logging
from typing import Sequence, TypeAlias

import matplotlib.cm as cm
import matplotlib.colors as cl
import numpy as np
from numpy.typing import NDArray

# types
RGBAColor: TypeAlias = Sequence[float]

# custom colormap
_color_dict = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.1),
        (1.0, 1.0, 1.0),
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    'blue': (
        (0.0, 0.0, 1.0),
        (0.5, 0.1, 0.0),
        (1.0, 0.0, 0.0),
    )
}
BlackSeismic = cl.LinearSegmentedColormap("BlackSeismic", _color_dict)


def sample_cmap(colormap: str, samples: int, index: int | None) -> RGBAColor:
    """
    Return a color sample from the given color map.

    This function requires the name of the colormap, the number of
    samples (to get the spacing right) and the index of the sample to
    return. The color is returned as RGBA color tuple.

    :param colormap: Name of the colormap to sample.
    :param samples: The number of evenly-spaced samples.
    :param index: The index of the sample for which to return the color.
    :return: An RGBA color as a tuple of floats.
    """
    if samples > 1001:
        logging.warning(
            "You are attempting to sample many values from a colormap. This "
            "may lead to unexpected results for high indices."
        )
    cmap = cm.get_cmap(colormap)
    stepsize = 1 / (samples - 1)
    indices = np.arange(0, 1.001, stepsize)
    return cmap(indices[index])


def custom_cmap(
    color_end: RGBAColor, color_start: RGBAColor = (256, 256, 256)
) -> cl.ListedColormap:
    """
    Return a colormap going to the specified color.

    The colormap can be used as a normal matplotlib colormap, for example
    in histograms. By default, the colormap starts at white, but the
    start color can also be given to achieve arbitrary, custom color
    maps between two colors. The color always uniformly transforms in
    each channel separately.

    :param color_end: The end color of the colormap in RGB. The colormap
        will go uniformly from to this color.
    :param color_start: The start color of the colormap in RGB. The
        colormap will start at this color and uniformly transform into
        the end color. Defaults to white.
    :return: Uniform colormap from white to the specified color.
    """
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(color_start[0] / 256, color_end[0] / 256, N)
    vals[:, 1] = np.linspace(color_start[1] / 256, color_end[1] / 256, N)
    vals[:, 2] = np.linspace(color_start[2] / 256, color_end[2] / 256, N)
    return cl.ListedColormap(vals)


def get_errorbar_lengths(
    median: NDArray, percentiles: NDArray | Sequence[NDArray]
) -> NDArray:
    """
    Return the error bar lengths as (2, N) shape array.

    The percentiles must have shape (2, N) where N is the length of the
    median array. The first entry on axis zero must be the position of
    the lower percentiles, the second the positions of the upper
    percentiles. The returned array has the same shape, but as values
    it holds the length of the respective error bar.


    :param median: Array of median values of length N.
    :param percentiles: Array of lower and upper percentiles. Must be of
        shape (2, N). First entry of first axis must be the position of
        the lower percentiles, second entry that of the upper percentiles.
    :return: Array of shape (2, N) giving the length of the lower and
        upper error bar respectively.
    """
    lower_ebars = median - percentiles[0]
    upper_ebars = percentiles[1] - median
    return np.array([lower_ebars, upper_ebars])
