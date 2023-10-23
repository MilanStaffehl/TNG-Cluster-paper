"""
Utility and convenience functions for plotting.
"""
import logging
from typing import Sequence, TypeAlias

import matplotlib.cm as cm
import matplotlib.colors as cl
import numpy as np

RGBAColor: TypeAlias = Sequence[float]


def sample_cmap(colormap: str, samples: int, index: int) -> RGBAColor:
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


def custom_cmap(color: tuple[float, float, float]) -> cl.ListedColormap:
    """
    Return a colormap going from white to the specified color.

    The colormap can be used as a normal matplotlib colormap, for example
    in histograms.

    :param color: The color in RGB. The colormap will go uniformly from
        to this color.
    :return: Uniform colormap from white to the specified color.
    """
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, color[0] / 256, N)
    vals[:, 1] = np.linspace(1, color[1] / 256, N)
    vals[:, 2] = np.linspace(1, color[2] / 256, N)
    return cl.ListedColormap(vals)
