"""
Utility and convenience functions for plotting.
"""
import logging
from typing import TypeAlias

import matplotlib.cm as cm
import numpy as np

RGBAColor: TypeAlias = tuple[float, float, float, float]


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
