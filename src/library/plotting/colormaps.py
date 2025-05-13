from __future__ import annotations

import logging
from typing import Sequence, TypeAlias

import matplotlib.colors
import numpy as np
from matplotlib import cm as cm
from matplotlib import colors as cl

# types
RGBColor: TypeAlias = tuple[float, float, float]
RGBAColor: TypeAlias = tuple[float, float, float, float]

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


# DEPRECATE: cmasher has a function like this: take_cmap_colors()
def sample_cmap(
    colormap: str,
    samples: int,
    index: int | Sequence[int] | None = None,
) -> RGBAColor | Sequence[RGBAColor]:
    """
    Return a color sample from the given color map.

    This function requires the name of the colormap, the number of
    samples (to get the spacing right) and the index of the sample to
    return. The color is returned as RGBA color tuple.

    If the ``index`` is set to None, the function instead returns not a
    single color value, but an array of colors, sampled in evenly spaced
    positions.

    :param colormap: Name of the colormap to sample.
    :param samples: The number of evenly-spaced samples.
    :param index: The index of the sample for which to return the color.
        If set to None, function returns all samples as an array of
        colors. If set to a valid array slice, only that slice of colors
        is returned.
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
    if index is None:
        return cmap(indices[:])
    return cmap(indices[index])


def custom_cmap(
    color_end: RGBColor | RGBAColor,
    color_start: RGBColor | RGBAColor = (1., 1., 1.),
) -> cl.ListedColormap:
    """
    Return a colormap going to the specified color.

    The colormap can be used as a normal matplotlib colormap, for example
    in histograms. By default, the colormap starts at white, but the
    start color can also be given to achieve arbitrary, custom color
    maps between two colors. The color always uniformly transforms in
    each channel separately.

    .. warning:: The color must be given as either RGB or RGBA,
        represented by floating point values for each channel.

    :param color_end: The end color of the colormap in RGB. The colormap
        will go uniformly from to this color.
    :param color_start: The start color of the colormap in RGB. The
        colormap will start at this color and uniformly transform into
        the end color. Defaults to white.
    :return: Uniform colormap from white to the specified color.
    """
    n = 256
    vals = np.ones((n, 4))
    vals[:, 0] = np.linspace(color_start[0], color_end[0], n)
    vals[:, 1] = np.linspace(color_start[1], color_end[1], n)
    vals[:, 2] = np.linspace(color_start[2], color_end[2], n)
    return cl.ListedColormap(vals)


def two_slope_cmap(
    lower: str | matplotlib.colors.Colormap,
    upper: str | matplotlib.colors.Colormap,
    vmin: float,
    vcenter: float,
    vmax: float,
    register_name: str = "custom_map",
) -> tuple[matplotlib.colors.Colormap, matplotlib.colors.TwoSlopeNorm] | None:
    """
    Return a colormap and a norm for a two-sloped quantity.

    Function takes two colormaps or names of known named colormaps and
    minimum, central, and maximum value of a color range, and returns a
    colormap that is stitched together from the ``lower`` colormap for
    values falling in between ``vmin`` and ``vcenter``, and the ``upper``
    colormap for values in between ``vcenter`` and ``vmax``. To use this
    colormap accordingly, a matplotlib norm object, configured for the
    given values is returned alongside the colormap.

    .. note:: Colormaps created using the :func:`custom_cmap` function
        are valid colormaps to use for ``lower`` and ``upper``.

    :param lower: Name of a registered colormap or colormap object. Must
        be able to sample the map by indexing. This will be the colormap
        for the lower value range from ``vmin`` to ``vcenter``.
    :param upper: Name of a registered colormap or colormap object. Must
        be able to sample the map by indexing. This will be the colormap
        for the upper value range from ``vcenter`` to ``vmax``.
    :param vmin: The lowest value to be plotted on the color range of
        ``lower``, i.e. the value that will be mapped to 0 on the
        ``lower`` colormap.
    :param vcenter: The central value, where the two colormaps touch.
    :param vmax: The highest value to be plotted on the color range of
        ``upper``, i.e. the value that will be mapped to 1 on the
        ``upper`` colormap.
    :param register_name: The name of the colormap. Useful to set if
        the colormap should be registered as a named colormap later.
    :return: Tuple of the colormap created from the two specified maps,
        and a ``TwoSlopeNorm`` configured according to the minimum,
        central, and maximum values supplied. If the supplied colormaps
        cannot be indexed, the function returns None.
    """
    # get indexable colormaps from names
    if isinstance(lower, str):
        try:
            lower = getattr(matplotlib.cm, lower)
        except AttributeError:
            logging.error(f"'{lower}' is not a registered colormap name.")
            return
    if isinstance(upper, str):
        try:
            upper = getattr(matplotlib.cm, upper)
        except AttributeError:
            logging.error(f"'{upper}' is not a registered colormap name.")
            return

    # stitch together the maps
    samples = np.linspace(0, 1, 256)
    lower_map = lower(samples)
    upper_map = upper(samples)
    full_map = np.vstack([lower_map, upper_map])

    # create actual colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        register_name, full_map
    )

    # create a two-slope-norm
    norm = matplotlib.colors.TwoSlopeNorm(vcenter, vmin=vmin, vmax=vmax)

    return cmap, norm
