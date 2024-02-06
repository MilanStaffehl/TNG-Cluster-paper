"""
Tools to load data for temperature distribution histograms from file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_histogram_data(
    filepath: str | Path,
    expected_shape: tuple[int, int] | None = None
) -> tuple[NDArray, NDArray, NDArray] | None:
    """
    Load stacked (averaged) histogram data from file.

    The file needs to be a numpy .npz archive, as saved by the method
    ``_post_process_data``. The resulting NpzFile instance must have
    keys 'hist_mean', 'hist_median' and 'hist_percentiles'. For all
    three arrays, the first axis must match in length the number of
    mass bins and the second axis must match the number of histogram
    bins ``self.len_data``.

    The loaded data is placed into the ``histograms_mean``,
    ``histograms_median`` and ``histograms_percentiles`` attributes
    respectively.

    :param filepath: file name of the numpy data file.
    :param expected_shape: The shape that the mean and median arrays
        are expected to have. If given, the function will verify that
        the loaded data has this shape, otherwise the loaded data will
        be returned unchecked.
    :return: Tuple of the histogram mean, median and percentiles, in
        that order.
    """
    logging.info("Loading saved histogram data from file.")
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file():
        logging.error(f"The given file {str(filepath)} is not a valid file.")
        return

    # attempt to load the data
    with np.load(filepath) as hist_data:
        hist_mean = hist_data["hist_mean"]
        hist_median = hist_data["hist_median"]
        hist_perc = hist_data["hist_percentiles"]
        halos_per_bin = hist_data["halos_per_bin"]

    if not expected_shape:
        # no verificatin of data
        logging.info("Returning histogram data without verification.")
        return hist_mean, hist_median, hist_perc, halos_per_bin

    # verify data:
    if not hist_median.shape == expected_shape:
        logging.error(
            f"Loaded histogram data does not match expected data in "
            f"shape:\nExpected shape: {expected_shape}, received shape: "
            f"{hist_mean.shape}"
        )
        return
    elif not hist_mean.shape == hist_median.shape:
        logging.error(
            f"The histogram mean array doe not match the median array in "
            f"shape: mean has shape {hist_mean.shape} but the median has "
            f"shape {hist_median.shape}. Median has correct shape, so the "
            f"mean data must have been saved wrong or is corrupted."
        )
        return
    elif (hist_perc.shape[0] != expected_shape[0]
          or hist_perc.shape[2] != expected_shape[1]):
        logging.error(
            f"Shape of histogram median is different from shape of "
            f"histogram percentiles:\nMedian shape: {hist_mean.shape}, "
            f"percentiles shape: {hist_perc.shape}\n"
            f"The medians have the expected shape; the percentiles must "
            f"have been saved wrong or are corrupted."
        )
        return
    else:
        logging.info("Successfully loaded and verified histogram data!")
        return hist_mean, hist_median, hist_perc, halos_per_bin


def load_virial_temperatures(filepath: str | Path) -> NDArray | None:
    """
    Load virial temperature data from file and return it.

    The file needs to be a numpy .npy file, as saved by the method
    ``get_virial_temperature``. The loaded data is placed into the
    ``virial_temperature`` attribute for use in plotting.

    :param filepath: File name of the numpy data file containing the
        virial temperatures.
    :return: Array of the virial temperatures.
    """
    logging.info("Loading saved virial temperature data from file.")
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file():
        logging.error(f"The given file {str(filepath)} is not a valid file.")
        return

    # attempt to load the data
    virial_temperatures = np.load(filepath)
    logging.info("Successfully loaded virial temperatures.")
    return virial_temperatures


def load_gallery_plot_data(
    filepath: str | Path,
    expected_shape: tuple[int, int] | None = None
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray] | None:
    """
    Load data for gallery plots from file.

    The file needs to be a numpy .npz archive, containing fields 'masses',
    'radii', 'indices', 'virial_temperatures' and 'hists'. The first axis
    of the arrays must match in length the expected length, defined by
    two times the number of bins times the number of plots per gallery.

    The loaded data is returned as a tuple of numpy arrays.

    :param filepath: File name and path of the numpy data file.
    :param expected_shape: The expected shape of the loaded arrays. If
        left empty, the loaded data is returned without verification,
        otherwise it is checked for its shape. Defaults to None.
    :return: The mases, radii, indices, virial temperatures and histogram
        data arrays as a tuple of arrays, in that order. If data
        verification fails, returns None.
    """
    logging.info("Loading saved gallery data from file.")
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file():
        logging.error(f"The given file {str(filepath)} is not a valid file.")
        return

    # attempt to load the data
    with np.load(filepath) as data:
        masses = data["selected_masses"]
        radii = data["selected_radii"]
        indices = data["selected_halo_ids"]
        virial_temperatures = data["virial_temperatures"]
        hists = data["histograms"]

    if not expected_shape:
        logging.info("Returning loaded data without verification.")
        return masses, radii, indices, virial_temperatures, hists

    attrs = [masses, radii, indices, virial_temperatures]
    if not all([x.shape == expected_shape for x in attrs]):
        logging.error(
            "Some of the loaded data does not have the expected "
            "number of entries. Data could not be loaded.\n"
            f"Shapes: {[x.shape for x in attrs]}."
        )
        return
    elif not hists.shape[:-1] == expected_shape:
        logging.error(
            f"The histogram data does not provide the expected number of "
            f"histrograms: Expected {expected_shape} hists, but got "
            f"{hists.shape} hists instead. Aborting loading of data."
        )
        return
    else:
        logging.info("Successfully loaded verified data.")
        return masses, radii, indices, virial_temperatures, hists
