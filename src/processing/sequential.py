"""
Fuctions for sequential data processing.

See the ``parallelization`` module for better performance.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def process_halo_data_sequentially(
    callback: Callable[..., NDArray],
    halo_ids: Sequence[int],
    data_shape: tuple[int],
    kwargs: dict[str, Any] | None = None,
    quiet: bool = False,
) -> None:
    """
    Process halo data sequentially.

    This method calls the given Callable ``callback`` with every halo
    ID in ``halo_ids`` in order or appearance. Optionally, keyworded
    arguments can be supplied to the Callable. The resulting data is
    bundled into one array and returned.

    To achieve a function that will take a halo ID and return processed
    data, it is advised to create a wrapper function around the DAQ
    function for temperature acquisition and the corresponding processing
    function.

    :param callback: A function to process halo data. Its signature is
        arbitrary, but it must take as first positional argument a halo
        ID. Further arguments may be supplied by ``kwargs``, but no
        verification is performed to ensure all arguments are matched.
        Callable must return the processed temperatures, NOT the values
        of the temperature!
    :param halo_ids: Sequence of halo IDs for all halos to process.
    :param data_shape: The shape of the NDArray that ``callback`` returns.
    :param kwargs: A dictionary of keyworded arguments for ``callback``.
        Defaults to None which is equivalent to no further arguments.
    :param quiet: Whether to suppress progress updates. Defaults to False.
    """
    if kwargs is None:
        kwargs = {}
    logging.info("Start processing halo data sequentially.")
    n_halos = len(halo_ids)
    data = np.zeros((n_halos, ) + data_shape)
    for i, halo_id in enumerate(halo_ids):
        if not quiet:
            perc = i / n_halos * 100
            print(f"Processing halo {i}/{n_halos} ({perc:.1f}%)", end="\r")
        data[i] = callback(halo_id, **kwargs)
    logging.info("Finished processing halo data.")
    return data
