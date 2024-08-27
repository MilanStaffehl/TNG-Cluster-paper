"""
Functions for parallelization of data processing.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Sequence

import multiprocess as mp
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def process_data_parallelized(
    callback: Callable[[int], NDArray],
    data: Sequence[int],
    processes: int,
    chunksize: int | None = None,
) -> NDArray:
    """
    Process data using multiprocessing.

    This method calls the given Callable ``callback`` with every entry
    in ``data`` in parallel. The chunking of data is done automatically
    based on the number of processes and the length of the list of halo
    indices. The callback can only have the described signature.

    To achieve a function that will take a single data point and return
    processed data, it is advised to create a wrapper function around the
    DAQ functions that are required.

    The function returns an ordered aray of the results of the call of
    the callback with the given data.

    :param callback: A function to process halo data. Must have the
        described signature (takes a single halo ID as argument, returns
        a numpy array of processed data). If the function requires more
        arguments, create a wrapper function that handles data injection.
        Callable must return the processed temperatures, NOT the values
        of the temperature!
    :param data: Sequence of data to process by handing it as the sole
        variable argument to the given callback. Can be for example a
        halo ID.
    :param processes: The number of processes to use.
    :param chunksize: The number of halos to process per subprocess. If
        left empty, an appropriate chunk size will be automatically
        calculated and used.
    :return: Array of the return values of ``callback`` after applying
        the argument list to it. Order is preserved as if it were
        sequentially processed.
    """
    logging.info("Start processing halo data on mutliple cores.")
    # determine a valid chunksize
    if chunksize is None or len(data) / processes < chunksize < 0:
        chunksize = round(len(data) / processes / 4, -2)
        logging.debug(f"Autosetting chunksize to {chunksize}.")
    else:
        chunksize = round(chunksize)
    # multiprocess the problem
    logging.info(
        f"Starting {processes} subprocesses with chunksize {chunksize}."
    )
    with mp.Pool(processes=processes) as pool:
        results = pool.map(callback, data, chunksize=int(chunksize))
        pool.close()
        pool.join()
    logging.info(f"Finished processing data on {processes} processes.")

    # return array of data
    return np.array(results)


def process_data_starmap(
    callback: Callable[..., NDArray | int],
    processes: int,
    *input_args: NDArray,
    chunksize: int | None = None,
) -> NDArray | None:
    """
    Process halo properties using multiprocessing starmap.

    The method calls the given Callable ``callback`` with the entries of
    the given ``input_args``. This means that the Callable must accept
    as many positional arguments as input argument arrays are given. The
    input arguments are fed into the callable column wise, i.e. input
    arguments

    ```
    process_halo_data_starmap(
        myfunc, 16, [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    )
    ```

    would result in the following parallelized function calls on 16 cores:

    ```
    myfunc(0, 5)
    myfunc(1, 6)
    myfunc(2, 7)
    myfunc(3, 8)
    myfunc(4, 9)
    ```

    This is useful for the calculation of quantities such as virial
    temperature, which depends on multiple halo properties.

    The function returns the ordered array of results of the call of
    ``callback``.

    :param callback: A function to process halo properties. Must take as
        many positional arguments as input args are given.
    :param processes: The number of processes to use.
    :param input_args: One-dimensional NDArrays of the same length. Must
        be as many as ``callback`` takes positional arguments.
    :param chunksize: The number of halos to process per subprocess. If
        left empty, an appropriate chunk size will be automatically
        calculated and used.
    :return: Array of the return values of ``callback`` for every column
        of applying ``input_args`` to ``callback``.
    """
    # verify data
    length = len(input_args[0])
    for arg in input_args:
        if not isinstance(arg, np.ndarray):
            logging.error(
                f"Received input args not of type NDArray: {type(arg)}"
            )
            return
        if len(arg.shape) > 1:
            logging.error(
                f"Input argument has too many dimensions: array has shape "
                f"{arg.shape}."
            )
            return
        if len(arg) != length:
            logging.error("Input arrays are not of the same length.")
            return

    # splice input args together
    argument_pairs = np.array(list(input_args)).transpose()
    # determine a valid chunksize
    if chunksize is None or length / processes < chunksize < 0:
        chunksize = round(length / processes / 4, -2)
        chunksize = max(chunksize, 1)
        logging.debug(f"Autosetting chunksize to {chunksize}.")
    else:
        chunksize = round(chunksize)
    # multiprocess the problem using starmap
    logging.info(
        f"Starting {processes} subprocesses with chunksize {chunksize}."
    )
    with mp.Pool(processes=processes) as pool:
        results = pool.starmap(
            callback, argument_pairs, chunksize=int(chunksize)
        )
        pool.close()
        pool.join()
    logging.info(f"Finished processing data on {processes} processes.")

    # return array of results
    return np.array(results)
