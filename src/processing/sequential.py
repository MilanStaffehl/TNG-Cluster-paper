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
) -> NDArray:
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
    :return: An array of shape (N, D) where N is the number of halos and
        D is ``data_shape``. Contains the result of ``callback`` for
        every halo.
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


def process_halo_data_multiargs(
    callback: Callable[..., NDArray],
    data_shape: tuple[int],
    *input_args: NDArray,
    quiet: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> NDArray:
    """
    Process halo data sequentially for callbacks with multiple arguments.

    This method is meant for callables that take more than one variable
    argument.

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

    would result in the following parallelized function calls:

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
    :param data_shape: The shape of the NDArray that ``callback`` returns.
    :param *input_args: One-dimensional NDArrays of the same length. Must
        be as many as ``callback`` takes positional arguments.
    :param quiet: Whether to suppress status information output. Keyword
        only arg, defaults to False.
    :kwargs: Optional keyworded arguments for ``callback`` that remain
        static across different calls.
    :return: Array of the return values of ``callback`` for every column
        of applying ``input_args`` to ``callback``.
    """
    if kwargs is None:
        kwargs = {}
    # verify input data
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

    logging.info("Start processing halo data sequentially.")
    data = np.zeros((length, ) + data_shape)
    input_data = np.array(input_args).transpose()
    for i in range(length):
        if not quiet:
            perc = i / length * 100
            print(f"Processing halo {i}/{length} ({perc:.1f}%)", end="\r")
        data[i] = callback(*input_data[i], **kwargs)
    logging.info("Finished processing halo data.")
    return data
