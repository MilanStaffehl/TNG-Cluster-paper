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


def process_data_sequentially(
    callback: Callable[..., NDArray],
    data: Sequence[int],
    data_shape: tuple[int],
    kwargs: dict[str, Any] | None = None,
    quiet: bool = False,
) -> NDArray:
    """
    Process data sequentially.

    This method calls the given Callable ``callback`` with every entry
    in ``data`` in order of appearance. Optionally, keyworded arguments
    can be supplied to the Callable. These keyworded arguments will be
    the same across calls, while the positional argument will be from
    the ``data`` array and thus vary per call. The results are bundled
    into one array and returned.

    To achieve a function that will take a single data point and return
    processed data, it is advised to create a wrapper function around
    the desired DAQ functions.

    :param callback: A function to process halo data. Its signature is
        arbitrary, but it must take as first positional argument a halo
        ID. Further arguments may be supplied by ``kwargs``, but no
        verification is performed to ensure all arguments are matched.
        Callable must return the processed temperatures, NOT the values
        of the temperature!
    :param data: Sequence of data points to hand to the callback.
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
    n_points = len(data)
    results = np.zeros((n_points, ) + data_shape)
    for i, data_point in enumerate(data):
        if not quiet:
            perc = i / n_points * 100
            print(f"Processing entry {i}/{n_points} ({perc:.1f}%)", end="\r")
        results[i] = callback(data_point, **kwargs)
    logging.info(f"Finished processing data.{' ' * 10}")
    return results


def process_data_multiargs(
    callback: Callable[..., NDArray],
    data_shape: tuple[int],
    *input_args: NDArray,
    quiet: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> NDArray | None:
    """
    Process data sequentially for callbacks with multiple arguments.

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
    :param input_args: One-dimensional NDArrays of the same length. Must
        be as many as ``callback`` takes positional arguments.
    :param quiet: Whether to suppress status information output. Keyword
        only arg, defaults to False.
    :param kwargs: Optional keyworded arguments for ``callback`` that remain
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
    results = np.zeros((length, ) + data_shape)
    input_data = np.array(input_args).transpose()
    for i in range(length):
        if not quiet:
            perc = i / length * 100
            print(f"Processing entry {i}/{length} ({perc:.1f}%)", end="\r")
        results[i] = callback(*input_data[i], **kwargs)
    logging.info(f"Finished processing data.{' ' * 10}")
    return results
