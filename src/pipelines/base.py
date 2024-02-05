"""
Base classes for pipelines.
"""
from __future__ import annotations

import logging
import logging.config
import time
import tracemalloc
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from matplotlib import pyplot as plt

import typedef
from library.config import logging_config

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    # Reason for noqa: https://github.com/PyCQA/pyflakes/issues/648
    from library.config import config  # noqa: F401


@dataclass
class Pipeline:
    """
    Base class for pipelines.

    Base class contains fields for a config class instance and a file
    paths dictionary. It also holds the number of processes for
    multiprocessing.

    The class defines one public method that must be implemented by all
    subclasses: ``run``, which executes the pipeline and returns a status
    code.
    """

    config: config.Config
    paths: typedef.FileDict | typedef.FileDictVT
    processes: int
    quiet: bool
    to_file: bool
    no_plots: bool
    fig_ext: Literal["pdf", "svg", "png", "jpeg", "jpg", "tif", "esp", "ps"]

    def __post_init__(self):
        # set up logging
        log = logging_config.get_logging_config("INFO")
        logging.config.dictConfig(log)

    @abstractmethod
    def run(self) -> int:
        pass

    def _create_directories(
        self,
        subdirs: Sequence[str | Path] | None = None,
        force: bool = False,
    ) -> None:
        """
        Create data directories if required.

        :param subdirs: Additional dubdirectories of the data directory
            to check and create if non-existent.
        :param force: Whether to force the creation of missing directories,
            even if ``self.to_file`` is False.
        """
        if not hasattr(self, "to_file"):
            return
        if subdirs is None:
            subdirs = []
        if self.to_file or force:
            data_path = Path(self.paths["data_dir"])
            if not data_path.exists():
                logging.info(
                    f"Creating missing data directory {str(data_path)}."
                )
                data_path.mkdir(parents=True)
            for subdirectory in subdirs:
                additional_path = data_path / subdirectory
                if not additional_path.exists():
                    logging.info(
                        f"Creating missing subdirectory {subdirectory}."
                    )
                    additional_path.mkdir()

    def _verify_directories(self) -> int:
        """
        Verify that the data directories exist (useful for loading).

        Also verifies that data files exist. Can also check for virial
        temperature files as well, if the paths dictionary indicates
        that they are required.

        :return: Exit code: zero signifies data directory exists, 1
            signifies the data directory does not exist, 2 signifies a
            faulty FileDict. Exit code 3 means one of the data files
            does not exist.
        """
        # directories
        if "data_dir" in self.paths.keys():
            data_dir = self.paths["data_dir"]
            if not data_dir.exists() or not data_dir.is_dir():
                logging.error(
                    f"Data directory under {data_dir} does not exist."
                )
                return 1
        else:
            logging.error(
                f"The FileDict received does not have a data directory "
                f"specified!\n{self.paths}"
            )
            return 2
        # virial temperature file
        if "virial_temp_file_stem" in self.paths.keys():
            data_dir = self.paths["data_dir"]
            data_file = data_dir / f"{self.paths['virial_temp_file_stem']}.npy"
            if not data_file.exists() or not data_file.is_file():
                logging.error(
                    f"Data file for virial temperature under {data_dir} does "
                    "not exist."
                )
                return 2
        return 0

    @staticmethod
    def _timeit(start_time: float, step_description: str) -> float:
        """
        Log the time passed since the given ``time``.

        Message will include the description of the time interval, i.e.
        what process was performed during this time.

        :param start_time: The time when the current process started in seconds
            since the epoch. Can be retrieved using ``time.time()``.
        :param step_description: Name or description of the step that
            was timed.
        :return: The current time since the epoch in seconds. Can be
            used to time the next process.
        """
        now = time.time()
        time_diff = now - start_time
        time_fmt = time.strftime('%H:%M:%S', time.gmtime(time_diff))
        logging.info(f"Spent {time_fmt} hours on {step_description}.")
        return now

    def _save_fig(
        self,
        figure: Figure,
        ident_flag: str = "",
        subdirs: Path | str | None = None,
    ) -> None:
        """
        Save the given figure to file.

        The figure is saved to file with the file extension set in the
        pipeline and under the path given in the ``paths`` attribute of
        the pipeline. If subdirectories are specified, they are added at
        the end of the path. The ``ident_flag`` is added to the end of
        the file stem given in the ``paths`` attribute of the pipeline.

        After saving, the figure is closed.

        :param figure: The matplotlib figure to save.
        :param ident_flag: The ident flag for the file name, i.e. the
            string that is appended to the end of the figure file stem
            as specified in ``self.paths``. Optional, defaults to an
            empty string (i.e. file is saved using the file stem in
            ``self.paths`` alone).
        :param subdirs: String or Path object of subdirectories, relative
            to the figures path given in ``self.paths``. Optional,
            defaults to None, which means the figure will be saved in
            the directory specified in ``self.paths``.
        :return: None
        """
        if self.no_plots:
            plt.close(figure)
            return

        # file name
        if ident_flag:
            ident_flag = f"_{ident_flag}"
        filename = (
            f"{self.paths['figures_file_stem']}{ident_flag}.{self.fig_ext}"
        )

        # file path
        filepath = Path(self.paths["figures_dir"])
        if subdirs:
            filepath = filepath / Path(subdirs)
        if not filepath.exists():
            logging.info("Creating missing figures directory.")
            filepath.mkdir(parents=True)
        figure.savefig(filepath / filename, bbox_inches="tight")
        logging.debug(f"Saved a plot to file: {str(filename)}")

        # close figure
        plt.close(figure)


class DiagnosticsPipeline(Pipeline):
    """
    A pipeline base class with additional diagnostic tools for memory usage.

    Class provides methods for logging memory usage with its own new
    logging level MEMLOG of severity 18. It also offers a handy method
    that combines timing and memory usage information.
    """

    def __post_init__(self):
        super().__post_init__()
        # define custom logging level for memory infos
        logging.addLevelName(18, "MEMORY")
        if not self.quiet:
            logging_config.change_level(18)

    @abstractmethod
    def run(self) -> int:
        pass

    def _diagnostics(
        self,
        start_time: float,
        step_description: str,
        reset_peak: bool = True,
        unit: Literal["kB", "MB", "GB"] = "GB"
    ) -> float:
        """
        Log diagnostic data.

        :param start_time: The start time of the step to diagnose in
            seconds since the epoch.
        :param step_description: A description of the step for which the
            diagnostics are logged.
        :param reset_peak: Whether to reset the peak of the traced
            memory (so that in the next step, the peak can be determined
            independently of the previous steps).
        :param unit: The unit to convert the memory into. Can be one of
            the following: kB, MB, GB. If omitted, the memory is given
            in bytes. Optional, defaults to display in gigabytes.
        :return: The time point of the diagnostic in seconds since the
            epoch.
        """
        # memory diagnostics
        mem = tracemalloc.get_traced_memory()
        self._memlog(
            f"Peak memory usage during {step_description}", mem[1], unit
        )
        self._memlog(
            f"Current memory usage after {step_description}", mem[0], unit
        )
        if reset_peak:
            tracemalloc.reset_peak()
        # runtime diagnostics
        return self._timeit(start_time, step_description)

    @staticmethod
    def _memlog(
        message: str,
        memory_used: float,
        unit: Literal["kB", "MB", "GB"] = "GB"
    ) -> None:
        """
        Helper function; logs memory usage message if set to verbose.

        The function will print the given message, followed by a colon
        and the given memory used, converted into the given unit.

        :param message: The message to log before the converted memory.
        :param memory_used: The memory currently used in units of bytes.
        :param unit: The unit to convert the memory into. Can be one of
            the following: kB, MB, GB. If omitted, the memory is given
            in bytes. Defaults to display in gigabytes.
        :return: None
        """
        match unit:
            case "kB":
                memory = memory_used / 1024.
            case "MB":
                memory = memory_used / 1024. / 1024.
            case "GB":
                memory = memory_used / 1024. / 1024. / 1024.
            case _:
                memory = memory_used
                unit = "Bytes"  # assume the unit is bytes
        logging.log(18, f"{message}: {memory:,.4} {unit}.")
