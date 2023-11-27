"""
Base class for pipelines.
"""
import logging
import logging.config
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import typedef
from library.config import config, logging_config


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

    def __post_init__(self):
        # set up logging
        log = logging_config.get_logging_config("INFO")
        logging.config.dictConfig(log)

    @abstractmethod
    def run(self) -> int:
        pass

    def _create_directories(
        self, subdirs: Sequence[str | Path] | None = None
    ) -> None:
        """
        Create data directories if required.

        :param subdirs: Additional dubdirectories of the data directory
            to check and create if non-existent.
        """
        if not hasattr(self, "to_file"):
            return
        if subdirs is None:
            subdirs = []
        if self.to_file:
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
        # files
        if "data_file_stem" in self.paths.keys():
            data_dir = self.paths["data_dir"]
            data_file = data_dir / f"{self.paths['data_file_stem']}.npz"
            if not data_file.exists() or not data_file.is_file():
                logging.error(f"Data file under {data_dir} does not exist.")
                return 3
        else:
            logging.error(
                f"The FileDict received does not have a data file stem "
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
                return 3
        return 0

    def _timeit(self, start_time: float, step_description: str) -> float:
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
