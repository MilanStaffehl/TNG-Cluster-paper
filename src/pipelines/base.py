"""
Base class for pipelines.
"""
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import typedef
from library.config import config


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

    @abstractmethod
    def run(self) -> int:
        pass

    def _create_directories(self) -> None:
        """
        Create data directories if required.
        """
        if not hasattr(self, "to_file"):
            return
        if self.to_file:
            data_path = Path(self.paths["data_dir"])
            if not data_path.exists():
                logging.info(
                    f"Creating missing data directory {str(data_path)}."
                )
                data_path.mkdir(parents=True)

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
