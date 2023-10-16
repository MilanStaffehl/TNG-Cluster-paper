"""
Base class for pipelines.
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import typedef
from config import config


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
