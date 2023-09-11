"""
Custom type definitions.
"""
from pathlib import Path
from typing import TypedDict


# custom dict type for Pipelines
class FileDict(TypedDict):
    """
    Dictionary type for pipeline paths.

    :param figures_dir: The full path to the directory where figures
        are saved. Figures will be saved directly under this path.
    :param data_dir: The full path to the directory where the data for
        the plots is saved. Data files will be saved directly under this
        path. When loading data from file, this directory will be queried
        for data files.
    :param figures_file_stem: The stem of the file name for figures, i.e.
        the name of the file without the file extension. This name might
        be extended by additional qualifiers (e.g. indices etc.), so it
        might not be equivalent to the final file stem.
    :param data_file_stem: The stem of the file name for the plot data,
        i.e. the name of the file name without file extension. The data
        will be saved directly under this name. When loading data from
        file, this name will be used to find the data file.
    :param virial_temp_file_stem: The stem of the file name for the
        virial temperatures, i.e. the file name without file extension.
        The virial temperatures will be saved directly under this name.
        When loading data from  file, this name will be used to find the
        virial temperatures data file.
    """
    figures_dir: str | Path
    data_dir: str | Path
    figures_file_stem: str
    data_file_stem: str
    virial_temp_file_stem: str
