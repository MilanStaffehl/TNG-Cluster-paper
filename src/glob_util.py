"""
Global utility tools.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typedef

if TYPE_CHECKING:
    from library.config import config


def assemble_path_dict(
    milestone: str,
    cfg: config.Config,
    type_flag: str,
    virial_temperatures: bool = False,
    alt_figure_dir: str | Path | None = None,
    alt_data_dir: str | Path | None = None,
    figures_subdirectory: str | Path | None = None,
) -> typedef.FileDict:
    """
    Assemble a valid file dictionary from the given input.

    This can be used in the scripts to quickly assemble a FileDict for
    the pipelines from the information given by the user. The function
    will also verify the validity of the paths assembled and can also
    optionally warn the user if files already exist and would be
    overwritten.

    The files will have the common file pattern::

        {milestone}_{type_flag}_{simulation}_{ident_flag}.{file_extension}

    where the file extension and possible identification flags are added
    by the pipeline. The data and figure directories are taken to be the
    default directories unless alternatve directories are specified.

    :param milestone: The name of the milestone. Example: ``mass_trends``.
    :param cfg: A Config configuration class instance, set up for
        the current task.
    :param type_flag: The type flag to be inserted after the milestone.
        Required.
    :param ident_flag: The ident flag to be suffixed to the name.
        Optional.
    :param virial_temperatures: Whether to include the file stem for
        virial temperature data files.
    :param alt_figure_dir: Alternative home directory for figures. Will
        need to be verified first. If the given directory does not exist
        or does not qualify, the default figure directory will be used
        as fallback.
    :param alt_data_dir: Alternative home directory for data. Will need
        to be verified first. If the given directory does not exist or
        does not qualify, the default data directory will be used as
        fallback.
    :param figures_subdirectory: An optional subdirectory inside the
        figures home where to save figures. Must be given relative to
        the figures home directory.
    :return: A valid file path dictionary.
    """
    figure_path = cfg.figures_home / milestone / cfg.sim_path
    if figures_subdirectory:
        figure_path = figure_path / Path(figures_subdirectory)
    figure_stem = f"{milestone}_{type_flag}_{cfg.sim_path}"

    if alt_figure_dir:
        new_path = Path(alt_figure_dir)
        if new_path.exists() and new_path.is_dir():
            figure_path = new_path
        else:
            logging.warning(
                f"Given figures path is invalid: {str(new_path)}."
                f"Using fallback path {str(figure_path)} instead."
            )

    data_path = cfg.data_home / milestone
    data_stem = f"{milestone}_{type_flag}_{cfg.sim_path}"

    if alt_data_dir:
        new_path = Path(alt_data_dir)
        if new_path.exists() and new_path.is_dir():
            data_path = new_path
        else:
            logging.warning(
                f"Given data path is invalid: {str(new_path)}."
                f"Attempting fallback path {str(data_path)} instead."
            )
    # assemble dict
    file_data = {
        "figures_dir": figure_path.resolve(),
        "data_dir": data_path.resolve(),
        "figures_file_stem": figure_stem,
        "data_file_stem": data_stem,
    }
    if virial_temperatures:
        file_data.update(
            {"virial_temp_file_stem": f"virial_temperatures_{cfg.sim_path}"}
        )
    return file_data


def translate_sim_name(name: str) -> str:
    """
    Return the simulation name from the development name.

    :param name: The development handle, i.e. MAIN_SIM or DEV_SIM.
    :raises ValueError: When the given name is not a valid simulation
        shorthand.
    :return: The actual name of the simulation, i.e. TNG300-1 or TNG50-3.
    """
    if name == "TEST_SIM":
        return "TNG50-4"
    elif name == "DEV_SIM":
        return "TNG50-3"
    elif name == "MAIN_SIM":
        return "TNG300-1"
    else:
        raise ValueError(f"Unknown simulation type {name}.")
