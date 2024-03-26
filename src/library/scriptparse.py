"""Parser base class for scripts"""
from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from pathlib import Path
from typing import TypeAlias

import typedef
from library.config import config, logging_config

# type def
PipelineKwargs: TypeAlias = dict[str, bool | str | int]


class BaseScriptParser(argparse.ArgumentParser):

    def __init__(
        self,
        allowed_sims=("TNG300", "TNG100", "TNG50"),
        prog=None,
        usage=None,
        description=None,
        epilog=None,
    ):
        """
        :param allowed_sims: List of the names of simulations that the
            script can run on. If there is no restriction, leave it as
            None. This list will merely be used for the help text of
            the ``--sim`` argument, so it does not need to match the
            list of simulations in the config file.
        :param prog: Command to execute the program.
        :param usage: Usage description.
        :param description: Description of what the program does.
        :param epilog: Epilog to print after usage.
        """
        super().__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
        )
        # set up default args
        if allowed_sims is None:
            add_str = ""
        else:
            add_str = (
                f"Supported simulations: {', '.join(allowed_sims)} (you "
                f"might need to specify resolution for these simulations)"
            )
        self.add_argument(
            "-s",
            "--sim",
            help=(
                f"Name of the simulation to use. Must match one of the names "
                f"given in the configuration file (config.yaml). {add_str}."
            ),
            dest="sim",
            type=str,
            default="TNG300-1",
        )
        self.add_argument(
            "-p",
            "--processes",
            help=(
                "Use multiprocessing, with the specified number of processes."
            ),
            type=int,
            default=0,
            dest="processes",
            metavar="NUMBER",
        )
        self.add_argument(
            "-f",
            "--to-file",
            help=(
                "When set, the data for the figures will be written to file. "
                "If not set, no data (including auxiliary data) will be "
                "written to file. Figures are not affected by this switch. "
                "To suppress figure creation, use the -x flag."
            ),
            dest="to_file",
            action="store_true",
        )
        self.add_argument(
            "-l",
            "--load-data",
            help=(
                "When set, data is loaded from data files rather than newly "
                "acquired from the simulation data. This only works if data "
                "files of the expected name are present. When used, the flags "
                "-p, -f, -q have no effect. Some other flags might also have "
                "no effect, in which case it is mentioned in their description."
            ),
            dest="from_file",
            action="store_true",
        )
        self.add_argument(
            "-x",
            "--no-plots",
            help=(
                "Suppresses creation of plots. Useful to prevent overwriting "
                "of existing figure files."
            ),
            dest="no_plots",
            action="store_true",
        )
        self.add_argument(
            "--ext",
            help="File extension for the plot files. Defaults to pdf.",
            dest="fig_ext",
            type=str,
            default="pdf",
            choices=["pdf", "png"]
        )
        exclusion_group = self.add_mutually_exclusive_group(required=False)
        exclusion_group.add_argument(
            "-v",
            help=(
                "Make the output more verbose. Stackable. Determines the log "
                "level and whether real-time updates are sent to stdout. If "
                "not set, the logging level is set to INFO. Setting -v means "
                "log level MEMORY for diagnostics, -vv means real-time status "
                "updates in loops are logged (not recommended when piping "
                "stdout to file!), and -vvv means log level DEBUG."
            ),
            dest="verbosity",
            action="count",
            default=0,
        )
        exclusion_group.add_argument(
            "-q",
            help=(
                "Reduce the verbosity of the script. Stackable. Corresponds to "
                "raising the logging level. Setting -q means log level "
                "WARNING, -qq means log level ERROR, and -qqq means log level "
                "CRITICAL."
            ),
            dest="quiet",
            action="count",
            default=0,
        )
        self.add_argument(
            "--figures-dir",
            help=(
                "The directory path under which to save the figures, if created. "
                "Directories that do not exist will be recursively created. "
                "It is recommended to leave this at the default value."
            ),
            dest="figurespath",
            default=None,
            metavar="DIRECTORY",
        )
        self.add_argument(
            "--data-dir",
            help=(
                "The directory path under which to save data files, if "
                "created. Directories that do not exist will be recursively "
                "created. When using --load-data, this directory is instead "
                "searched for data files to load data from. It is recommended "
                "to leave this at the default value unless the expected data "
                "has been saved somewhere else and needs to be loaded from "
                "there."
            ),
            dest="datapath",
            default=None,
            metavar="DIRECTORY",
        )

    def remove_argument(self, arg):
        """
        Remove argument from the parser.

        .. note:: The argument can still be parsed, i.e. using it will
            not raise a SystemExit, but will quietly add the argument
            to the namespace anyway. This is acceptable though, as the
            script using the parser will simply discard all unwanted
            namespace attributes unused. The main point is that the
            argument does not appear in the help and usage texts.

        :param arg: The name of the argument, without leading dashes.
            For arguments with multiple names, use the destination name.
        :return: None
        """
        for action in self._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                self._remove_action(action)
                break

        for action in self._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return


def startup(
    namespace: argparse.Namespace,
    milestone: str,
    type_flag: str,
    with_virial_temperatures: bool = False,
    figures_subdirectory: str | Path | None = None,
    data_subdirectory: str | Path | None = None,
    suppress_sim_name_in_files: bool = False,
) -> PipelineKwargs:
    """
    Common set-up for scripts.

    Function sets up logging according to the received verbosity, and
    creates a base kwargs dictionary for pipelines, to be amended by
    script-specific keyword arguments.

    The logging setup also includes the addition of the custom logging
    level MEMORY with a numeric value of 18 for memory monitoring.

    :param namespace: The namespace returned by the parser. Can be
        given as-is, and will not be altered.
    :param milestone: The name of the milestone. Example: ``mass_trends``.
    :param type_flag: The type flag to be inserted after the milestone.
    :param with_virial_temperatures: Whether to include the file stem for
        virial temperature data files in the paths dictionary.
    :param figures_subdirectory: An optional subdirectory inside the
        figures home where to save figures. Must be given relative to
        the figures home directory.
    :param data_subdirectory: An optional subdirectory inside the
        data home where to save data files. Must be given relative to
        the data home directory.
    :param suppress_sim_name_in_files: When set to True, the figure and
        data file names will not contain the name of the simulation
        that is set in the namespace ``sim`` field. Otherwise, the
        name of these files will contain the simulation name given in
        the namespace object.
    :return: A dictionary of keyword arguments suitable to start up a
        base :class:`~library.pipelines.base.Pipeline`, by using it
        with the ``**`` operator as init args for a pipeline. This
        dictionary can be updated with additional information required
        for other subclasses of pipelines afterward.
    """
    # set up logging
    log_level = parse_verbosity(namespace)
    log_config = logging_config.get_logging_config(log_level)
    logging.config.dictConfig(log_config)
    logging.addLevelName(18, "MEMORY")  # custom level
    # parse namespace for initial kwargs dictionary for pipelines
    return parse_namespace(
        namespace,
        milestone,
        type_flag,
        with_virial_temperatures,
        figures_subdirectory,
        data_subdirectory,
        suppress_sim_name_in_files,
    )


def parse_verbosity(namespace: argparse.Namespace) -> int:
    """
    Translate the verbosity information into a log level.

    :param namespace: The script parser namespace.
    :return: The logging level determined from the verbosity args.
    """
    if namespace.verbosity >= 3:
        return 10
    elif namespace.verbosity == 2:
        return 15
    elif namespace.verbosity == 1:
        return 18
    elif namespace.quiet == 1:
        return 30
    elif namespace.quiet == 2:
        return 40
    elif namespace.quiet >= 3:
        return 50
    else:
        return 20


def parse_namespace(
    namespace: argparse.Namespace,
    milestone: str,
    type_flag: str,
    with_virial_temperatures: bool = False,
    figures_subdirectory: str | Path | None = None,
    data_subdirectory: str | Path | None = None,
    no_sim_name: bool = False,
) -> PipelineKwargs:
    """
    Parse the namespace of a script base parser for base arguments.

    Return a dictionary containing field-value pairs capable of acting
    as the init-parameters for a base pipeline, i.e. a dictionary with
    keys 'config', 'paths', 'processes', 'to_file', 'no_plots',
    'fig_ext', all with corresponding values taken from the namespace.
    Where the namespace did not supply values, sensible defaults are
    applied, assuming that the corresponding pipeline field will not be
    used and the namespace therefore purposefully did not include them.

    The function also constructs a default config object and a paths
    dictionary as required by the pipeline and possibly the script itself.

    :param namespace: The namespace returned by the parser. Can be
        given as-is, and will not be altered.
    :param milestone: The name of the milestone. Example: ``mass_trends``.
    :param type_flag: The type flag to be inserted after the milestone.
    :param with_virial_temperatures: Whether to include the file stem for
        virial temperature data files in the paths dictionary.
    :param figures_subdirectory: An optional subdirectory inside the
        figures home where to save figures. Must be given relative to
        the figures home directory.
    :param data_subdirectory: An optional subdirectory inside the
        data home where to save data files. Must be given relative to
        the data home directory.
    :param no_sim_name: When set to True, data and figure files will
        not contain the name of the simulation set in the config.
        Useful for runs with mixed simulations (e.g. when running any
        task on clusters from both TNG300 and TNG-Cluster).
    :return: A dictionary of keyword arguments suitable to start up a
        base :class:`~library.pipelines.base.Pipeline`, by using it
        with the ``**`` operator as init args for a pipeline. This
        dictionary can be updated with additional information required
        for other subclasses of pipelines afterward.
    """
    if not hasattr(namespace, "sim"):
        namespace.sim = "TNG300-1"  # throwaway name to get valid config

    # get a config object
    try:
        cfg = config.get_default_config(namespace.sim)
    except config.InvalidSimulationNameError:
        logging.fatal(f"Unsupported simulation: {namespace.sim}")
        sys.exit(1)

    # get a paths dictionary
    paths = _assemble_path_dict(
        milestone,
        cfg,
        type_flag,
        with_virial_temperatures,
        namespace.figurespath,
        namespace.datapath,
        figures_subdirectory,
        data_subdirectory,
        no_sim_name,
    )

    # assemble rst of the dict
    kwargs = {"config": cfg, "paths": paths}

    # set defaults for other base arguments
    defaults = {
        "processes": 0, "to_file": False, "no_plots": False, "fig_ext": "pdf"
    }
    # apply defaults if not explicitly set to arrive at a full valid kwargs
    # dictionary that can be used to initialise a base pipeline
    for field, default_value in defaults.items():
        if not hasattr(namespace, field):
            logging.debug(
                f"Filling missing argument '{field}' with default value "
                f"{default_value}."
            )
            kwargs.update({field: default_value})
        else:
            # use existing value
            kwargs.update({field: getattr(namespace, field)})

    return kwargs


def _assemble_path_dict(
    milestone: str,
    cfg: config.Config,
    type_flag: str,
    virial_temperatures: bool = False,
    alt_figure_dir: str | Path | None = None,
    alt_data_dir: str | Path | None = None,
    figures_subdirectory: str | Path | None = None,
    data_subdirectory: str | Path | None = None,
    suppress_sim_path_in_names: bool = False
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
    default directories unless alternative directories are specified.

    When the argument ``suppress_sim_path_in_names`` is set, the name of
    the simulation will be removed from this pattern.

    :param milestone: The name of the milestone. Example: ``mass_trends``.
    :param cfg: A Config configuration class instance, set up for
        the current task.
    :param type_flag: The type flag to be inserted after the milestone.
        Required.
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
    :param data_subdirectory: An optional subdirectory inside the
        data home where to save data files. Must be given relative to
        the data home directory.
    :param suppress_sim_path_in_names: Whether to exclude the simulation
        name from data and figure file stems. This si useful when
        combining multiple simulations, to avoid the files being named
        after a simulation they do not actually belong to.
    :return: A valid file path dictionary.
    """
    figure_path = cfg.figures_home / milestone / cfg.sim_path
    data_path = cfg.data_home / milestone
    if not suppress_sim_path_in_names:
        file_stem = f"{milestone}_{type_flag}_{cfg.sim_path}"
    else:
        file_stem = f"{milestone}_{type_flag}"

    if figures_subdirectory:
        figure_path = figure_path / Path(figures_subdirectory)

    if alt_figure_dir:
        new_path = Path(alt_figure_dir)
        if new_path.exists() and new_path.is_dir():
            figure_path = new_path
        else:
            logging.warning(
                f"Given figures path is invalid: {str(new_path)}. "
                f"Using fallback path {str(figure_path)} instead."
            )

    if data_subdirectory:
        data_path = data_path / Path(data_subdirectory)

    if alt_data_dir:
        new_path = Path(alt_data_dir)
        if new_path.exists() and new_path.is_dir():
            data_path = new_path
        else:
            logging.warning(
                f"Given data path is invalid: {str(new_path)}. "
                f"Attempting fallback path {str(data_path)} instead."
            )
    # assemble dict
    file_data = {
        "figures_dir": figure_path.resolve(),
        "data_dir": data_path.resolve(),
        "figures_file_stem": file_stem,
        "data_file_stem": file_stem,
    }
    if virial_temperatures:
        file_data.update(
            {"virial_temp_file_stem": f"virial_temperatures_{cfg.sim_path}"}
        )
    return file_data
