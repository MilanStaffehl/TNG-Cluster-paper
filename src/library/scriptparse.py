"""Parser base class for scripts"""
import argparse


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
            "-q",
            "--quiet",
            help=(
                "Prevent progress information to be emitted. Has no effect when "
                "multiprocessing is used."
            ),
            dest="quiet",
            action="store_true",
        )
        self.add_argument(
            "--ext",
            help="File extension for the plot files. Defaults to pdf.",
            dest="extension",
            type=str,
            default="pdf",
            choices=["pdf", "png"]
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
