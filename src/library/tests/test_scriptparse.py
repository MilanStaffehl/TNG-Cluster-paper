"""Tests for the custom argparser"""
from pathlib import Path

import pytest

from library import scriptparse
from library.config import config


@pytest.fixture
def base_parser():
    """Yield a base script parser"""
    yield scriptparse.BaseScriptParser(
        prog="Test parsing",
        description="A test parser with no actual job",
    )


def test_base_parser_basics(base_parser):
    """Test that the base parser functions as expected"""
    # test config
    assert base_parser.description == "A test parser with no actual job"
    assert base_parser.prog == "Test parsing"
    # test that the parser has no positional args
    args = base_parser.parse_args([])
    assert hasattr(args, "sim")
    assert hasattr(args, "processes")
    assert hasattr(args, "to_file")
    assert hasattr(args, "from_file")
    assert hasattr(args, "no_plots")
    assert hasattr(args, "figurespath")
    assert hasattr(args, "datapath")
    # assert the default values are set correctly
    assert args.sim == "TNG300-1"
    assert args.processes == 0
    assert args.to_file is False
    assert args.from_file is False
    assert args.no_plots is False
    assert args.figurespath is None
    assert args.datapath is None
    # changes introduced later: quiet is no longer a boolean
    assert hasattr(args, "quiet")
    assert args.quiet == 0
    assert hasattr(args, "verbosity")
    assert args.verbosity == 0


def test_base_parser_verbosity(base_parser):
    """Change the behavior when setting verbosity flags"""
    # test stacking of -v flag
    for level, flag in enumerate(["-v", "-vv", "-vvv"]):
        args = base_parser.parse_args([flag])
        assert args.verbosity == level + 1
        assert args.quiet == 0
    # test stacking of -q flag
    for level, flag in enumerate(["-q", "-qq", "-qqq"]):
        args = base_parser.parse_args([flag])
        assert args.quiet == level + 1
        assert args.verbosity == 0
    # test that args are mutually exclusive
    with pytest.raises(SystemExit):
        base_parser.parse_args(["-v", "-q"])


def test_base_parser_add_argument(base_parser):
    """Test that arguments can be added to the parser normally"""
    base_parser.add_argument(
        "-t",
        "--testing",
        dest="testing",
        type=int,
    )
    # try with fully qualified name
    args = base_parser.parse_args(["--testing", "12"])
    assert hasattr(args, "testing")
    assert isinstance(args.testing, int)
    assert args.testing == 12
    # try with short name
    args = base_parser.parse_args(["-t", "12"])
    assert hasattr(args, "testing")
    assert args.testing == 12


def test_base_parser_remove_args(base_parser, capsys):
    """Test the remove_argument method of the parser"""
    # add a test argument
    parser = base_parser
    parser.add_argument(
        "-t",
        "--testing",
        dest="testing",
        type=int,
    )
    # attempt to remove the argument
    parser.remove_argument("testing")

    # check the usage and help message
    parser.print_help()
    captured = capsys.readouterr()
    testing_help = "-t TESTING, --testing TESTING"
    assert testing_help not in captured.out
    parser.print_usage()
    captured = capsys.readouterr()
    testing_usage = "[-t TESTING]"
    assert testing_usage not in captured.out
    # parse and verify the argument is not in the namespace
    args = parser.parse_args([])
    assert not hasattr(args, "testing")

    # for completeness: test that the argument can still be parsed (this
    # is expected, albeit undesired behavior)
    args = parser.parse_args(["--testing", "12"])
    assert hasattr(args, "testing")
    assert args.testing == 12


def test_base_parser_allowed_sims(capsys):
    """Test that the list of allowed sims can be set"""
    parser = scriptparse.BaseScriptParser(
        description="A testing parser",
        prog="Test parser",
        allowed_sims=["SIM_A", "SIM_B", "SIMBA"],
    )

    # check that the help and usage text are updated accordingly
    parser.print_help()
    captured = capsys.readouterr()
    testing_help = "Supported simulations: SIM_A, SIM_B, SIMBA"
    assert testing_help in captured.out
    parser.print_usage()
    captured = capsys.readouterr()
    testing_usage = "[-s SIM]"
    assert testing_usage in captured.out

    # check that the argument parser parses the names correctly
    for sim in ["SIM_A", "SIM_B", "SIMBA"]:
        args = parser.parse_args(["--sim", sim])
        assert args.sim == sim


@pytest.fixture
def mock_config():
    """A mock config object"""
    yield config.Config(
        "TNG300-1",
        str(Path().home() / ".local"),
        99,
        "Group_M_Crit200",
        "Group_R_Crit200",
        Path().home() / ".local" / "data",
        Path().home() / ".local" / "figures",
    )


def test_assemble_path_dict(mock_config):
    """Test the function to assemble a path dictionary"""
    output = scriptparse._assemble_path_dict(
        milestone="test_milestone",
        cfg=mock_config,
        type_flag="type_flag",
    )
    # assert validity of dict
    assert "figures_dir" in output.keys()
    assert "data_dir" in output.keys()
    assert "figures_file_stem" in output.keys()
    assert "data_file_stem" in output.keys()
    # assert validity of values
    fig_path = (
        Path().home() / ".local" / "figures" / "test_milestone" / "TNG300_1"
    ).resolve()
    assert output["figures_dir"] == fig_path
    data_path = (Path().home() / ".local" / "data"
                 / "test_milestone").resolve()
    assert output["data_dir"] == data_path
    stem = "test_milestone_type_flag_TNG300_1"
    assert output["figures_file_stem"] == stem
    assert output["data_file_stem"] == stem


def test_assemble_path_dict_custom_dirs(mock_config):
    """Test that custom data and figure homes are respected"""
    # paths must actually exist!
    output = scriptparse._assemble_path_dict(
        milestone="test_milestone",
        cfg=mock_config,
        type_flag="type_flag",
        alt_data_dir=Path().home(),
        alt_figure_dir=Path().home(),
    )
    fig_path = Path().home().resolve()
    assert output["figures_dir"] == fig_path
    data_path = Path().home().resolve()
    assert output["data_dir"] == data_path


def test_assemble_path_dict_invalid_custom_dir(mock_config, caplog):
    """Test that only valid paths are accepted"""
    output = scriptparse._assemble_path_dict(
        milestone="test_milestone",
        cfg=mock_config,
        type_flag="type_flag",
        alt_figure_dir=Path("i/do/not/exist"),
        alt_data_dir=Path("and/neither/do/i"),
    )
    fig_path = mock_config.figures_home / "test_milestone" / "TNG300_1"
    assert output["figures_dir"] == fig_path.resolve()
    expected_msg = (
        f"Given figures path is invalid: i/do/not/exist. Using fallback path"
        f" {str(fig_path)} instead."
    )
    assert expected_msg in caplog.records[0].message
    data_path = mock_config.data_home / "test_milestone"
    assert output["data_dir"] == data_path.resolve()
    expected_msg = (
        f"Given data path is invalid: and/neither/do/i. Attempting fallback "
        f"path {str(data_path)} instead."
    )
    assert expected_msg in caplog.records[1].message


def test_assemble_path_dict_virial_temperature(mock_config):
    """Test that the virial temperature stem can be included"""
    output = scriptparse._assemble_path_dict(
        milestone="test_milestone",
        cfg=mock_config,
        type_flag="type_flag",
        virial_temperatures=True
    )
    assert "virial_temp_file_stem" in output.keys()
    assert output["virial_temp_file_stem"] == "virial_temperatures_TNG300_1"


def test_assemble_path_dict_subdirs(mock_config):
    """Test that subdirectories may be specified"""
    output = scriptparse._assemble_path_dict(
        milestone="test_milestone",
        cfg=mock_config,
        type_flag="type_flag",
        figures_subdirectory="./fig_subdir",
        data_subdirectory="./data_subdir",
    )
    fig_path = (
        mock_config.figures_home.resolve() / "test_milestone" / "TNG300_1"
    )
    assert output["figures_dir"] == fig_path / "fig_subdir"
    data_path = mock_config.data_home.resolve() / "test_milestone"
    assert output["data_dir"] == data_path / "data_subdir"


# TODO: test the parse_namespace function
# TODO: test the startup function for correct logging setup
