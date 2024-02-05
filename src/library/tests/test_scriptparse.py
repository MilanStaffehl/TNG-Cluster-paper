"""Tests for the custom argparser"""
import pytest

from library import scriptparse


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
    assert hasattr(args, "quiet")
    assert hasattr(args, "figurespath")
    assert hasattr(args, "datapath")
    # assert the default values are set correctly
    assert args.sim == "MAIN_SIM"
    assert args.processes == 0
    assert args.to_file is False
    assert args.from_file is False
    assert args.no_plots is False
    assert args.quiet is False
    assert args.figurespath is None
    assert args.datapath is None


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
    testing_help = "-s {SIM_A,SIM_B,SIMBA}, --sim {SIM_A,SIM_B,SIMBA}"
    assert testing_help in captured.out
    parser.print_usage()
    captured = capsys.readouterr()
    testing_usage = "[-s {SIM_A,SIM_B,SIMBA}]"
    assert testing_usage in captured.out

    # check that the argument parser parses the names correctly
    for sim in ["SIM_A", "SIM_B", "SIMBA"]:
        args = parser.parse_args(["--sim", sim])
        assert args.sim == sim
    # check that these are the only valid options
    with pytest.raises(SystemExit):
        parser.parse_args(["--sim", "NOT_SUPPORTED"])
