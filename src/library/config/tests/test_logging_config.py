"""
Test the logging_config module.
"""
import logging
import logging.config

from library.config import logging_config


def test_get_logging_config_function():
    """
    Test that the function returns a correct logging set-up.
    """
    # standard level
    test_config = logging_config.get_logging_config("INFO")
    assert "version" in test_config.keys()
    assert "formatters" in test_config.keys()
    assert "handlers" in test_config.keys()
    assert "root" in test_config.keys()
    # assert handlers are correct
    assert "base" in test_config["handlers"].keys()
    assert len(test_config["handlers"]) == 1  # only one handler
    # given level should be present
    assert test_config["root"]["level"] == "INFO"
    assert test_config["handlers"]["base"]["level"] == "INFO"
    # custom level
    test_config = logging_config.get_logging_config("TRACE")
    assert test_config["root"]["level"] == "TRACE"
    assert test_config["handlers"]["base"]["level"] == "TRACE"


def test_logging_config(mocker):
    """
    Test that when used in logging, the config works as intended.
    """
    test_config = logging_config.get_logging_config("WARNING")
    logging.config.dictConfig(test_config)
    # test root has expected handlers and level
    root_logger = logging.getLogger("root")
    assert root_logger.level == 30
    assert len(root_logger.handlers) == 1
    assert root_logger.handlers[0].name == "base"
    assert root_logger.handlers[0].level == 30  # numeric WARNING
    # test that only messages of the expected level are emitted
    mock_stdout = mocker.patch("sys.stdout.write")
    logging.info("I should not show up.")
    mock_stdout.assert_not_called()
    logging.warning("I am a warning!")
    mock_stdout.assert_called_with("WARNING: I am a warning!\n")


def test_change_level(mocker):
    """
    Test the function to change the logging level.
    """
    test_config = logging_config.get_logging_config("WARNING")
    logging.config.dictConfig(test_config)
    mock_stdout = mocker.patch("sys.stdout.write")
    # log a message that should not show up
    logging.info("I should not show up.")
    mock_stdout.assert_not_called()
    # change the logging level
    logging_config.change_level("INFO")
    # check levels
    root_logger = logging.getLogger("root")
    assert root_logger.level == 20
    assert root_logger.handlers[0].level == 20
    # check log emission
    logging.info("I am an information.")
    mock_stdout.assert_called_with("INFO: I am an information.\n")
