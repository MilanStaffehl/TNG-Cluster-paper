import logging


def get_logging_config(logging_level: str | int) -> None:
    """
    Return a simple, Jupyter notebook compatible logging config.

    :param logging_level: logging level to use
    :return: logging config for use with ``dictConfig`` of logging module
    """
    logging_config = {
        "version": 1,
        "formatters": {
            "stdout": {
                "format": "%(levelname)s: %(message)s"
            }
        },
        "handlers": {
            "base": {
                "class": "logging.StreamHandler",
                "level": logging_level,
                "formatter": "stdout",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": logging_level,
            "handlers": ["base"]
        },
    }
    return logging_config


def change_level(logging_level: str | int) -> None:
    """
    Change the logging level of the logging environment.

    Function finds the root logger and set sits level to the specified
    logging level. It then goes through all handlers of the root logger
    and sets their level to the specified logging level as well.

    :param logging_level: logging level, either as name or as integer
    """
    root_logger = logging.getLogger("root")
    root_logger.setLevel(logging_level)
    for handler in root_logger.handlers:
        handler.setLevel(logging_level)
