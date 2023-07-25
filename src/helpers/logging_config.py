def get_logging_config(logging_level):
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
