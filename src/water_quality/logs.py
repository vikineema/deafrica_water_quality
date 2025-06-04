import logging


def setup_logging(verbose: int = 4):
    """
    Setup logging to print to stdout with configurable verbosity.
    """
    if verbose == 1:
        level = logging.CRITICAL
    elif verbose == 2:
        level = logging.ERROR
    elif verbose == 3:
        level = logging.WARNING
    elif verbose == 4:
        level = logging.INFO
    elif verbose == 5:
        level = logging.DEBUG
    else:
        raise ValueError("Maximum verbosity is -vvvvv (verbose=5)")

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    return logger
