import logging


def setup_logging(verbose: int = 3):
    """
    Setup logging to print to stdout with default logging level being INFO.
    """
    if verbose == 1:
        level = logging.CRITICAL
    elif verbose == 2:
        level = logging.ERROR
    elif verbose == 3:
        level = logging.INFO
    elif verbose == 4:
        level = logging.DEBUG
    else:
        raise ValueError("Maximum verbosity is -vvvv (verbose=4)")

    log = logging.getLogger(__name__)
    console = logging.StreamHandler()
    log.addHandler(console)
    log.setLevel(level)

    return log