def check_config(analysis_config: dict) -> dict:
    """Validates the provided analysis configuration dictionary.

    This function ensures that the `analysis_config` is not None and
    contains all the necessary parameters required for the analysis.
    If any essential parameters are missing or the config itself is
    empty, a ValueError is raised.

    Parameters
    ----------
    analysis_config : dict
        A dictionary containing the configuration parameters for the
        analysis.

    Returns
    -------
    dict
        The validated `analysis_config` dictionary if all checks pass.
    """

    if analysis_config is None:
        raise ValueError(
            "Please provide a config for the analysis parameters in "
            "yaml format, file or text"
        )

    config_items = [
        "resolution",
        "instruments_to_use",
        "water_frequency_threshold_high",
        "water_frequency_threshold_low",
        "permanent_water_threshold",
        "sigma_coefficient",
    ]
    missing_parameters = []
    for k in config_items:
        if k not in list(analysis_config.keys()):
            missing_parameters.append(k)
    if missing_parameters:
        raise ValueError(
            f"The following analysis parameters not found {', '.join(missing_parameters)} "
        )

    return analysis_config
