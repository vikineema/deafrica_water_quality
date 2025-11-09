"""
This module provides functions to implement water detection using DE
Africa Water Observations from Space (WOfS) products.
"""

import logging

import xarray as xr

log = logging.getLogger(__name__)


def water_analysis(
    ds: xr.Dataset,
    water_frequency_threshold: float = 0.5,
    wofs_varname: str = "wofs_ann_freq",
    permanent_water_threshold: float = 0.875,
    sigma_coefficient: float = 1.2,
):
    """Performs water detection analysis on DE Africa WOfS Annual
    Summary data.

    Parameters
    ----------
    ds : Dataset
        An xarray Dataset containing WOfS annual frequency data.
    water_frequency_threshold : float, optional
        The frequency threshold above which a pixel is classified as
        general water, by default 0.5
    wofs_varname : str, optional
        The name of the WOfS variable to use for analysis,
        by default "wofs_ann_freq"
    permanent_water_threshold : float, optional
        The base frequency threshold for identifying permanent water,
        by default 0.875
    sigma_coefficient : float, optional
        A coefficient used to adjust the permanent water threshold
        based on the standard deviation of the annual frequency,
        by default 1.2

    Returns
    -------
    xarray.Dataset
        The input Dataset with the following new data variables added:
        - `wofs_ann_freq_sigma` (float): Standard deviation of the
            annual water frequency.
        - `wofs_ann_confidence` (int16): Confidence percentage of the
            annual water frequency.
        - `wofs_pw_threshold` (float): Dynamic threshold for permanent
            water.
        - `wofs_ann_pwater` (float): Annual permanent water classification.
        - `wofs_ann_water` (float): Annual general water classification.
        - `watermask` (float): A mask showing where general water is detected.

    """
    VALID_VARIABLES = ["wofs_ann_freq"]
    if wofs_varname not in VALID_VARIABLES:
        log.error(
            f"Invalid variable name {wofs_varname}! "
            "Defaulting to wofs_ann_freq"
        )
        wofs_varname = "wofs_ann_freq"

    data_vars = list(ds.data_vars)
    if wofs_varname not in data_vars:
        raise ValueError(
            "The provided dataset `ds` does not contain the required "
            f"wofs_varname:  {' or '.join(VALID_VARIABLES)}"
        )

    if wofs_varname == "wofs_ann_freq":
        # Standard deviation of the annual frequency at each pixel
        # should really be dividing by n-1 but then I would need to
        # change SC
        ds["wofs_ann_freq_sigma"] = (
            (ds.wofs_ann_freq * (1 - ds.wofs_ann_freq))
            / ds.wofs_ann_clearcount
        ) ** 0.5
        ds["wofs_ann_confidence"] = (
            (1.0 - (ds.wofs_ann_freq_sigma / ds.wofs_ann_freq)) * 100
        ).astype("int16")
        ds["wofs_pw_threshold"] = (
            -1 * ds.wofs_ann_freq_sigma * sigma_coefficient
        ) + permanent_water_threshold  # --- threshold varies with p and n
        ds["wofs_ann_pwater"] = xr.where(
            ds[wofs_varname] > ds.wofs_pw_threshold, ds[wofs_varname], 0
        )
        ds["wofs_ann_water"] = xr.where(
            ds[wofs_varname] > water_frequency_threshold, ds[wofs_varname], 0
        )
        # A variable called watermask is used in places.
        # I set the value of the mask as sigma or nan
        # Renamed this from watermask to wofs_ann_watermask to prevent
        # confusion with the 5 year summary watermask
        ds["wofs_ann_watermask"] = ds["wofs_ann_freq_sigma"].where(
            ds[wofs_varname] > water_frequency_threshold
        )
    else:
        raise NotImplementedError(
            f"Water detection for {wofs_varname} not implemented."
        )

    return ds
