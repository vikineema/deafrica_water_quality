import gc
import logging
from typing import Any

import numpy as np
import scipy as sp
import xarray as xr

log = logging.getLogger(__name__)


def robust_regression(
    ds: xr.Dataset,
    var_name: str,
    baseline_period: tuple[str],
    target_years: tuple[str],
    option: str = "median",
) -> dict[str, Any]:
    """
    Use a robust regression to compare data between the baseline period
    and the target years. The regression gives a "yes" / "True" or "no" /
    "False" about whether the periods are different, based on a 95%
    confidence level. The outputs also include a regression line (slope,
    intercept).
    The regression method has the advantage of picking up improvements
    in water quality, unlike the proposed SDG Indicator 6.6.1
    methodology.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the baseline period and target years data.
    var_name : str
        Variable name of the band to perform the regression on.
    baseline_period : tuple[str]
        The baseline period start and end years. 
    target_years : tuple[str]
        The target years start and end years. 
    option : str, optional
        Options for summarizing data., by default "median"

    Returns
    -------
    dict[str, Any]
        Dictionary including the results of the regression inlcuding:
        * slope: Theil slope
        * intercept: Intercept of the Theil line
        * slope_low: Lower bound of the confidence interval on slope.
        * slope_high: Upper bound of the confidence interval on slope.
        * significant: "True" or "False" about whether the periods are \
          different.
        * declining: "True" or "False" on whether the conditions are \
            degrading.
        * increasing: "True" or "False" on whether the conditions are \
            improving.
    """
    baseline_slice = slice(min(baseline_period), max(baseline_period))
    baseline_times = ds.sel(time=baseline_slice).time.values

    target_slice = slice(min(target_years), max(target_years))
    target_times = ds.sel(time=target_slice).time.values

    da = ds[var_name]
    baseline_da = da.sel(time=baseline_times)
    target_da = da.sel(time=target_times)

    aggregation_funcs = {
        "median": lambda x: x.median(dim=("x", "y")),
        "mean": lambda x: x.mean(dim=("x", "y")),
        "sum": lambda x: x.sum(dim=("x", "y")),
        "count": lambda x: x.count(dim=("x", "y")),
    }

    if option not in aggregation_funcs:
        raise ValueError(f"Unsupported option '{option}'.")

    agg_func = aggregation_funcs[option]

    baseline_series = agg_func(baseline_da).values
    target_series = agg_func(target_da).values

    series = np.hstack([baseline_series, target_series])

    # Replace infinite values with nans and nans with the mean of
    # the series
    mask_inf = np.isinf(series)
    if np.any(mask_inf):
        series[mask_inf] = np.nan

    mask_nan = np.isnan(series)
    if np.any(mask_nan):
        series_mean = np.nanmean(series)
        series[mask_nan] = series_mean

    times = np.concatenate([baseline_times, target_times])

    # Perform regression
    slope, intercept, slope_low, slope_high = sp.stats.theilslopes(
        series, times, method="separate"
    )

    # Calculate significance flags
    significant = np.sign(slope_low) == np.sign(slope_high)
    declining = significant and slope_high < 0
    increasing = significant and slope_low > 0

    if option in ["mean", "median"]:
        agg_func = aggregation_funcs["mean"]
        baseline_agg = agg_func(baseline_da).mean(dim="time").item()
        target_agg = agg_func(target_da).mean(dim="time").item()
    elif option in ["count", "sum"]:
        agg_func = aggregation_funcs["sum"]
        baseline_agg = agg_func(baseline_da).mean(dim="time").item()
        target_agg = agg_func(target_da).mean(dim="time").item()
    else:
        raise ValueError(f"Unsupported option '{option}'.")

    y_mean = np.mean([baseline_agg, target_agg])

    # Calculate model mean efficiently
    time_mean = np.mean(
        [
            baseline_times.min().astype("float"),
            target_times.max().astype("float"),
        ]
    )
    model_mean = time_mean * slope

    if not np.isnan(y_mean - model_mean):
        intercept = y_mean - model_mean
    else:
        raise NotImplementedError("Cannot calculate intercept with NaN values")

    log.info(f"Affected = {significant}")
    log.info(f"Declining = {declining}")

    # Memory optimization: use explicit cleanup
    del baseline_da, target_da, series, times
    if "series_mean" in locals():
        del series_mean

    gc.collect()

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_low": slope_low,
        "slope_high": slope_high,
        "significant": significant,
        "declining": declining,
        "increasing": increasing,
    }


def classify_permanent_water_using_threshold(
    input_ds: xr.Dataset, permanent_water_threshold: float
) -> xr.Dataset:
    """
    Classify permanent water pixels in a dataset by summarizing the
    dataset over time and applying the permanent water threshold.

    Parameters
    ----------
    input_ds : xr.Dataset
        Input dataset for the period to classify water for.
    permanent_water_threshold : float
        Water threshold value above which a pixel is classified as permanent
        water.

    Returns
    -------
    xr.Dataset
        Summarized dataset containing the bands:
        * `wofs_freq`: the band contains the frequency with which a
            pixel was classified as wet.
        * `pwater_wofs_freq_method`: this band contains the water
            classification where permanent water pixels have a value
            of 1.

    """
    ds = xr.Dataset()
    ds["wofs_wetcount"] = input_ds["wofs_ann_wetcount"].sum(dim="time")
    ds["wofs_clearcount"] = input_ds["wofs_ann_clearcount"].sum(dim="time")
    ds["wofs_freq_median"] = (input_ds["wofs_ann_freq"].median(dim="time"),)
    ds["wofs_freq"] = ds["wofs_wetcount"] / ds["wofs_clearcount"]
    ds["wofs_freq_sdv"] = np.sqrt(
        (ds["wofs_freq"] * (1.0 - ds["wofs_freq"]))
        / (ds["wofs_clearcount"] - 1)
    )
    ds["wofs_pwater_threshold"] = (
        -1 * ds["wofs_freq_sdv"] * 1.2
    ) + permanent_water_threshold

    ds["pwater_wofs_freq_method"] = (
        ds["wofs_freq"] > ds["wofs_pwater_threshold"]
    ).astype(int)
    return ds[["wofs_freq", "pwater_wofs_freq_method"]]


def classify_permanent_water_using_sdg_method(
    input_ds: xr.Dataset,
) -> xr.DataArray:
    var_name = "wofs_ann_pwater"
    da = input_ds[var_name]

    # Determine  water state (permanent, seasonal or no water)
    # by majority rule.
    is_valid = ~da.isnull()
    is_wet = (da > 0) & is_valid
    is_dry = (da <= 0) & is_valid

    wet_count = is_wet.sum(dim="time")
    dry_count = is_dry.sum(dim="time")

    # Determine the water state (permanent, seasonal or no water)
    # by a majority rule.
    permanent_water = (wet_count > dry_count).astype(int)
    return permanent_water


def classify_water(
    input_ds: xr.Dataset,
    permanent_water_threshold: float,
    water_threshold: float,
    ephemeral_water_threshold: float,
) -> xr.Dataset:
    """
    Classify water pixels in a dataset into the following classes:
    * 0 : no water
    * 1 : ephemeral water
    * 2 : seasonal water
    * 3 : permanent water

    Parameters
    ----------
    input_ds : xr.Dataset
        Input dataset to carry out the classification on.
    permanent_water_threshold : float
        Water threshold value above which a pixel is classified as
        permanent water.
    water_threshold : float
        Water threshold value above which a pixel is classified as
        seasonal water.
    ephemeral_water_threshold : float
        Water threshold value above which a pixel is classified as
        ephemeral water.

    Returns
    -------
    xr.Dataset
        Dataset containing the band `water` where the pixels are classified
        as no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).
    """
    ds = classify_permanent_water_using_threshold(
        input_ds=input_ds,
        permanent_water_threshold=permanent_water_threshold,
    )
    ds["pwater_sdg_method"] = classify_permanent_water_using_sdg_method(
        input_ds=input_ds
    )
    # encode water as 0,1,2,3 (no water, ephemeral, seasonal, permanent
    ds["water"] = (
        ds["pwater_sdg_method"]
        + (ds["wofs_freq"] > water_threshold).astype(int)
        + (ds["wofs_freq"] > ephemeral_water_threshold).astype(int)
    )
    return ds


# Lakes and rivers permanent water area change (%)
# EN_LKRV_PWAC
def permanent_water_area_change(
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
    water_frequency_thresholds: list[float] = [0, 0.15, 0.875],
):
    """
    Calculate permanent water area change using robust regression.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing water observation frequency data.
    baseline_period : tuple[str]
        The baseline period start and end years.
    target_years : str
        The target years for comparison.
    water_frequency_thresholds : list[float], optional
        Water frequency thresholds, by default [0, 0.15, 0.875].

    Returns
    -------
    dict
        Results from robust regression analysis.
    """
    pwater_threshold = water_frequency_thresholds[0]
    water_threshold = water_frequency_thresholds[1]
    ephemeral_water_threshold = water_frequency_thresholds[2]

    ds["wofs_ann_pwater_nonzero"] = ds["wofs_ann_pwater"].where(
        (ds["wofs_ann_pwater"] != 0) & (~np.isnan(ds["wofs_ann_pwater"]))
    )
    regression = robust_regression(
        ds=ds,
        var_name="wofs_ann_pwater_nonzero",
        baseline_period=baseline_period,
        target_years=target_years,
        option="count",
    )
    # Baseine period data processing
    baseline_slice = slice(min(baseline_period), max(baseline_period))
    baseline_ds = classify_water(
        input_ds=ds.sel(time=baseline_slice),
        permanent_water_threshold=pwater_threshold,
        water_threshold=water_threshold,
        ephemeral_water_threshold=ephemeral_water_threshold,
    )
    # Target years data processing
    target_slice = slice(min(target_years), max(target_years))
    target_ds = classify_water(
        input_ds=ds.sel(time=target_slice),
        permanent_water_threshold=pwater_threshold,
        water_threshold=water_threshold,
        ephemeral_water_threshold=ephemeral_water_threshold,
    )

    # conversion of a no water place into a permanent water place
    new_permanent_water = None
    # conversion of a permanent water place into a no water place
    lost_permanent_water = None
    # Conversion of a no water place into a seasonal water place
    new_seasonal_water = None
    # Conversion of a season water place into a no water place
    lost_seasonal_water = None
    # Conversion of  a permanent water place into seasonal water
    permanent_to_seasonal_water = None
    # Conversion of seaonal water to permanent water
    seasonal_to_permanent_water = None
    # Area where water is always observerd
    permanent_water_surfaces = None
    # Area where seasonal water is always observerd
    seasonal_water_surfaces = None
