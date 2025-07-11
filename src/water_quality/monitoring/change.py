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
    baseline_period_times = ds.sel(
        time=slice(min(baseline_period), max(baseline_period))
    ).time.values
    target_years_times = ds.sel(
        time=slice(min(target_years), max(target_years))
    ).time.values
    times = np.hstack((baseline_period_times, target_years_times))

    baseline_period_da = ds[var_name].sel(time=baseline_period_times)
    target_years_da = ds[var_name].sel(time=target_years_times)

    if option == "median":
        baseline_period_series = baseline_period_da.median(
            dim=("x", "y")
        ).values
        target_years_series = target_years_da.median(dim=("x", "y")).values
    elif option == "mean":
        baseline_period_series = baseline_period_da.mean(dim=("x", "y")).values
        target_years_series = target_years_da.mean(dim=("x", "y")).values
    elif option == "sum":
        baseline_period_series = baseline_period_da.sum(dim=("x", "y")).values
        target_years_series = target_years_da.sum(dim=("x", "y")).values
    elif option == "count":
        baseline_period_series = baseline_period_da.count(
            dim=("x", "y")
        ).values
        target_years_series = target_years_da.count(dim=("x", "y")).values
    else:
        raise ValueError(f"Unsupported option '{option}'.")

    # Replace infinite values with nans and nans with the mean of
    # the series
    series = np.hstack((baseline_period_series, target_years_series))
    series = np.where(np.isinf(series), np.nan, series)
    series = np.where(np.isnan(series), np.nanmean(series), series)

    slope, intercept, slope_low, slope_high = sp.stats.theilslopes(
        series, times, method="separate"
    )
    significant = np.sign(slope_low) == np.sign(slope_high)
    declining = significant and slope_high < 0
    increasing = significant and slope_low > 0

    # The slope is the important parameter, but a better estimate of the
    # intercept is useful for graphing purposes.
    if option in ["mean", "median"]:
        y_mean = np.mean(
            [
                baseline_period_da.mean(dim=("x", "y"))
                .mean(dim="time")
                .item(),
                target_years_da.mean(dim=("x", "y")).mean(dim="time").item(),
            ]
        )
    elif option in ["count", "sum"]:
        y_mean = np.mean(
            [
                baseline_period_da.sum(dim=("x", "y")).mean(dim="time").item(),
                target_years_da.sum(dim=("x", "y")).mean(dim="time").item(),
            ]
        )
    else:
        raise ValueError(f"Unsupported option '{option}'.")

    model_mean = (
        np.mean(
            [
                baseline_period_times.min().astype("float"),
                target_years_times.max().astype("float"),
            ]
        )
        * slope
    )

    if not np.isnan(y_mean - model_mean):
        intercept = y_mean - model_mean
    else:
        raise NotImplementedError()

    log.info(f"Affected = {significant}")
    log.info(f"Declining = {declining}")

    parameters = dict(
        slope=slope,
        intercept=intercept,
        slope_low=slope_low,
        slope_high=slope_high,
        significant=significant,
        declining=declining,
        increasing=increasing,
    )
    return parameters


# Lakes and rivers permanent water area change (%)
# EN_LKRV_PWAC
def permanent_water_area_change(
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
    water_frequency_thresholds: list[float] = [0, 0.15, 0.875],
):
    pwater_threshold = water_frequency_thresholds[0]
    water_threshold = water_frequency_thresholds[1]
    ephemeral_water_threshold = water_frequency_thresholds[2]

    ds["wofs_ann_pwater_nonzero"] = ds["wofs_ann_pwater"].where(
        (ds["wofs_ann_pwater"] != 0) & ~np.isnan(ds["wofs_ann_pwater"])
    )
    regression = robust_regression(
        ds=ds,
        var_name="wofs_ann_pwater_nonzero",
        baseline_period=baseline_period,
        target_years=target_years,
        option="count",
    )
