import gc
import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import scipy as sp
import xarray as xr

log = logging.getLogger(__name__)


@contextmanager
def memory_cleanup():
    """Context manager for explicit memory cleanup."""
    try:
        yield
    finally:
        gc.collect()


def optimize_xarray_dataset(
    ds: xr.Dataset, chunk_size: dict = None
) -> xr.Dataset:
    """
    Optimize xarray dataset for memory efficiency.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to optimize.
    chunk_size : dict, optional
        Chunk sizes for dask arrays. Default uses auto chunking.

    Returns
    -------
    xr.Dataset
        Optimized dataset with chunking applied.
    """
    if chunk_size is None:
        chunk_size = {"time": "auto", "x": "auto", "y": "auto"}

    # Apply chunking for memory-efficient operations
    if not any(
        isinstance(var.data, type(ds.chunks)) for var in ds.data_vars.values()
    ):
        ds = ds.chunk(chunk_size)

    return ds


def process_data_in_batches(data: xr.DataArray, batch_size: int = 10):
    """
    Process large datasets in batches to reduce memory usage.

    Parameters
    ----------
    data : xr.DataArray
        Input data array.
    batch_size : int
        Number of time steps to process at once.

    Yields
    ------
    xr.DataArray
        Batched data slices.
    """
    time_dim = data.dims[0] if "time" in data.dims else data.dims[0]
    total_size = data.sizes[time_dim]

    for i in range(0, total_size, batch_size):
        end_idx = min(i + batch_size, total_size)
        yield data.isel({time_dim: slice(i, end_idx)})


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

    # Create non-zero permanent water mask for analysis
    ds["wofs_ann_pwater_nonzero"] = ds["wofs_ann_pwater"].where(
        (ds["wofs_ann_pwater"] != 0) & ~np.isnan(ds["wofs_ann_pwater"])
    )

    # Perform robust regression analysis
    return robust_regression(
        ds=ds,
        var_name="wofs_ann_pwater_nonzero",
        baseline_period=baseline_period,
        target_years=target_years,
        option="count",
    )
