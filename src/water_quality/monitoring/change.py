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
    ds["wofs_freq_median"] = input_ds["wofs_ann_freq"].median(dim="time")
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


def permanent_surface_water_dynamics_sdg(
    baseline_period_water: xr.DataArray, target_years_water: xr.DataArray
) -> dict[str, float]:
    """
    Compute the percent change in spatial extent for permanent surface
    water using the SDG Indicator 6.6.1 default methodology (i.e. the
    currently documented method that can be used globally).

    Parameters
    ----------
    baseline_period_water : xr.DataArray
        Array covering the baseline period where pixels are classified
        as no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).
    target_years_water : xr.DataArray
        Array covering the target years where pixels are classified as
        no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).

    Returns
    -------
    dict[str, float]
        Percent change in spatial extent for permanent surface water.
    """
    pixel_area = (
        abs(
            baseline_period_water.odc.geobox.resolution.x
            * baseline_period_water.odc.geobox.resolution.y
        )
        / 1000000
    )

    def _area(mask: xr.DataArray) -> float:
        return mask.sum().item() * pixel_area

    # conversion of a no water place into a permanent water place
    new_permanent_water = (baseline_period_water == 0) & (
        target_years_water == 3
    )
    alpha = _area(new_permanent_water)
    # conversion of a permanent water place into a no water place
    lost_permanent_water = (baseline_period_water == 3) & (
        target_years_water == 0
    )
    beta = _area(lost_permanent_water)
    # Conversion of seasonal water to permanent water
    seasonal_to_permanent_water = (baseline_period_water == 2) & (
        target_years_water == 3
    )
    phi = _area(seasonal_to_permanent_water)
    # Conversion of permanent water place into seasonal water
    permanent_to_seasonal_water = (baseline_period_water == 3) & (
        target_years_water == 2
    )
    sigma = _area(permanent_to_seasonal_water)
    # Area where water is always observerd
    permanent_water_surfaces = (baseline_period_water == 3) & (
        target_years_water == 3
    )
    epsilon = _area(permanent_water_surfaces)

    # Percentage change in spatial extent for permanent surface
    # water dynamics
    delta = (((alpha - beta) + (phi - sigma)) / (epsilon + beta + sigma)) * 100

    return {"permanent_water_area_change_%": delta}


def seasonal_surface_water_dynamics_sdg(
    baseline_period_water: xr.DataArray, target_years_water: xr.DataArray
) -> dict[str, float]:
    """
    Compute the percent change in spatial extent for seasonal surface
    water using the SDG Indicator 6.6.1 default methodology (i.e. the
    currently documented method that can be used globally).

    Parameters
    ----------
    baseline_period_water : xr.DataArray
        Array covering the baseline period where pixels are classified
        as no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).
    target_years_water : xr.DataArray
        Array covering the target years where pixels are classified as
        no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).

    Returns
    -------
    dict[str, float]
        Percent change in spatial extent for seasonal surface water.
    """
    pixel_area = (
        abs(
            baseline_period_water.odc.geobox.resolution.x
            * baseline_period_water.odc.geobox.resolution.y
        )
        / 1000000
    )

    def _area(mask: xr.DataArray) -> float:
        return mask.sum().item() * pixel_area

    # conversion of a no water place into a seasonal water place
    new_seasonal_water = (baseline_period_water == 0) & (
        target_years_water == 2
    )
    alpha = _area(new_seasonal_water)
    # conversion of a seasonal water place into a no water place
    lost_seasonal_water = (baseline_period_water == 2) & (
        target_years_water == 0
    )
    beta = _area(lost_seasonal_water)
    # conversion of permanent water into seasonal water
    permanent_to_seasonal = (baseline_period_water == 3) & (
        target_years_water == 2
    )
    phi = _area(permanent_to_seasonal)
    # Conversion of seasonal water into permanent water
    seasonal_to_permanent_water = (baseline_period_water == 2) & (
        target_years_water == 3
    )
    sigma = _area(seasonal_to_permanent_water)
    # area where seasonal water is always observed
    seasonal_water_surfaces = (baseline_period_water == 2) & (
        target_years_water == 2
    )
    epsilon = _area(seasonal_water_surfaces)

    # Percentage change in spatial extent for seasonal surface
    # water dynamics
    delta = (((alpha - beta) + (phi - sigma)) / (epsilon + beta + sigma)) * 100

    return {"seasonal_water_area_change_%": delta}


def permanent_surface_water_dynamics_simple(
    baseline_period_water: xr.DataArray, target_years_water: xr.DataArray
) -> dict[str, float]:
    """
    Compute the change in spatial extent for permanent surface water.

    Parameters
    ----------
    baseline_period_water : xr.DataArray
        Array covering the baseline period where pixels are classified
        as no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).
    target_years_water : xr.DataArray
        Array covering the target years where pixels are classified as
        no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).

    Returns
    -------
    dict[str, float]
        Change in spatial extent for permanent surface water.
    """
    pixel_area = (
        abs(
            baseline_period_water.odc.geobox.resolution.x
            * baseline_period_water.odc.geobox.resolution.y
        )
        / 1_000_000
    )

    def _area(mask: xr.DataArray) -> float:
        return mask.sum().item() * pixel_area

    baseline_period_water_area = _area(baseline_period_water == 2)
    target_years_water_area = _area(target_years_water == 2)

    change_in_water_area = target_years_water_area - baseline_period_water_area
    pc_change_in_water_area = np.round(
        (target_years_water_area / baseline_period_water_area - 1.0) * 100, 2
    )
    results = {
        "baseline_period_permanent_water_area_km2": baseline_period_water_area,
        "target_years_permanent_water_area_km2": target_years_water_area,
        "permanent_water_area_change_km2": change_in_water_area,
        "permanent_water_area_change_%": pc_change_in_water_area,
    }
    return results


def seasonal_surface_water_dynamics_simple(
    baseline_period_water: xr.DataArray, target_years_water: xr.DataArray
) -> dict[str, float]:
    """
    Compute the change in spatial extent for seasonal surface water.

    Parameters
    ----------
    baseline_period_water : xr.DataArray
        Array covering the baseline period where pixels are classified
        as no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).
    target_years_water : xr.DataArray
        Array covering the target years where pixels are classified as
        no water (0), ephemeral water(1), seasonal water(2) and
        permanent water(3).

    Returns
    -------
    dict[str, float]
        Change in spatial extent for seasonal surface water.
    """
    pixel_area = (
        abs(
            baseline_period_water.odc.geobox.resolution.x
            * baseline_period_water.odc.geobox.resolution.y
        )
        / 1000000
    )

    def _area(mask: xr.DataArray) -> float:
        return mask.sum().item() * pixel_area

    baseline_period_water_area = _area(
        (baseline_period_water > 0) & (baseline_period_water < 3)
    )
    target_years_water_area = _area(
        (target_years_water > 0) & (target_years_water < 3)
    )
    change_in_water_area = target_years_water_area - baseline_period_water_area
    pc_change_in_water_area = np.round(
        (target_years_water_area / baseline_period_water_area - 1.0) * 100, 2
    )
    results = {
        "baseline_period_seasonal_water_area_km2": baseline_period_water_area,
        "target_years_seasonal_water_area_km2": target_years_water_area,
        "seasonal_water_area_change_km2": change_in_water_area,
        "seasonal_water_area_change_%": pc_change_in_water_area,
    }
    return results


def permanent_water_area_change(
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
    water_frequency_thresholds: list[float] = [0, 0.15, 0.875],
) -> dict[str, float]:
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
    dict[str, float]
        Changes in permanent water area.
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
    regression_results = {
        "permanent_water_regression_slope": regression["slope"],
        "permanent_water_regression_intercept": regression["intercept"],
        "permanent_water_regression_slope_low": regression["slope_low"],
        "permanent_water_regression_slope_high": regression["slope_high"],
        "permanent_water_regression_significant": regression["significant"],
        "permenent_water_declining": regression["declining"],
        "permenent_water_increasing": regression["increasing"],
    }
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

    permanent_water_statistics = permanent_surface_water_dynamics_simple(
        baseline_period_water=baseline_ds["water"],
        target_years_water=target_ds["water"],
    )
    seasonal_water_statistics = seasonal_surface_water_dynamics_simple(
        baseline_period_water=baseline_ds["water"],
        target_years_water=target_ds["water"],
    )

    # TODO: switch keys to sdg symbols
    # EN_LKRV_PWAN: Lakes and rivers permanent water area (square kilometres)
    # EN_LKRV_SWAN: Lakes and rivers seasonal water area (square kilometres)
    # EN_LKRV_PWAC: Lakes and rivers permanent water area change (%)
    # EN_LKRV_SWAC: Lakes and rivers seasonal water area change (%)
    lkrv_pwac = {
        **permanent_water_statistics,
        **seasonal_water_statistics,
        **regression_results,
    }
    return lkrv_pwac


def classify_deviation(deviation_da: xr.DataArray) -> xr.DataArray:
    """
    Classify pixels in a monthly or annual deviation synthesis into
    deviation levels.

    Pixels are classified into 4 classes based the range of values the
    pixel value falls into.
    - 1: Low (0-25%)
    - 2: Medium (25–50%)
    - 3: High (50–75%)
    - 4: Extreme (75-100%)

    Parameters
    ----------
    deviation_da : xr.DataArray
        Monthly or annual deviation synthesis for a waterbody.

    Returns
    -------
    xr.DataArray
        Array of classified pixel values, where each pixel is assigned:
        1 (low), 2 (medium), 3 (high), or 4 (extreme).
    """
    conditions = [
        (deviation_da < 25),
        ((deviation_da >= 25) & (deviation_da < 50)),
        ((deviation_da >= 50) & (deviation_da < 75)),
        (deviation_da >= 75),
    ]
    classes = [1, 2, 3, 4]
    classified_deviation = xr.DataArray(
        data=np.select(conditions, classes, default=np.nan),
        coords=deviation_da.coords,
        dims=deviation_da.dims,
    )
    return classified_deviation


def waterbody_is_affected(
    deviation_classes: xr.DataArray,
) -> dict[str, float | bool]:
    """
    Determine whether a waterbody is affected.

    Compares the number of 'high' (3) and 'extreme' (4) deviation pixels
    to the number of 'low' (1) and 'medium' (2) deviation pixels.
    A waterbody is considered affected if the number of high and extreme
    pixels exceeds the number of low and medium pixels.

    Parameters
    ----------
    deviation_classes : xr.DataArray
        Dataset containing classified annual deviation or monthly
        deviation values.

    Returns
    -------
    dict[str, float | bool]:
        A dictionary containing the area of high and extreme pixels in
        km2, area of low and medium pixels in km2 and a boolean indicating
        whether a waterbody is affected (True) or not.
    """

    pixel_area = (
        abs(
            deviation_classes.odc.geobox.resolution.x
            * deviation_classes.odc.geobox.resolution.y
        )
        / 1000000
    )

    def _area(mask: xr.DataArray) -> float:
        return mask.sum().item() * pixel_area

    high_and_extreme = _area(deviation_classes.where(deviation_classes >= 3))
    low_and_medium = _area(deviation_classes.where(deviation_classes <= 2))
    affected = high_and_extreme > low_and_medium
    results = {
        "HE_km2": high_and_extreme,
        "LM_km2": low_and_medium,
        "affected": affected,
    }
    return results


def change_in_turbidity(
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
) -> dict[str, Any]:
    """
    Compute the change in turbidity for a waterbody

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing Total Suspended Solids (TSS) data.
    baseline_period : tuple[str]
        The baseline period start and end years.
    target_years : str
        The target years for comparison.
    Returns
    -------
    dict[str, Any]
        Change in turbidity computed using a robust regression and
        classification on whether a waterbody is affected.
    """
    # Total Suspended Solids (TSS)
    var_name = "tss_agm_med"
    regression = robust_regression(
        ds=ds,
        var_name=var_name,
        baseline_period=baseline_period,
        target_years=target_years,
        option="median",
    )
    tss_regression = {
        "tss_regression_slope": regression["slope"],
        "tss_regression_intercept": regression["intercept"],
        "tss_regression_slope_low": regression["slope_low"],
        "tss_regression_slope_high": regression["slope_high"],
        "tss_regression_significant": regression["significant"],
        "tss_declining": regression["declining"],
        "tss_increasing": regression["increasing"],
    }

    da = ds[var_name]
    baseline_slice = slice(min(baseline_period), max(baseline_period))
    baseline_da = da.sel(time=baseline_slice).median(dim="time")

    target_slice = slice(min(target_years), max(target_years))
    target_da = da.sel(time=target_slice).median(dim="time")

    # Annual deviation synthesis
    tss_percent_change = ((target_da - baseline_da) / baseline_da) * 100
    tss_percent_change = tss_percent_change.clip(min=0, max=100)
    tss_classified_deviation = classify_deviation(tss_percent_change)
    affected_results = waterbody_is_affected(tss_classified_deviation)

    lkw_qltrb = {**tss_regression, **affected_results}
    return lkw_qltrb


def change_in_trophic_state(
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
):
    """
    Compute the change in trophic state for a waterbody

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing Trophic State Index (TSI) data and
        chlorophyll-a concentration data.
    baseline_period : tuple[str]
        The baseline period start and end years.
    target_years : str
        The target years for comparison.
    Returns
    -------
    dict[str, Any]
        Change in trophic state computed using a robust regression and
        classification on whether a waterbody is affected.
    """
    # chlorophyll-a concentration
    var_name = "chla_agm_med"
    regression = robust_regression(
        ds=ds,
        var_name=var_name,
        baseline_period=baseline_period,
        target_years=target_years,
        option="median",
    )
    chla_regression = {
        "chla_regression_slope": regression["slope"],
        "chla_regression_intercept": regression["intercept"],
        "chla_regression_slope_low": regression["slope_low"],
        "chla_regression_slope_high": regression["slope_high"],
        "chla_regression_significant": regression["significant"],
        "chla_declining": regression["declining"],
        "chla_increasing": regression["increasing"],
    }

    # Trophic State Index (TSI)
    var_name = "TSI"
    regression = robust_regression(
        ds=ds,
        var_name=var_name,
        baseline_period=baseline_period,
        target_years=target_years,
        option="mean",
    )
    tsi_regression = {
        "TSI_regression_slope": regression["slope"],
        "TSI_regression_intercept": regression["intercept"],
        "TSI_regression_slope_low": regression["slope_low"],
        "TSI_regression_slope_high": regression["slope_high"],
        "TSI_regression_significant": regression["significant"],
        "TSI_declining": regression["declining"],
        "TSI_increasing": regression["increasing"],
    }

    da = ds[var_name]
    baseline_slice = slice(min(baseline_period), max(baseline_period))
    baseline_da = da.sel(time=baseline_slice).median(dim="time")

    target_slice = slice(min(target_years), max(target_years))
    target_da = da.sel(time=target_slice).median(dim="time")

    # Annual deviation synthesis
    TSI_percent_change = ((target_da - baseline_da) / baseline_da) * 100
    TSI_percent_change = TSI_percent_change.clip(min=0, max=100)
    TSI_classified_deviation = classify_deviation(TSI_percent_change)
    affected_results = waterbody_is_affected(TSI_classified_deviation)

    lkw_qltrst = {**chla_regression, **tsi_regression, **affected_results}
    return lkw_qltrst


def water_quality_change(
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
) -> tuple[dict]:
    """
    Compute the change in water quality of a waterbody.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing Total Suspended Solids (TSS), Trophic State
        Index (TSI) and chlorophyll-a concentration data.
    baseline_period : tuple[str]
        The baseline period start and end years.
    target_years : str
        The target years for comparison.

    Returns
    -------
    tuple[dict]
        Change in turbidity and trophic state computed using a robust
        regression and classification on whether a waterbody is affected.

    """
    lkw_qltrb = change_in_turbidity(ds, baseline_period, target_years)
    lkw_qltrst = change_in_trophic_state(ds, baseline_period, target_years)
    return lkw_qltrb, lkw_qltrst
