from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

from water_quality.monitoring.change import piecewise_linreg


def get_figure_template() -> dict[str, Any]:
    """Get template for components that make up a figure.

    Returns
    -------
    dict[str, Any]
        Template of expected plots making up a figure
    """

    figure_template = dict(
        figure_title="",
        data_points_plot={
            "x": None,
            "y": None,
            "color": "k",
            "linestyle": "",
            "marker": "X",
            "label": "{label_prefix}, mean for the waterbody (data points)",
        },
        trend_curve_plot={
            "x": None,
            "y": None,
            "color": "k",
            "linestyle": "dotted",
            "marker": "none",
            "label": "{label_prefix}, mean for the waterbody (smoothed fit)",
        },
        sdg_baseline_period_plot={
            "x": None,
            "y": None,
            "color": "g",
            "linestyle": "dashed",
            "marker": "none",
            "label": "{label_prefix} for baseline period (SDG method)",
        },
        sdg_target_years_plot={
            "x": None,
            "y": None,
            "color": "r",
            "linestyle": "dashed",
            "marker": "none",
            "label": "{label_prefix} for target years (SDG method)",
        },
        regression_line_plot={
            "x": None,
            "y": None,
            "color": "b",
            "linestyle": "--",
            "marker": "none",
            "label": "Robust regression comparing baseline period and target years",
        },
        y_label="",
        x_label="Year",
        output_file_name="",
    )
    return figure_template


def plot_figure(figure_components: dict[str, Any]):
    figure_title = figure_components["figure_title"]
    data_points_plot = figure_components["data_points_plot"]
    trend_curve_plot = figure_components["trend_curve_plot"]
    sdg_baseline_period_plot = figure_components["sdg_baseline_period_plot"]
    sdg_target_years_plot = figure_components["sdg_target_years_plot"]
    regression_line_plot = figure_components["regression_line_plot"]
    y_label = figure_components["y_label"]
    x_label = figure_components["x_label"]
    output_file_name = figure_components["output_file_name"]

    # Unpack
    plt.figure(figsize=[12, 6])

    plt.plot(
        data_points_plot["x"],
        data_points_plot["y"],
        color=data_points_plot["color"],
        linestyle=data_points_plot["linestyle"],
        marker=data_points_plot["marker"],
        label=data_points_plot["label"],
    )

    plt.plot(
        trend_curve_plot["x"],
        trend_curve_plot["y"],
        color=trend_curve_plot["color"],
        linestyle=trend_curve_plot["linestyle"],
        marker=trend_curve_plot["marker"],
        label=trend_curve_plot["label"],
    )

    plt.plot(
        sdg_baseline_period_plot["x"],
        sdg_baseline_period_plot["y"],
        color=sdg_baseline_period_plot["color"],
        linestyle=sdg_baseline_period_plot["linestyle"],
        marker=sdg_baseline_period_plot["marker"],
        label=sdg_baseline_period_plot["label"],
    )

    plt.plot(
        sdg_target_years_plot["x"],
        sdg_target_years_plot["y"],
        color=sdg_target_years_plot["color"],
        linestyle=sdg_target_years_plot["linestyle"],
        marker=sdg_target_years_plot["marker"],
        label=sdg_target_years_plot["label"],
    )

    plt.plot(
        regression_line_plot["x"],
        regression_line_plot["y"],
        color=regression_line_plot["color"],
        linestyle=regression_line_plot["linestyle"],
        marker=regression_line_plot["marker"],
        label=regression_line_plot["label"],
    )

    plt.title(figure_title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.savefig(output_file_name, dpi=150)
    plt.show()


def plot_change_in_permanent_water_area(
    waterbody_uid: str,
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
    lkrv_pwac: dict[str, Any],
):
    permenent_water_area_figure = get_figure_template()

    permenent_water_area_figure["figure_title"] = (
        f"Waterbody {waterbody_uid} - SDG : LKRV_WAPC (Change in area of "
        "Permanent Water)\n Annual surface areas of Permanent Water (PW)"
        f"\n Baseline period: {min(baseline_period)} to {max(baseline_period)} "
        f"Target years: {min(target_years)} to {max(target_years)}"
        f"\n PW level change ({lkrv_pwac['target_years_permanent_water_area_km2']} "
        f"- {lkrv_pwac['baseline_period_permanent_water_area_km2']}) /  "
        f"{lkrv_pwac['baseline_period_permanent_water_area_km2']} = "
        f"{lkrv_pwac['permanent_water_area_change_%']} %"
        "\n Significant trend based on robust regression over reference and "
        f"assessment years = {lkrv_pwac['permanent_water_regression_significant']}"
        f"\n Decline in water area = {lkrv_pwac['permenent_water_declining']}"
    )

    # Data points plot
    pixel_area = (
        abs(ds.odc.geobox.resolution.x * ds.odc.geobox.resolution.y) / 1000000
    )

    def _area(mask: xr.DataArray) -> float:
        return mask.sum(dim=("x", "y")) * pixel_area

    permanent_water_area = _area(
        (
            ~np.isnan(ds["wofs_ann_pwater"]) & (ds["wofs_ann_pwater"] > 0)
        ).astype(int)
    )
    permanent_water_area = permanent_water_area.where(
        ~np.isinf(permanent_water_area), np.nan
    )

    label_prefix = "Permanent water area"
    data_points_plot = permenent_water_area_figure["data_points_plot"]
    data_points_plot.update(
        {
            "x": permanent_water_area.time.values,
            "y": permanent_water_area.values,
            "label": data_points_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Trend curve plot
    # Create a two step interpolation function: a piecewise linear
    # regression followed by a quadratic interpolator
    _, fitted_y, *_ = piecewise_linreg(
        x=np.arange(0, permanent_water_area.time.size),
        y=permanent_water_area.values,
    )
    interp_func = sp.interpolate.interp1d(
        x=permanent_water_area.time,
        y=fitted_y,
        kind="quadratic",
        fill_value="extrapolate",
    )
    # Generate a new, denser time range for interpolation
    time_new = pd.date_range(
        permanent_water_area.time.min().values,
        permanent_water_area.time.max().values,
        freq="2M",
    )
    # Fit a trend curve to the data
    trend_curve = np.clip(
        interp_func(time_new),
        0,
        permanent_water_area.max().item(),
    )

    trend_curve_plot = permenent_water_area_figure["trend_curve_plot"]
    trend_curve_plot.update(
        {
            "x": time_new,
            "y": trend_curve,
            "label": trend_curve_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # SDG method baseline period plot
    baseline_slice = slice(min(baseline_period), max(baseline_period))
    baseline_times = permanent_water_area.sel(time=baseline_slice).time.values
    baseline_period_mean_values = np.full(
        baseline_times.shape,
        lkrv_pwac["baseline_period_permanent_water_area_km2"],
    )

    sdg_baseline_period_plot = permenent_water_area_figure[
        "sdg_baseline_period_plot"
    ]
    sdg_baseline_period_plot.update(
        {
            "x": baseline_times,
            "y": baseline_period_mean_values,
            "label": sdg_baseline_period_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # SDG method target years plot
    target_slice = slice(min(target_years), max(target_years))
    target_times = permanent_water_area.sel(time=target_slice).time.values
    target_years_mean_values = np.full(
        target_times.shape, lkrv_pwac["target_years_permanent_water_area_km2"]
    )
    sdg_target_years_plot = permenent_water_area_figure[
        "sdg_target_years_plot"
    ]
    sdg_target_years_plot.update(
        {
            "x": target_times,
            "y": target_years_mean_values,
            "label": sdg_target_years_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Robust regression line comparing baseline period and target years
    regression_span = permanent_water_area.sel(
        time=slice(min(baseline_times), max(target_times))
    ).time.values
    regression_line_x = regression_span.astype("float")
    regression_line_slope = lkrv_pwac["permanent_water_regression_slope"]
    regression_line_intercept = lkrv_pwac[
        "permanent_water_regression_intercept"
    ]
    regression_line_scale = pixel_area
    regression_line_y = (
        (regression_line_slope * regression_line_x) + regression_line_intercept
    ) * regression_line_scale
    regression_line_plot = permenent_water_area_figure["regression_line_plot"]
    regression_line_plot.update(
        {
            "x": regression_span,
            "y": regression_line_y,
            "label": regression_line_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )
    permenent_water_area_figure["y_label"] = "Area (km2)"
    permenent_water_area_figure["output_file_name"] = (
        f"{waterbody_uid}_lkrv_pwac.png"
    )
    plot_figure(permenent_water_area_figure)


def plot_change_in_turbidity(
    waterbody_uid: str,
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
    lkw_qltrb={},
):
    tss_figure = get_figure_template()

    tss_da = ds.tss_agm_med.mean(dim=("x", "y"))
    tss_da = tss_da.where(~np.isinf(tss_da), np.nan)

    # Data points plot
    label_prefix = "TSS values"
    data_points_plot = tss_figure["data_points_plot"]
    data_points_plot.update(
        {
            "x": tss_da.time.values,
            "y": tss_da.values,
            "label": data_points_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Trend curve plot
    # Create a two step interpolation function: a piecewise linear
    # regression followed by a quadratic interpolator
    _, fitted_y, *_ = piecewise_linreg(
        x=np.arange(0, tss_da.time.size),
        y=tss_da.values,
    )
    interp_func = sp.interpolate.interp1d(
        x=tss_da.time,
        y=fitted_y,
        kind="quadratic",
        fill_value="extrapolate",
    )
    # Generate a new, denser time range for interpolation
    time_new = pd.date_range(
        tss_da.time.min().values,
        tss_da.time.max().values,
        freq="2M",
    )
    # Fit a trend curve to the data
    baseline_slice = slice(min(baseline_period), max(baseline_period))
    baseline_times = tss_da.sel(time=baseline_slice).time.values
    baseline_period_mean = (
        tss_da.sel(time=baseline_times).mean(dim="time").item()
    )
    baseline_period_mean = np.round(baseline_period_mean, 1)

    target_slice = slice(min(target_years), max(target_years))
    target_times = tss_da.sel(time=target_slice).time.values
    target_years_mean = tss_da.sel(time=target_times).mean(dim="time").item()
    target_years_mean = np.round(target_years_mean, 1)

    clip_max = np.max((100, baseline_period_mean, target_years_mean))
    trend_curve = np.clip(
        interp_func(time_new),
        0,
        clip_max,
    )

    trend_curve_plot = tss_figure["trend_curve_plot"]
    trend_curve_plot.update(
        {
            "x": time_new,
            "y": trend_curve,
            "label": trend_curve_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # SDG method baseline plot
    baseline_period_mean_values = np.full(
        baseline_times.shape,
        baseline_period_mean,
    )
    sdg_baseline_period_plot = tss_figure["sdg_baseline_period_plot"]
    sdg_baseline_period_plot.update(
        {
            "x": baseline_times,
            "y": baseline_period_mean_values,
            "label": sdg_baseline_period_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # SDG method target years plot
    target_years_mean_values = np.full(target_times.shape, target_years_mean)
    sdg_target_years_plot = tss_figure["sdg_target_years_plot"]
    sdg_target_years_plot.update(
        {
            "x": target_times,
            "y": target_years_mean_values,
            "label": sdg_target_years_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Robust regression line comparing baseline period and target years
    regression_span = tss_da.sel(
        time=slice(min(baseline_times), max(target_times))
    ).time.values
    regression_line_x = regression_span.astype("float")
    regression_line_slope = lkw_qltrb["tss_regression_slope"]
    regression_line_intercept = lkw_qltrb["tss_regression_intercept"]
    regression_line_scale = 1
    regression_line_y = (
        (regression_line_slope * regression_line_x) + regression_line_intercept
    ) * regression_line_scale

    regression_line_plot = tss_figure["regression_line_plot"]
    regression_line_plot.update(
        {
            "x": regression_span,
            "y": regression_line_y,
            "label": regression_line_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Title
    change_percent = ((target_years_mean / baseline_period_mean) - 1.0) * 100
    change_percent = np.round(change_percent, 2)

    tss_figure["figure_title"] = (
        f"Waterbody {waterbody_uid} - SDG : LKW_QLTRB (Turbidity)"
        "\n Annual TSS for areas of permanent water"
        f"\n Baseline period: {min(baseline_period)} to {max(baseline_period)} "
        f"Target years: {min(target_years)} to {max(target_years)}"
        f"\n TSS level change ({target_years_mean} - {baseline_period_mean}) / "
        f"{baseline_period_mean} = {change_percent} %"
        f"\nSDG 'Affected' : {lkw_qltrb['affected']} \n"
        "Significant trend based on robust regression over baseline period and "
        f"target years = {lkw_qltrb['tss_regression_significant']} \n"
        f"Decline in TSS (improvement) = {lkw_qltrb['tss_declining']}"
    )

    tss_figure["y_label"] = "TSS (mg/l)"
    tss_figure["output_file_name"] = f"{waterbody_uid}_lkw_qltrb.png"
    plot_figure(tss_figure)


def plot_change_in_trophic_state_chla(
    waterbody_uid: str,
    ds: xr.Dataset,
    baseline_period: tuple[str],
    target_years: str,
    lkw_qltrst={},
):
    chla_figure = get_figure_template()

    chla_da = ds["chla_agm_med"].mean(dim=("x", "y"))
    chla_da = chla_da.where(~np.isinf(chla_da), np.nan)

    # Data points plot
    label_prefix = "Chla values"
    data_points_plot = chla_figure["data_points_plot"]
    data_points_plot.update(
        {
            "x": chla_da.time.values,
            "y": chla_da.values,
            "label": data_points_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Trend curve plot
    # Create a two step interpolation function: a piecewise linear
    # regression followed by a quadratic interpolator
    _, fitted_y, *_ = piecewise_linreg(
        x=np.arange(0, chla_da.time.size),
        y=chla_da.values,
    )
    interp_func = sp.interpolate.interp1d(
        x=chla_da.time,
        y=fitted_y,
        kind="quadratic",
        fill_value="extrapolate",
    )
    # Generate a new, denser time range for interpolation
    time_new = pd.date_range(
        chla_da.time.min().values,
        chla_da.time.max().values,
        freq="2M",
    )
    # Fit a trend curve to the data
    baseline_slice = slice(min(baseline_period), max(baseline_period))
    baseline_times = chla_da.sel(time=baseline_slice).time.values
    baseline_period_mean = (
        chla_da.sel(time=baseline_times).mean(dim="time").item()
    )
    baseline_period_mean = np.round(baseline_period_mean, 1)

    target_slice = slice(min(target_years), max(target_years))
    target_times = chla_da.sel(time=target_slice).time.values
    target_years_mean = chla_da.sel(time=target_times).mean(dim="time").item()
    target_years_mean = np.round(target_years_mean, 1)

    clip_max = 150
    trend_curve = np.clip(
        interp_func(time_new),
        0,
        clip_max,
    )

    trend_curve_plot = chla_figure["trend_curve_plot"]
    trend_curve_plot.update(
        {
            "x": time_new,
            "y": trend_curve,
            "label": trend_curve_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # SDG method baseline plot
    baseline_period_mean_values = np.full(
        baseline_times.shape,
        baseline_period_mean,
    )
    sdg_baseline_period_plot = chla_figure["sdg_baseline_period_plot"]
    sdg_baseline_period_plot.update(
        {
            "x": baseline_times,
            "y": baseline_period_mean_values,
            "label": sdg_baseline_period_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # SDG method target years plot
    target_years_mean_values = np.full(target_times.shape, target_years_mean)
    sdg_target_years_plot = chla_figure["sdg_target_years_plot"]
    sdg_target_years_plot.update(
        {
            "x": target_times,
            "y": target_years_mean_values,
            "label": sdg_target_years_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Robust regression line comparing baseline period and target years
    regression_span = chla_da.sel(
        time=slice(min(baseline_times), max(target_times))
    ).time.values
    regression_line_x = regression_span.astype("float")
    regression_line_slope = lkw_qltrst["chla_regression_slope"]
    regression_line_intercept = lkw_qltrst["chla_regression_intercept"]
    regression_line_scale = 1
    regression_line_y = (
        (regression_line_slope * regression_line_x) + regression_line_intercept
    ) * regression_line_scale

    regression_line_plot = chla_figure["regression_line_plot"]
    regression_line_plot.update(
        {
            "x": regression_span,
            "y": regression_line_y,
            "label": regression_line_plot["label"].format(
                label_prefix=label_prefix
            ),
        },
    )

    # Title
    change_percent = ((target_years_mean / baseline_period_mean) - 1.0) * 100
    change_percent = np.round(change_percent, 2)

    chla_figure["figure_title"] = (
        f"Waterbody {waterbody_uid} - SDG : LKW_QLTRST (Trophic State)"
        "\n Annual Chla for areas of permanent water"
        f"\n Baseline period: {min(baseline_period)} to {max(baseline_period)} "
        f"Target years: {min(target_years)} to {max(target_years)}"
        f"\n Chla level change ({target_years_mean} - {baseline_period_mean}) / "
        f"{baseline_period_mean} = {change_percent} %"
        f"\nSDG 'Affected' : {lkw_qltrst['affected']} \n"
        "Significant trend based on robust regression over baseline period and "
        f"target years = {lkw_qltrst['chla_regression_significant']} \n"
        f"Decline in Chla (improvement) = {lkw_qltrst['chla_declining']}"
    )

    chla_figure["y_label"] = "Chla (mg/l)"
    chla_figure["output_file_name"] = f"{waterbody_uid}_lkw_qltrst_chla.png"
    plot_figure(chla_figure)
