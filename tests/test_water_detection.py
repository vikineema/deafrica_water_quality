import pytest
import xarray as xr

from water_quality.water_detection import water_analysis


def test_water_detection_invalid_wofs_varname(random_xr_dataset):
    ds = random_xr_dataset(var_names=["test"], shape=(1, 2, 3))
    with pytest.raises(ValueError):
        water_analysis(ds, wofs_varname="wofs_ann_freq")


def test_water_detection_wofs_all_freq_not_implemented(random_xr_dataset):
    ds = random_xr_dataset(var_names=["wofs_all_freq"], shape=(1, 2, 3))
    with pytest.raises(NotImplementedError):
        water_analysis(ds, wofs_varname="wofs_all_freq")


def test_water_detection_on_valid_ds(
    build_dataset_validation_ds, water_analysis_validation_ds
):
    expected_results = water_analysis_validation_ds
    results = water_analysis(build_dataset_validation_ds, wofs_varname="wofs_ann_freq")
    expected_added_vars = [
        "wofs_ann_freq_sigma",
        "wofs_ann_confidence",
        "wofs_pw_threshold",
        "wofs_ann_pwater",
        "wofs_ann_water",
        "watermask",
    ]
    assert all(item in list(results.data_vars) for item in expected_added_vars)
    xr.testing.assert_equal(results, expected_results)
