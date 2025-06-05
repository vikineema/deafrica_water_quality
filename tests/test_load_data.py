import pytest
import xarray as xr

from water_quality.load_data import (
    build_dc_queries,
    build_wq_dataset,
    get_dc_measurements,
    get_dc_products,
    get_measurements_name_dict,
)


def test_get_product_for_invalid_instrument():
    instrument_name = "msi"
    with pytest.raises(NotImplementedError):
        get_dc_products(instrument_name)


def test_get_product_for_valid_instrument():
    instrument_name = "tm_agm"
    expected_results = ["gm_ls5_ls7_annual"]
    assert get_dc_products(instrument_name) == expected_results


def test_get_measurements_for_invalid_instrument():
    instrument_name = "oli"
    with pytest.raises(NotImplementedError):
        get_dc_measurements(instrument_name)


def test_get_measurements_for_valid_instrument():
    instrument_name = "wofs_all"
    expected_results = sorted(["frequency", "count_clear", "count_wet"])
    assert sorted(get_dc_measurements(instrument_name)) == expected_results


def test_get_measurements_name_dict_for_invalid_instrument():
    instrument_name = "tm"
    with pytest.raises(NotImplementedError):
        get_measurements_name_dict(instrument_name)


def test_get_measurements_name_dict_for_valid_instrument():
    instrument_name = "oli_agm"
    expected_results = {
        "SR_B2": "oli02_agm",
        "SR_B3": "oli03_agm",
        "SR_B4": "oli04_agm",
        "SR_B5": "oli05_agm",
        "SR_B6": "oli06_agm",
        "SR_B7": "oli07_agm",
        "smad": "oli_agm_smad",
        "count": "oli_agm_count",
    }
    assert get_measurements_name_dict(instrument_name) == expected_results


def test_build_dc_queries_single_instrument(sample_tile_geobox):
    instruments_to_use = {"wofs_all": {"use": True}}

    start_date = "2024"
    end_date = "2024"
    resampling = "bilinear"

    expected_result = {
        "wofs_all": {
            "product": ["wofs_ls_summary_alltime"],
            "measurements": ["frequency", "count_clear", "count_wet"],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        }
    }
    result = build_dc_queries(
        instruments_to_use, sample_tile_geobox, start_date, end_date, resampling
    )
    assert expected_result == result


def test_build_dc_queries_multi_instrument(sample_tile_geobox):
    instruments_to_use = {
        "msi_agm": {"use": True},
        "oli_agm": {"use": True},
        "wofs_ann": {"use": True},
        "wofs_all": {"use": True},
    }
    start_date = "2024"
    end_date = "2024"
    resampling = "bilinear"

    expected_results = {
        "oli_agm": {
            "product": ["gm_ls8_annual", "gm_ls8_ls9_annual"],
            "measurements": [
                "SR_B2",
                "SR_B3",
                "SR_B4",
                "SR_B5",
                "SR_B6",
                "SR_B7",
                "smad",
                "count",
            ],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
        "msi_agm": {
            "product": ["gm_s2_annual"],
            "measurements": [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B12",
                "smad",
                "count",
            ],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
        "wofs_ann": {
            "product": ["wofs_ls_summary_annual"],
            "measurements": ["frequency", "count_clear", "count_wet"],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
        "wofs_all": {
            "product": ["wofs_ls_summary_alltime"],
            "measurements": ["frequency", "count_clear", "count_wet"],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
    }
    results = build_dc_queries(
        instruments_to_use, sample_tile_geobox, start_date, end_date, resampling
    )

    expected_instruments = ["oli_agm", "msi_agm", "wofs_ann", "wofs_all"]

    assert sorted(results.keys()) == sorted(expected_instruments)

    assert expected_results == results


def test_build_wq_dataset_single_instrument(
    sample_tile_geobox, sample_single_instrument_wq_dataset
):
    dc_queries = {
        "wofs_ann": {
            "product": ["wofs_ls_summary_annual"],
            "measurements": ["frequency", "count_clear", "count_wet"],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
    }
    ds = build_wq_dataset(dc_queries)

    xr.testing.assert_equal(ds, sample_single_instrument_wq_dataset)


def test_build_wq_dataset_multi_instrument(
    sample_tile_geobox, sample_multi_instrument_wq_dataset
):
    dc_queries = {
        "oli_agm": {
            "product": ["gm_ls8_annual", "gm_ls8_ls9_annual"],
            "measurements": [
                "SR_B2",
                "SR_B3",
                "SR_B4",
                "SR_B5",
                "SR_B6",
                "SR_B7",
                "smad",
                "count",
            ],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
        "msi_agm": {
            "product": ["gm_s2_annual"],
            "measurements": [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B12",
                "smad",
                "count",
            ],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
        "wofs_ann": {
            "product": ["wofs_ls_summary_annual"],
            "measurements": ["frequency", "count_clear", "count_wet"],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
        "wofs_all": {
            "product": ["wofs_ls_summary_alltime"],
            "measurements": ["frequency", "count_clear", "count_wet"],
            "like": sample_tile_geobox,
            "time": ("2024", "2024"),
            "resampling": "bilinear",
        },
    }
    ds = build_wq_dataset(dc_queries)

    xr.testing.assert_equal(ds, sample_multi_instrument_wq_dataset)
