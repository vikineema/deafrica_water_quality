import pytest

from water_quality.datasets import (
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
