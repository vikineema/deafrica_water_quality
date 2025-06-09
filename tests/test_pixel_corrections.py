import pytest
import xarray as xr

from water_quality.pixel_corrections import R_correction

TEST_DP_ADJUST = {
    "msi_agm": {
        "ref_var": "msi12_agm",
        "var_list": [
            "msi04_agm",
            "msi03_agm",
            "msi02_agm",
            "msi05_agm",
            "msi06_agm",
            "msi07_agm",
        ],
    },
    "oli_agm": {
        "ref_var": "oli07_agm",
        "var_list": ["oli04_agm", "oli03_agm", "oli02_agm"],
    },
    "tm_agm": {
        "ref_var": "tm07_agm",
        "var_list": ["tm04_agm", "tm03_agm", "tm02_agm", "tm01_agm"],
    },
}

TEST_INSTRUMENTS_TO_USE = {
    "oli_agm": {"use": True},
    "oli": {"use": False},
    "msi_agm": {"use": True},
    "msi": {"use": False},
    "tm_agm": {"use": False},
    "tm": {"use": False},
    "tirs": {"use": False},
    "wofs_ann": {"use": True},
    "wofs_all": {"use": True},
}


def test_r_correction_missing_instrument(sample_multi_instrument_wq_dataset):
    dp_adjust = TEST_DP_ADJUST
    instruments_to_use = {
        k: v for k, v in TEST_INSTRUMENTS_TO_USE.items() if k != "oli_agm"
    }
    ds = sample_multi_instrument_wq_dataset

    with pytest.raises(ValueError):
        results = R_correction(ds, dp_adjust, instruments_to_use)


def test_r_correction_instrument_not_used(sample_multi_instrument_wq_dataset):
    dp_adjust = TEST_DP_ADJUST
    instruments_to_use = {
        k: ({"use": False} if k == "oli_agm" else v)
        for k, v in TEST_INSTRUMENTS_TO_USE.items()
    }
    ds = sample_multi_instrument_wq_dataset

    with pytest.raises(ValueError):
        results = R_correction(ds, dp_adjust, instruments_to_use)


def test_r_correction_missing_ref_var(sample_multi_instrument_wq_dataset):
    dp_adjust = TEST_DP_ADJUST
    instruments_to_use = TEST_INSTRUMENTS_TO_USE
    ds = sample_multi_instrument_wq_dataset.drop_vars("oli07_agm")
    with pytest.raises(ValueError):
        results = R_correction(ds, dp_adjust, instruments_to_use)


def test_r_correction_missing_target_var(sample_multi_instrument_wq_dataset):
    dp_adjust = TEST_DP_ADJUST
    instruments_to_use = TEST_INSTRUMENTS_TO_USE
    ds = sample_multi_instrument_wq_dataset.drop_vars("msi05_agm")
    with pytest.raises(ValueError):
        results = R_correction(ds, dp_adjust, instruments_to_use)


def test_r_correction_valid(sample_multi_instrument_wq_dataset):
    dp_adjust = {k: v for k, v in TEST_DP_ADJUST.items() if k != "tm_agm"}
    instruments_to_use = TEST_INSTRUMENTS_TO_USE
    ds = sample_multi_instrument_wq_dataset

    results = R_correction(ds, dp_adjust, instruments_to_use)

    expected_added_vars = [f"{i}r" for k, v in dp_adjust.items() for i in v["var_list"]]
    assert all(item in list(results.data_vars) for item in expected_added_vars)
