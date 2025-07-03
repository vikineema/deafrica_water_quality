import pytest
import xarray as xr

from water_quality.hue import hue_calculation


def test_hue_oli_agm_instrument(pixel_corrections_validation_ds):
    with pytest.raises(NotImplementedError):
        results = hue_calculation(
            pixel_corrections_validation_ds, instrument="oli_agm"
        )


def test_hue_calculation_msi_instrument(pixel_corrections_validation_ds):
    with pytest.raises(KeyError):
        results = hue_calculation(
            pixel_corrections_validation_ds, instrument="msi"
        )


def test_hue_calculation_msi_agm_instrument(
    pixel_corrections_validation_ds, hue_calc_validation_ds
):
    results = hue_calculation(
        pixel_corrections_validation_ds, instrument="msi_agm"
    )

    xr.testing.assert_allclose(results, hue_calc_validation_ds["hue"])
