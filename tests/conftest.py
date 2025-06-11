import calendar
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from odc.geo import CRS
from odc.geo.geobox import GeoBox
from odc.geo.xr import assign_crs

TEST_DATA_DIR = "tests/data"


@pytest.fixture
def sample_tile_geobox():
    shape = (22, 39)
    crs = CRS("EPSG:6933")
    affine = Affine(10.0, 0.0, 1880900.0, 0.0, 10.0, -4072040.0)
    tile_geobox = GeoBox(shape, affine, crs)
    return tile_geobox


@pytest.fixture
def build_dataset_validation_ds():
    ds = xr.open_dataset(os.path.join(TEST_DATA_DIR, "wq_agm_mutli_sensor_dataset.nc"))
    return ds


@pytest.fixture
def water_analysis_validation_ds():
    ds = xr.open_dataset(
        os.path.join(TEST_DATA_DIR, "wq_agm_mutli_sensor_dataset_water_analysis.nc")
    )
    return ds


@pytest.fixture
def pixel_corrections_validation_ds():
    ds = xr.open_dataset(
        os.path.join(TEST_DATA_DIR, "wq_agm_mutli_sensor_dataset_pixel_correction.nc")
    )
    return ds


@pytest.fixture
def hue_calc_validation_ds():
    ds = xr.open_dataset(
        os.path.join(TEST_DATA_DIR, "wq_agm_mutli_sensor_dataset_hue_calc.nc")
    )
    return ds


@pytest.fixture
def random_xr_dataset():
    def _factory(
        var_names: list[str], shape: tuple[int], seed: int | None = None
    ) -> xr.Dataset:
        dims = ["time", "y", "x"]
        time_freq = "D"
        crs = "EPSG:4326"
        crs_coord_name = "spatial_ref"

        # Create time values
        year = np.random.randint(2000, 2025)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, calendar.monthrange(year, month)[1] + 1)
        start_date = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
        time = pd.date_range(start=start_date, periods=shape[0], freq=time_freq)

        if seed is not None:
            np.random.seed(seed)

        coords = {
            "time": time,
            "y": np.linspace(-90, 90, shape[1]),
            "x": np.linspace(-180, 180, shape[2], endpoint=False),
        }
        data_vars = {}
        for var in var_names:
            data_vars[var] = (dims, np.random.rand(*shape))

        ds = xr.Dataset(data_vars, coords=coords)

        ds = assign_crs(ds, crs=crs, crs_coord_name=crs_coord_name)

        ds.attrs = {"crs": crs.lower(), "grid_mapping": crs_coord_name}
        return ds

    return _factory
