import os

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo import CRS
from odc.geo.geobox import GeoBox

TEST_DATA_DIR = "tests/data"


@pytest.fixture
def sample_tile_geobox():
    shape = (8, 14)
    crs = CRS("EPSG:6933")
    affine = Affine(30.0, 0.0, 1880880.0, 0.0, -30.0, -4071810.0)
    tile_geobox = GeoBox(shape, affine, crs)
    return tile_geobox


@pytest.fixture
def sample_multi_instrument_wq_dataset():
    ds = xr.open_dataset(os.path.join(TEST_DATA_DIR, "wq_dataset_multi_instrument.nc"))
    return ds


@pytest.fixture
def sample_single_instrument_wq_dataset():
    ds = xr.open_dataset(os.path.join(TEST_DATA_DIR, "wq_dataset_single_instrument.nc"))
    return ds
