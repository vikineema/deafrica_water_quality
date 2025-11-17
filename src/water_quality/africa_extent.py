"""
Africa extent in various formats.
"""

import geopandas as gpd
from odc.geo.geom import Geometry

# GDAL format: [ulx, uly, lrx, lry]
AFRICA_BBOX = [-26.36, 38.35, 64.50, -47.97]
AFRICA_EXTENT_BBOX_URL = "https://raw.githubusercontent.com/digitalearthafrica/deafrica-extent/master/africa-extent-bbox.json"
AFRICA_EXTENT_URL = "https://raw.githubusercontent.com/digitalearthafrica/deafrica-extent/refs/heads/master/africa-extent.json"


def africa_extent_geometry():
    """Load Africa extent as a Geometry object."""
    africa_extent = gpd.read_file(AFRICA_EXTENT_URL)
    africa_extent_geom = Geometry(
        geom=africa_extent.iloc[0].geometry, crs=africa_extent.crs
    )
    return africa_extent_geom
