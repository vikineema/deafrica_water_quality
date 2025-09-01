"""
Create per dataset metadata (stac files) for the Water Quality Service
products.
"""

from typing import Any

from water_quality.io import find_geotiff_files, parse_wq_cog_url
from water_quality.monitoring.load_data import NORMALISATION_PARAMETERS


def dummy_product_yaml(
    product_name: str,
    dataset_path: str,
) -> dict[str, Any]:
    # Create a dummy product config with the measurements
    # specific to the dataset.
    description = "DE Africa Water Quality Service annual water quality "
    "variables for at 10m resolution - version 1."
    metadata_type = "eo3"
    license = "CC-BY-4.0"
    load = dict(crs="EPSG:6933", resolution=dict(x=10, y=-10))

    measurements = []
    measurement_paths = find_geotiff_files(dataset_path)
    for measurement_path in measurement_paths:
        measurement_info = {}
        _, _, _, band = parse_wq_cog_url(measurement_path)

        scale_and_offset = NORMALISATION_PARAMETERS.get(band, None)
        if scale_and_offset is not None:
            scale_factor = scale_and_offset["scale"]
            add_offset = scale_and_offset["offset"]
        else:
            scale_factor = 1.0
            add_offset = 0.0

        exists = any(item.get("name") == band for item in measurements)
        if exists:
            continue
        else:
            measurement_info["name"] = band
            measurement_info["dtype"] = "float32"
            measurement_info["nodata"] = "NaN"
            measurement_info["units"] = "1"
            measurement_info["scale_factor"] = scale_factor
            measurement_info["add_offset"] = add_offset

            measurements.append(measurement_info)

    measurements = sorted(measurements, key=lambda x: x["name"])
    product_config = dict(
        name=product_name,
        description=description,
        metadata_type=metadata_type,
        license=license,
        load=load,
        measurements=measurements,
    )

    return product_config
