"""
Create per dataset metadata (stac files) for the Water Quality Service
products.
"""

import logging
from pathlib import PurePath
from typing import Any

import toolz
import yaml
from eodatasets3.model import DatasetDoc
from yarl import URL

from water_quality.io import (
    find_geotiff_files,
    get_parent_dir,
    is_local_path,
    parse_wq_cog_url,
)
from water_quality.mapping.algorithms import NORMALISATION_PARAMETERS

log = logging.getLogger(__name__)


def generate_annual_product_template(tile_directory: str) -> dict[str, Any]:
    """
    Generate a ODC product template file for the annual water
    quality variables, based on all the data available for the years
    2000-2024.

    Parameters
    ----------
    tile_directory : str
        S3 or local directory containing all the COG files for a single
        tile across multiple years.

    Returns
    -------
    dict[str, Any]
        ODC product template for the product wqs_annual_v1 product.
    """
    all_tile_geotifs = find_geotiff_files(tile_directory)

    datasets = toolz.groupby(lambda f: get_parent_dir(f), all_tile_geotifs)

    # Compile a list of unique measurements and their metadata
    #  across the years of data available for the tile.
    measurements = []
    for dataset_path, dataset_measurements in datasets.items():
        for measurement_path in dataset_measurements:
            measurement_info = {}
            product_name, region_code, temporal_id, band = parse_wq_cog_url(
                measurement_path
            )

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
    # Create the product config.
    product_name = "wqs_annual"
    description = "DE Africa Water Quality Service annual water quality variables at 10m resolution - version 1."
    metadata_type = "eo3"
    license = "CC-BY-4.0"
    load = dict(crs="EPSG:6933", resolution=dict(x=10, y=-10))

    product_config = dict(
        name=product_name,
        description=description,
        metadata_type=metadata_type,
        license=license,
        load=load,
        measurements=measurements,
    )

    output_file = "wqs_annual.odc-product.yaml"
    with open(output_file, "w") as file:
        yaml.dump(product_config, file, sort_keys=False)
    log.info(f"{product_name} ODC product template written to {output_file} ")

    return product_config


def get_dummy_product_yaml(
    dataset_path: str,
) -> dict[str, Any]:
    measurements = []
    dataset_measurements = find_geotiff_files(dataset_path)
    for measurement_path in dataset_measurements:
        measurement_info = {}
        product_name, region_code, temporal_id, band = parse_wq_cog_url(
            measurement_path
        )

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

    # Create a dummy product config with the measurements
    # specific to a dataset.
    description = f"DE Africa Water Quality Service {product_name}."
    metadata_type = "eo3"
    license = "CC-BY-4.0"
    load = dict(crs="EPSG:6933", resolution=dict(x=10, y=-10))

    product_config = dict(
        name=product_name,
        description=description,
        metadata_type=metadata_type,
        license=license,
        load=load,
        measurements=measurements,
    )

    return product_config


def get_dataset_tile_id(dataset_path: str):
    """
    Get a unique tile id given a dataset path.
    Expected dataset path is based on `_get_wq_parent_dir`
    in the `io` module.
    """
    if is_local_path(dataset_path):
        (
            product_name,
            product_version_dashed,
            region_code_x,
            region_code_y,
            temporal_id,
        ) = PurePath(dataset_path).parts[-5:]
    else:
        (
            product_name,
            product_version_dashed,
            region_code_x,
            region_code_y,
            temporal_id,
        ) = URL(dataset_path).path.strip("/").split("/")[-5:]

    dataset_tile_id = f"{product_name}_{product_version_dashed}_{region_code_x}{region_code_y}_{temporal_id}"
    return dataset_tile_id


def prepare_dataset(
    dataset_path: str,
    # product_definition: str,
    output_path: str,
) -> DatasetDoc:
    """Prepare an eo3 metadata file for a data product.

    Parameters
    ----------
    tile_id : str
        Unique tile ID for a single dataset to prepare.
    dataset_path : str
        Directory of the datasets
    product_definition : str
        Path to the product definition yaml file.
    output_path : str
        Path to write the output eo3 metadata file.

    Returns
    -------
    DatasetDoc
        eo3 metadata document
    """
    tile_id = get_dataset_tile_id(dataset_path)

    ## Initialise and validate inputs
    # Creates variables (see EasiPrepare for others):
    # - p.dataset_path
    # - p.product_name
    product_definition = get_dummy_product_yaml(dataset_path)
