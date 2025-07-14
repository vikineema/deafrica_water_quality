"""
Utilities for interacting with local, cloud (S3, GCS), and HTTP filesystems
"""

import json
import logging
import os
import posixpath
import re
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import xarray as xr
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from odc.aws import s3_url_parse
from odc.geo.xr import assign_crs
from s3fs.core import S3FileSystem
from tqdm import tqdm

from water_quality.tiling import get_region_code

log = logging.getLogger(__name__)


def is_s3_path(path: str) -> bool:
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, S3FileSystem)


def is_gcsfs_path(path: str) -> bool:
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, GCSFileSystem)


def is_http_url(path: str) -> bool:
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, HTTPFileSystem)


def is_local_path(path: str) -> bool:
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, LocalFileSystem)


def join_url(base, *paths) -> str:
    if is_local_path(base):
        return os.path.join(base, *paths)
    else:
        # Ensure urls join correctly
        return posixpath.join(base, *paths)


def split_path(path: str):
    if is_local_path(path):
        return os.path.split(path)
    else:
        return posixpath.split(path)


def get_basename(path: str):
    if is_local_path(path):
        return os.path.basename(path)
    else:
        return posixpath.basename(path)


def get_filesystem(
    path: str,
    anon: bool = True,
) -> S3FileSystem | LocalFileSystem | GCSFileSystem:
    if is_s3_path(path=path):
        fs = S3FileSystem(
            anon=anon,
            # Use profile only on sandbox
            # profile="default",
            s3_additional_kwargs={"ACL": "bucket-owner-full-control"},
        )
    elif is_gcsfs_path(path=path):
        if anon:
            fs = GCSFileSystem(token="anon")
        else:
            fs = GCSFileSystem()
    elif is_http_url(path):
        fs = HTTPFileSystem()
    elif is_local_path(path=path):
        fs = LocalFileSystem()
    return fs


def check_file_exists(path: str) -> bool:
    fs = get_filesystem(path=path, anon=True)
    if fs.exists(path) and fs.isfile(path):
        return True
    else:
        return False


def check_directory_exists(path: str) -> bool:
    fs = get_filesystem(path=path, anon=True)
    if fs.exists(path) and fs.isdir(path):
        return True
    else:
        return False


def check_file_extension(
    path: str, accepted_file_extensions: list[str]
) -> bool:
    _, file_extension = os.path.splitext(path)
    if file_extension.lower() in accepted_file_extensions:
        return True
    else:
        return False


def is_geotiff(path: str) -> bool:
    accepted_geotiff_extensions = [".tif", ".tiff", ".gtiff"]
    return check_file_extension(
        path=path, accepted_file_extensions=accepted_geotiff_extensions
    )


def find_geotiff_files(
    directory_path: str, file_name_pattern: str = ".*"
) -> list[str]:
    file_name_pattern = re.compile(file_name_pattern)

    fs = get_filesystem(path=directory_path, anon=True)

    geotiff_file_paths = []

    for root, dirs, files in fs.walk(directory_path):
        for file_name in files:
            if is_geotiff(path=file_name):
                if re.search(file_name_pattern, file_name):
                    geotiff_file_paths.append(os.path.join(root, file_name))
                else:
                    continue
            else:
                continue

    if is_s3_path(path=directory_path):
        geotiff_file_paths = [f"s3://{file}" for file in geotiff_file_paths]
    elif is_gcsfs_path(path=directory_path):
        geotiff_file_paths = [f"gs://{file}" for file in geotiff_file_paths]
    return geotiff_file_paths


def is_json(path: str) -> bool:
    accepted_json_extensions = [".json"]
    return check_file_extension(
        path=path, accepted_file_extensions=accepted_json_extensions
    )


def find_json_files(
    directory_path: str, file_name_pattern: str = ".*"
) -> list[str]:
    file_name_pattern = re.compile(file_name_pattern)

    fs = get_filesystem(path=directory_path, anon=True)

    json_file_paths = []

    for root, dirs, files in fs.walk(directory_path):
        for file_name in files:
            if is_json(path=file_name):
                if re.search(file_name_pattern, file_name):
                    json_file_paths.append(os.path.join(root, file_name))
                else:
                    continue
            else:
                continue

    if is_s3_path(path=directory_path):
        json_file_paths = [f"s3://{file}" for file in json_file_paths]
    elif is_gcsfs_path(path=directory_path):
        json_file_paths = [f"gs://{file}" for file in json_file_paths]
    return json_file_paths


def _get_wq_parent_dir(
    output_directory: str,
    tile_id: tuple[int, int],
    year: int,
) -> str:
    # output_dir/x/y/year/file_name
    region_code = get_region_code(tile_id, sep="/")
    year = str(year)
    parent_dir = join_url(output_directory, region_code, str(year))
    if not check_directory_exists(parent_dir):
        fs = get_filesystem(parent_dir, anon=False)
        fs.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def get_wq_cog_url(
    output_directory: str, tile_id: tuple[int, int], year: int, band_name: str
):
    parent_dir = _get_wq_parent_dir(output_directory, tile_id, year)

    # f"{band}_{region_code}_{year}.tif"
    region_code = get_region_code(tile_id, sep="")
    file_name = f"{band_name}_{region_code}_{year}.tif"
    cog_url = join_url(parent_dir, file_name)
    return cog_url


def parse_wq_cog_url(cog_url: str):
    # f"{band}_{region_code}_{year}.tif"
    base = get_basename(cog_url)
    base = os.path.splitext(base)[0]
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError("Filename does not contain enough parts")

    band = "_".join(parts[:-2])
    region_code = parts[-2]
    year = parts[-1]
    return band, region_code, year


def get_wq_csv_url(output_directory: str, tile_id: tuple[int, int], year: int):
    parent_dir = _get_wq_parent_dir(output_directory, tile_id, year)

    region_code = get_region_code(tile_id, sep="")
    file_name = f"water_quality_measures_{region_code}_{year}.csv"
    csv_url = join_url(parent_dir, file_name)
    return csv_url
