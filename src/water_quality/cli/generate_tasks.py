import json
import random
from collections import defaultdict, namedtuple
from datetime import datetime
from itertools import chain, islice
from types import SimpleNamespace
from typing import Any

import click
import geopandas as gpd
import toolz
from datacube import Datacube
from datacube.model import Dataset
from datacube.utils.dates import normalise_dt
from dateutil.relativedelta import relativedelta
from odc import dscache
from odc.dscache import DatasetCache
from odc.dscache.tools import bin_dataset_stream, ordered_dss
from odc.dscache.tools.profiling import ds_stream_test_func
from odc.geo import XY, Resolution
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.gridspec import GridSpec
from odc.stats._cli_common import click_yaml_cfg
from odc.stats._gjson import compute_grid_info, gjson_from_tasks, gs_bounds
from odc.stats.model import DateTimeRange
from odc.stats.utils import _rolling_tasks, bin_annual, rolling_season_binner

from water_quality.africa_extent import AFRICA_EXTENT_URL
from water_quality.grid import get_waterbodies_grid
from water_quality.logs import setup_logging
from water_quality.mapping.config import check_config
from water_quality.mapping.instruments import check_instrument_dates
from water_quality.mapping.load_data import INSTRUMENTS_PRODUCTS

SUPPORTED_FREQUENCY = [
    "annual",
]

CompressedDataset = namedtuple(
    "CompressedDataset", ["id", "time", "instrument"]
)
dt_range = SimpleNamespace(start=None, end=None)

Cell = Any


def compress_ds(ds: Dataset) -> CompressedDataset:
    """
    Reduce a datacube dataset into the required components

    Parameters
    ----------
    ds : Dataset
        Dataset streamed from the datacube.

    Returns
    -------
    CompressedDataset
        Compressed representation of the dataset.
    """
    input_products = {
        v: k for k, vals in INSTRUMENTS_PRODUCTS.items() for v in vals
    }
    dt = normalise_dt(ds.center_time)
    instrument = input_products[ds.product.name]
    return CompressedDataset(ds.id, dt, instrument)


def update_start_end(x: datetime, out: SimpleNamespace):
    """
    Add a start and end datetime to a CompressedDataset.

    Parameters
    ----------
    x : datetime
        Datetime or date range to use.
    out : SimpleNamespace
        An empty simplenamespace object to fill.
    """
    if out.start is None:
        out.start = x
        out.end = x
    else:
        out.start = min(out.start, x)
        out.end = max(out.end, x)


def persist(ds: Dataset) -> CompressedDataset:
    """
    Mapping function to use when binning a dataset stream.
    """
    # Convert the dataset to a CompressedDataset.
    _ds = compress_ds(ds)
    # Add a start and end datetime to the CompressedDataset.
    update_start_end(_ds.time, dt_range)
    return _ds


def bin_by_instrument(
    cells: dict[tuple[int, int], Cell],
) -> dict[tuple[str, int, int], Cell]:
    """
    Bin the datasets in each cell by instrument.

    Parameters
    ----------
    cells : dict[tuple[int, int], Cell]
        The ``cells`` dictionary is a mapping from (x,y) tile index to
        object with the following properties:
         - .idx     - tile index (x,y)
         - .geobox  - tile geobox
         - .utc_offset - timedelta to add to timestamp to get day component
                        in local time
         - .dss - list of UUIDs, or results of ``persist(dataset)``
            if custom ``persist`` was supplied to `bin_dataset_stream`

    Returns
    -------
    dict[tuple[str, int, int], Cell]
        The updated ``cells`` dictionary is a mapping from (x,y) tile
        index and instrument to object with the following properties:
         - .idx     - tile index (x,y) and instrument
         - .geobox  - tile geobox
         - .utc_offset - timedelta to add to timestamp to get day component
                        in local time
         - .dss - list of UUIDs, or results of ``persist(dataset)``
            if custom ``persist`` was supplied to `bin_dataset_stream`

    """
    output_cells = {}
    for tile_index, cell in cells.items():
        grouped_by_instrument = toolz.groupby(
            lambda ds: ds.instrument, cell.dss
        )
        for instrument, dss in grouped_by_instrument.items():
            instrument_key = (instrument,)
            output_cell_index = instrument_key + tile_index
            output_cells[output_cell_index] = SimpleNamespace(
                geobox=cell.geobox,
                idx=output_cell_index,
                utc_offset=cell.utc_offset,
                dss=dss,
            )
    return output_cells


def mk_wofs_ann_5yr_rules(
    temporal_range: DateTimeRange,
) -> dict[str, DateTimeRange]:
    """
    Construct rules for binning wofs_ann instrument data, with each year
    containing the specified year's data plus data for the
    previous 4 years.

    Parameters
    ----------
    temporal_range : DateTimeRange
        Time range for which datasets have been loaded.

    Returns
    -------
    dict[str, DateTimeRange]
        Rules mapping time ranges for wofs_ann datasets to the year
        to be binned into.
    """
    # As per product specification
    # the Water Quality service will be run for 2000-present
    # meaning there will always be 5 years worth of historical data
    # available for a year hence no exceptions have been made
    # here for the start date of the temporal range.
    start_date = temporal_range.start
    end_date = temporal_range.end

    rules = {}
    season_start = start_date

    while (
        DateTimeRange(f"{season_start.strftime('%Y-%m-%d')}--P1Y").end
        <= end_date
    ):
        five_year_season_start = season_start - relativedelta(years=4)
        rules[f"{season_start.strftime('%Y')}--P1Y"] = DateTimeRange(
            f"{five_year_season_start.strftime('%Y-%m-%d')}--P5Y"
        )
        season_start += relativedelta(years=1)

    return rules


def bin_5yr_wofs_ann(
    cells: dict[tuple[str, int, int], Cell],
    temporal_range: DateTimeRange,
) -> dict[tuple[str, int, int], list[CompressedDataset]]:
    """
    Bin wofs_ann instrument datasets in each cell.

    Parameters
    ----------
    cells : dict[tuple[str, int, int], Cell]
        Cells dictionary containing datasets for the wofs_ann instrument.
    temporal_range : DateTimeRange
        Temporal range for which datasets have been loaded.

    Returns
    -------
    dict[tuple[str, int, int], list[CompressedDataset]]
        Mapping of instrument and tile index to list of CompressedDatasets.
    """
    binner = rolling_season_binner(mk_wofs_ann_5yr_rules(temporal_range))

    return _rolling_tasks(cells, binner)


def write_tasks_to_csv(
    csv_path: str,
    tasks: dict[tuple[str, int, int], set[CompressedDataset]],
):
    """
    Write a summary of the tasks to a csv file.

    Parameters
    ----------
    csv_path : str
        Path to write the csv file to.
    tasks : dict[tuple[str, int, int], set[CompressedDataset]]
        Tasks to summarise.
    """
    with open(csv_path, "w", encoding="utf8") as f:
        f.write('"T","X","Y","datasets","days"\n')

        for p, x, y in sorted(tasks):
            dss = tasks[(p, x, y)]
            n_dss = len(dss)
            n_days = len({ds.time.date() for ds in dss})
            line = f'"{p}", {x:+05d}, {y:+05d}, {n_dss:4d}, {n_days:4d}\n'
            f.write(line)
            # TODO: log the write


def write_tasks_to_geojson(
    file_name_prefix: str,
    gridspec: GridSpec,
    cells: dict[tuple[int, int], Cell],
    tasks: dict[tuple[str, int, int], set[CompressedDataset]],
):
    grid_info = compute_grid_info(
        cells, resolution=max(gridspec.tile_size.xy) / 4
    )
    tasks_geo = gjson_from_tasks(tasks, grid_info)
    for temporal_range, gjson in tasks_geo.items():
        fname = f"{file_name_prefix}-{temporal_range}.geojson"
        with open(fname, "w", encoding="utf8") as f:
            json.dump(gjson, f)
            # TODO: log the write


@click.command(
    name="save-tasks",
    no_args_is_help=True,
)
@click_yaml_cfg(
    "--analysis-config",
    required=True,
    help="Config for the analysis parameters in yaml format, file or text",
)
@click.argument(
    "temporal-range",
    type=str,
)
@click.argument(
    "frequency",
)
def cli(
    analysis_config: dict,
    temporal_range: str,
    frequency: str,
):
    """
    # TODO: Add description
    """
    log = setup_logging()

    # Verify inputs ------
    try:
        temporal_range = DateTimeRange(temporal_range)
    except ValueError:
        e = ValueError(
            f"Failed to parse supplied temporal_range: '{temporal_range}'",
        )
        log.error(e)
        raise e

    if frequency not in SUPPORTED_FREQUENCY:
        e = ValueError(
            f"Frequency must be one of {'|'.join(SUPPORTED_FREQUENCY)} and not "
            f"{frequency}"
        )
        log.error(e)
        raise e

    analysis_config = check_config(analysis_config)

    instruments_to_use = analysis_config["instruments_to_use"]
    instruments_to_use = check_instrument_dates(
        instruments_to_use,
        temporal_range.start.strftime("%Y-%m-%d"),
        temporal_range.end.strftime("%Y-%m-%d"),
    )

    # Define prerequisites
    FILE_NAME_PREFIX = f"wq_{frequency}_{temporal_range.short}_"

    cache_db_fp = f"{FILE_NAME_PREFIX}_cache.db"
    tasks_csv_fp = f"{FILE_NAME_PREFIX}_tasks.csv"
    grid_name = "water_quality_grid"

    dc = Datacube(app="FindDatasets")

    # Find datasets for each instrument ----

    resolution = analysis_config["resolution"]
    grid_spec = get_waterbodies_grid(resolution)
    africa_extent = gpd.read_file(AFRICA_EXTENT_URL)
    africa_extent_geom = Geometry(
        geom=africa_extent.iloc[0].geometry, crs=africa_extent.crs
    )
    africa_geobox = GeoBox.from_geopolygon(
        africa_extent_geom, resolution=grid_spec.resolution, crs=grid_spec.crs
    )
    query = dict(
        like=africa_geobox, time=(temporal_range.start, temporal_range.end)
    )

    # zstandard compression dictionary
    # for dataset cache
    zdict = None

    datasets_list = []
    log.info("Connecting to the datacube and streaming datasets")
    for inst in instruments_to_use.keys():
        if instruments_to_use[inst]["use"]:
            input_products = INSTRUMENTS_PRODUCTS[inst]
            log.info(f"Streaming datasets for the instrument: {inst}")
            # wofs_ann instrument (wofs_ls_summary_annual ODC product)
            # requires data starting 4 years before the temporal
            # range start date due to the 5 year water mask generated
            # when processing the annual water quality variables.
            if inst == "wofs_ann":
                time_range = query["time"]
                time_range = (
                    time_range[0] - relativedelta(years=4),
                    time_range[1],
                )

                dc_query = {}
                for k, v in query.items():
                    if k != "time":
                        dc_query[k] = v
                    dc_query.update(
                        {"time": time_range, "product": input_products}
                    )
            else:
                dc_query = {"product": input_products, **query}

            datasets = ordered_dss(
                dc,
                freq="Y",
                key=lambda ds: (
                    (ds.center_time, ds.metadata.region_code)
                    if hasattr(ds.metadata, "region_code")
                    else (ds.center_time,)
                ),
                **dc_query,
            )
            datasets_slice = list(islice(datasets, 0, 100))
            if len(datasets_slice) == 0:
                log.info(f"Found no datasets for the instrument: {inst}")
            else:
                if zdict is None and len(datasets_slice) >= 100:
                    log.info("Train compression dictionary")
                    samples = datasets_slice.copy()
                    random.shuffle(samples)
                    zdict = DatasetCache.train_dictionary(samples, 8 * 1024)
                    log.info("Training complete.")

                datasets = chain(datasets_slice, datasets)
                datasets_list.append(datasets)

    dss = chain(*datasets_list)

    cache = dscache.create_cache(
        path=cache_db_fp, complevel=6, zdict=zdict, truncate=True
    )
    cache.add_grid(grid_spec, grid_name)

    dss = cache.tee(dss)

    cells = {}
    dss = bin_dataset_stream(grid_spec, dss, cells, persist=persist)
    rr = ds_stream_test_func(dss)
    log.info(rr.text)

    # TODO: Add filtering by tile id
    # prune out tiles that were not requested

    # TODO: Add filtering by temporal range
    # Prune Datasets outside of temporal range (after correcting for UTC offset)
    log.info(f"Total of {len(cells):,d} spatial tiles")

    # Not rewriting cells because original `cells` is needed during
    # writing tasks to geojson.
    updated_cells = bin_by_instrument(cells)

    # Cells are split here based on whether they contain wofs_ann datasets
    # or not. This is because binning for wofs_ann data is different from
    # all other instruments due to the temporal range adjustment needed for
    # wofs_ann datasets for the water mask generation process.
    not_wofs_ann_cells = {
        k: v for k, v in updated_cells.items() if "wofs_ann" not in k
    }
    wofs_ann_cells = {
        k: v for k, v in updated_cells.items() if "wofs_ann" in k
    }

    wofs_ann_tasks = bin_5yr_wofs_ann(
        wofs_ann_cells, temporal_range=temporal_range
    )
    not_wofs_ann_tasks = bin_annual(not_wofs_ann_cells)

    # Recombine tasks without instruments
    tasks = defaultdict(list)
    for d in (wofs_ann_tasks, not_wofs_ann_tasks):
        for (p, inst, x, y), dss in d.items():
            tasks[(p, x, y)].extend(dss)

    # Remove duplicate source uids
    # Duplicates occur when queried datasets are captured around UTC midnight
    # and around weekly boundary
    tasks = {k: set(dss) for k, dss in tasks.items()}
    tasks_uuid = {k: [ds.id for ds in dss] for k, dss in tasks.items()}

    log.info(f"Saving tasks to disk ({len(tasks)})")
    cache.add_grid_tiles(grid_name, tasks_uuid)

    write_tasks_to_csv(tasks_csv_fp, tasks)
    log.info(f"Wrote tasks summary to: {tasks_csv_fp}")

    write_tasks_to_geojson(FILE_NAME_PREFIX, grid_spec, cells, tasks)
