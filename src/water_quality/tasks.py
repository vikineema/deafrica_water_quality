"""
This module provides functions to create tasks used as
as building blocks in the DE Africa Water Quality workflow.
"""

from odc.stats._text import split_and_check

from water_quality.tiling import get_region_code, parse_region_code


def create_task_id(temporal_id: str, tile_id: tuple[int, int] | str) -> str:
    """Create  a task given a temporal range and a tile id.

    Parameters
    ----------
    temporal_id : str
        Temproal range to create the task for.
    tile_id : tuple[int, int] | str
        Tile ID for the tile to create the task for.

    Returns
    -------
    str
        Task ID
    """
    # task id format "{temporal_id}/x{x:03d}/y{y:03d}"
    region_code = get_region_code(tile_id, sep="/")
    task_id = f"{temporal_id}/{region_code}"
    return task_id


def parse_task_id(task_id: str) -> tuple[str, tuple[int, int]]:
    """
    Parse a task ID into the temporal ID and tile ID it was created from.

    Parameters
    ----------
    task_id : str
        Task ID to parse.

    Returns
    -------
    tuple[str, tuple[int, int]]
        Temporal ID and tile ID components of the task.
    """
    # Check Task id has only 3 parts
    sep = "/"  # based on seperator used in create_task_id
    parts = split_and_check(task_id, sep, 3)

    # Get the tile ID
    tile_id = parse_region_code(task_id)

    # Get the temporal ID
    temporal_id = [p for p in parts if not p.startswith(("x", "y"))]
    # Should always be one item in the temporal_id.
    temporal_id = temporal_id[0]
    return temporal_id, tile_id
