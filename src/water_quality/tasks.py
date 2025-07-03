from water_quality.tiling import get_region_code, parse_region_code


def create_task_id(year: str | int, tile_id: tuple[int, int] | str) -> str:
    """Create  a task given a year and a tile id.

    Parameters
    ----------
    year : str | int
        Year to create the task for.
    tile_id : tuple[int, int] | str
        Tile ID for the tile to create the task for.

    Returns
    -------
    str
        Task ID
    """
    if isinstance(year, int):
        year = str(year)
    # task id format "{year}/x{x:02d}/y{y:02d}"
    region_code = get_region_code(tile_id, sep="/")
    task_id = f"{year}/{region_code}"
    return task_id


def parse_task_id(task_id: str) -> tuple[int, tuple[int, int]]:
    """
    Parse a task ID into the year and tile ID it was created from.

    Parameters
    ----------
    task_id : str
        Task ID to parse.

    Returns
    -------
    tuple[int, tuple[int, int]]
        Year and tile ID components of the task.
    """
    tile_id = parse_region_code(task_id)
    year_str = task_id.split("/")[0]
    year = int(year_str)
    return year, tile_id
