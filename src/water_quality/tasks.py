def create_task_id(year: str | int, tile_id: tuple[int, int] | str):
    if isinstance(year, int):
        year = str(year)

    task_id_format = "{year}/x{x:02d}/y{y:02d}"
    tile_id_x = tile_id[0]
    tile_id_y = tile_id[1]
    task_id = task_id_format.format(year=year, x=tile_id_x, y=tile_id_y)
    return task_id
