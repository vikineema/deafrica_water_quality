from water_quality.monitoring.load_data import NORMALISATION_PARAMETERS


def create_wq_odc_product(
    product_name: str,
    product_description: str,
    measurements: list[str],
):
    measurements_config = []
    for band in measurments:
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
