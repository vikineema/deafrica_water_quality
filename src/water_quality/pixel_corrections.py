import logging
from typing import Any

from xarray import Dataset

log = logging.getLogger(__name__)


def R_correction(
    ds: Dataset,
    dp_adjust: dict[str, dict[str, Any]],
    instruments_to_use: dict[str, dict[str, bool]],
    water_frequency_threshold: float = 0.9,
):
    for sensor in dp_adjust.keys():
        if sensor in instruments_to_use.keys():
            usage = instruments_to_use[sensor]["use"]
            if usage:
                log.info(f"Performing dark pixel correction for sensor {sensor} ...")
                ref_var = dp_adjust[sensor]["ref_var"]
                if ref_var not in ds.data_vars:
                    raise ValueError(
                        f"Variable {ref_var} expected  but not found in the dataset - correction FAILING for sensor {sensor}",
                    )
                else:
                    for target_var in dp_adjust[sensor]["var_list"]:
                        if target_var not in ds.data_vars:
                            raise ValueError(
                                f"Variable {target_var} expected  but not found in the dataset; terminating the R_correction",
                            )
                        else:
                            new_var = str(target_var + "r")
                            ds[new_var] = (
                                (ds[target_var] - ds[ref_var])
                                .where(ds[target_var] > ds[ref_var], 0)
                                .where(ds[target_var] > 0)
                                .where(
                                    ds.wofs_ann_freq > water_frequency_threshold,
                                    ds[target_var],
                                )
                            )

            else:
                raise ValueError(
                    f"Dark pixel correction requested for sensor {sensor} , but sensor is not used in analysis."
                )
        else:
            raise ValueError(
                f"Dark pixel correction requested for sensor {sensor}, but sensor is not listed in instruments to use."
            )
    return ds
