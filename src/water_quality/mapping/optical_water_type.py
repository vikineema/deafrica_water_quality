import logging
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def get_OWT_description(owt):
    """
    Get description of a OWT value
    """
    if not (1 <= owt <= 13):
        raise ValueError("OWT value must be between 1 and 13")

    data = [
        [3, "oligotrophic (clear)"],
        [9, "oligotrophic (clear)"],
        [13, "oligotrophic (clear)"],
        [1, "eutrophic and blue-green"],
        [2, "eutrophic and blue-green"],
        [4, "eutrophic and blue-green"],
        [5, "eutrophic and blue-green"],
        [11, "eutrophic and blue-green"],
        [12, "eutrophic and blue-green"],
        [6, "hyper-eutrophic and green-brown"],
        [7, "hyper-eutrophic and green-brown"],
        [8, "hyper-eutrophic and green-brown"],
        [10, "hyper-eutrophic and green-brown"],
    ]
    data.sort()
    columns = ["OWT", "description"]

    df = pd.DataFrame(data=data, columns=columns)
    desc = df[df["OWT"] == owt]["description"].item()
    return desc


def create_OWT_response_model() -> dict[str, pd.DataFrame]:
    """
    Create Optical Water Type (OWT) response models and write to
    csv files for all supported instruments.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of DataFrames containing OWT response models for each
        instrument
    """
    sensors = [
        "msi",
        "tm",
        "oli",
    ]

    OWT_spectra_fp = files("water_quality.data").joinpath(
        "Vagelis_OWT_allwaters_mean_standardised.csv"
    )
    OWT_spectra = pd.read_csv(OWT_spectra_fp)
    # Limit to inland water types
    OWT_spectra = OWT_spectra[OWT_spectra.index < 13]

    data = {}

    # Read in the wavelengths for the bands in oli, tm and msi
    # (spectral response models)
    for sensor in sensors:
        # (spectal response models)
        if sensor.endswith("_agm"):
            suffix = "_agm"
            sensor_data_fp = files("water_quality.data").joinpath(
                f"sensor bands-{sensor[: -len(suffix)]}.csv"
            )
        else:
            suffix = ""
            sensor_data_fp = files("water_quality.data").joinpath(
                f"sensor bands-{sensor}.csv"
            )
        sensor_data = pd.read_csv(sensor_data_fp)
        sensor_data["band_name"] = sensor_data["band_name"] + suffix

        # Rename columns to avoid clashes with reserved words
        sensor_data.rename(
            columns={"max": "l_max", "min": "l_min"}, inplace=True
        )

        # set up a  dataframe to take results for this instrument
        owt_values = OWT_spectra["wl"].to_list()
        owt_labels = [f"OWT-{i}" for i in owt_values]
        inst_OWT = pd.DataFrame(data={"OWT": owt_values}, index=owt_labels)

        # list the bands  that are going to be relevant , ie., less than 800nm
        band_list = sensor_data[sensor_data["l_max"] < 800][
            "band_name"
        ].to_list()

        for band_name in band_list:
            band_data = sensor_data[
                sensor_data["band_name"] == band_name
            ].iloc[0]

            # Determine the integration interval based on the central
            # wavlength and the width
            delta = band_data["width"] * 0.8 * 0.5
            central = band_data["central"]

            # Find start and end column numbers in the response functions
            # and average over those for each OWT, must allow for the
            # spectral band width.
            start = str(int(central - delta))
            end = str(int(central + delta))
            inst_OWT.insert(
                inst_OWT.columns.size,
                band_name,
                OWT_spectra.loc[:, start:end].T.mean().to_list(),
            )
            inst_OWT.to_csv(
                files("water_quality.data").joinpath(
                    f"{sensor}_OWT_vectors.csv"
                )
            )
            log.info(f"Written OWT response model for {sensor} to csv file.")
        data[sensor] = inst_OWT
    return data


def OWT(
    ds,
    instrument,
    OWT_vectors,
    agm=False,
    dp_corrected=False,
    check_type=False,
):
    """
    Calculate per-pixel Optical Water Type by comparing to OWT spectral vectors.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing spectral band data.

    instrument: str
        Name of instrument.

    OWT_vectors: pd.DataFrame
        OWT spectral vectors, expected to have an index column formated as OWT-index.

    agm: bool, optional
        Whether the calculation is for geomedian instruments. Default is False.

    dp_corrected: bool, optional
        Whether to use Rayleigh corrected bands as input. Default is False.

    check_type: bool, optional
        Whether to check the prevailing water type in intermediate steps. Default is False.

    Returns
    -------
    xr.DataArray
        DataArray containing OWT estimate.
    """

    # --- this version uses a long-hand approach to calculating the dot product. Inelegant but simple.
    # --- identify the instrument bands that are relevant and in the dataset
    log.info(
        f"Calculating Optical Water Type (OWT) for instrument: {instrument}"
    )
    suffix = ""
    if agm:
        suffix = suffix + "_agm"
    if dp_corrected:
        suffix = suffix + "r"

    # This loop renames the columns so that we can match them with the
    # data variables
    for col in OWT_vectors.columns.values:
        if col.find(instrument) > -1:
            OWT_vectors = OWT_vectors.rename(columns={col: col + suffix})
    band_list = list(set(ds.data_vars) & set(OWT_vectors.columns.values))
    band_list.sort()

    # Ditch unnecessary columns and rows in the vectors table and
    # calculate the magnitude of each vector
    OWT_vectors = OWT_vectors.drop(
        columns=list(set(OWT_vectors.columns.values) - set(band_list))
    )
    OWT_vectors["length"] = (
        (OWT_vectors**2).sum(axis=1)
    ) ** 0.5  # calculate the length of each vector

    # --- loop through the Optical water types ---

    # initiate variables to hold current best OWT match & largest cos
    varname = band_list[0]
    owt_cos_max = xr.full_like(
        ds[varname], dtype=float, fill_value=-2
    )  # a number smaller than any cosine
    owt_closest = xr.full_like(
        ds[varname],
        dtype="uint8",
        fill_value=int(OWT_vectors.index[0].split("-")[1]),
    )

    for idx, vec in OWT_vectors.iterrows():
        OWT_index = int(idx.split("-")[1])  # the OWT numnber
        # print(OWT_index)

        # turn into a data arrays
        da = ds[band_list].to_array(dim="band")
        vec_da = xr.DataArray(vec, dims=["band"]).sel(band=band_list)
        # vectorised computation
        self_product = (da**2).sum(dim="band") ** 0.5
        vector_product = (da * vec_da).sum(dim="band")
        owt_cos = vector_product / (self_product * vec["length"])

        # now find the closest with the largest cos
        owt_closest = xr.where(
            owt_cos > owt_cos_max,
            OWT_index,
            owt_closest,
        )

        owt_cos_max = xr.where(
            owt_cos > owt_cos_max,
            owt_cos,
            owt_cos_max,
        )

        if check_type:
            # only use for debugging
            owt = owt_closest.median().item()
            desc = get_OWT_description(owt)
            print("Prevailng water type is ", owt, " :  ", desc)
    return owt_closest


def run_OWT(ds: xr.Dataset) -> xr.Dataset:
    """
    A function to run the Optical Water Type (OWT) classification on a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing spectral band data.

    Returns
    -------
    xr.Dataset
        Dataset with OWT classification results.
    """
    suffix = "_agm"
    for instrument in ["msi", "oli", "tm"]:
        inst_agm = instrument + suffix

        # OWT_vector only depends on the instrument
        OWT_vectors = pd.read_csv(
            files("water_quality.data").joinpath(
                f"{instrument}_OWT_vectors.csv"
            ),
            index_col=0,
        )
        # Use the count band as an indicator that data for the geomedian
        # instrument exists in the dataset.
        count_band = f"{inst_agm}_count"
        if count_band in ds.data_vars:
            OWT_data = OWT(
                ds.where(ds.clearwater == 1),
                instrument,
                OWT_vectors,
                agm=True,
            )
            varname = instrument + "_agm_owt"
            if varname in ds.data_vars:
                ds[varname] = xr.where(
                    ~np.isnan(OWT_data), OWT_data, ds[varname]
                )
            else:
                ds[varname] = OWT_data
    return ds
