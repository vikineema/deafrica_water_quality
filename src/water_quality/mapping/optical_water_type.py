import logging
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def create_OWT_response_models(agm=True) -> pd.DataFrame:
    """
    Create Optical Water Type (OWT) response models.

    Parameters
    ----------
    agm : bool, optional
        Whether to create models for geomedian instruments, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing OWT response models.
    """
    if agm:
        suffix = "_agm"
    else:
        suffix = ""

    OWT_spectra_fp = files("water_quality.data").joinpath(
        "Vagelis_OWT_allwaters_mean_standardised.csv"
    )
    OWT_spectra = pd.read_csv(OWT_spectra_fp)
    # Limit to inland water types
    OWT_spectra = OWT_spectra[OWT_spectra.index < 13]

    data = {}

    # Read in the wavelengths for the bands in oli, tm and msi
    # (spectral response models)
    for sensor in ["msi", "tm", "oli"]:
        sensor_data_fp = files("water_quality.data").joinpath(
            f"sensor bands-{sensor}.csv"
        )
        sensor_data = pd.read_csv(sensor_data_fp)
        # Rename columns to avoid clashes with reserved words
        sensor_data.rename(
            columns={"max": "l_max", "min": "l_min"}, inplace=True
        )
        a = sensor_data
        # list the bands  that are going to be relevant , ie., less than 800nm
        band_list = (
            a[a["l_max"] < 800].band_name
        ).values  # & set(ds.data_vars)

        # Set up a  dataframe to take results for this instrument
        labels = []
        for i in np.arange(1, 14):
            labels = np.append(labels, "OWT-" + str(i))

        inst_OWT = pd.DataFrame(
            data={"OWT": np.arange(1, 14, 1)}, index=labels
        )
        # Run through the bands ...
        for band_name in band_list:
            # determine the integration interval based on the central
            # wavlength and the width; extract these as integers
            delta = a[a["band_name"] == band_name]["width"].values * 0.8 * 0.5
            start = int(
                (a[a["band_name"] == band_name]["central"].values - delta)[0]
            )
            end = int(
                (a[a["band_name"] == band_name]["central"].values + delta)[0]
            )
            # print(sensor,band_name, start,end)

            # find the start and end column numbers in the response functions
            # and sum over those for each OWT
            b = OWT_spectra.loc[:, str(start) : str(end)].T.sum()
            # add to the data frame
            inst_OWT.insert(
                inst_OWT.columns.size, band_name + suffix, (b.values)
            )
        # --- add this data frame to the data dictionary ---
        data[sensor] = inst_OWT
        # --- write to a csv file in the current location ---
        # inst_OWT.to_csv(sensor+'_OWT_vectors.csv')
    return None


def OWT(
    ds,
    instrument,
    OWT_vectors,
    agm=False,
    dp_corrected=False,
):
    # --- this version uses a long-hand approch to calculating the dot product. Inelegant but simple.
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
    OWT_vectors = OWT_vectors.reset_index().rename(
        columns={"index": "OWT"}
    )  # brings the OWT labels in as a normal column,'OWT'

    # --- loop through the Optical water types ---
    start = True
    for OWT in OWT_vectors["OWT"]:
        vec = OWT_vectors[OWT_vectors["OWT"] == OWT]  # the vector for this OWT
        OWT_index = int(OWT[OWT.find("-") + 1 :])  # the OWT numnber

        # print(OWT_vectors[OWT_vectors['OWT']==OWT])
        # --- create a dataset for this OWT based on one of he instrument bands ---
        varname = band_list[0]
        if start:
            start = False
            # working variiables
            mydataset = xr.Dataset({"owt_current": ds[varname]})
            #            mydataset = xr.Dataset({'owt_current' : ds[varname][:,::resample_rate,::resample_rate]})
            mydataset["owt_cos_max"] = (
                mydataset["owt_current"] * 0 + -2
            )  # a number smaller than any cosine
            mydataset["owt_cos"] = mydataset["owt_current"] * 0
            mydataset["owt_closest"] = mydataset["owt_current"] * 0 + OWT_index
            mydataset["self_product"] = mydataset["owt_current"] * 0
            mydataset["vector_product"] = mydataset["owt_current"] * 0

        mydataset["self_product"] = 0
        mydataset["vector_product"] = 0
        for band in band_list:
            mydataset["self_product"] = (
                mydataset["self_product"] + ds[band] ** 2
            )
            mydataset["vector_product"] = (
                mydataset["vector_product"] + ds[band] * vec[band].item()
            )

        mydataset["self_product"] = mydataset["self_product"] ** 0.5
        mydataset["owt_cos"] = mydataset["vector_product"] / (
            mydataset["self_product"] * vec["length"].values
        )
        mydataset["owt_closest"] = xr.where(
            mydataset["owt_cos"] > mydataset["owt_cos_max"],
            OWT_index,
            mydataset["owt_closest"],
        )
        mydataset["owt_cos_max"] = xr.where(
            mydataset["owt_cos"] > mydataset["owt_cos_max"],
            mydataset["owt_cos"],
            mydataset["owt_cos_max"],
        )

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
        owt = mydataset.owt_closest.median().item()
        desc = df[df["OWT"] == owt]["description"].item()
        print("Prevailng water type is ", owt, " :  ", desc)
    return mydataset["owt_closest"]


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
    OWT_vector_data = create_OWT_response_models(agm=True)
    suffix = "_agm"
    for instrument in ["msi", "oli", "tm"]:
        inst_agm = instrument + suffix
        # Use the smad band as an indicator that data for the geomedian
        # instrument exists in the dataset.
        smad_band = f"{inst_agm}_smad"
        if smad_band in ds.data_vars:
            OWT_vectors = OWT_vector_data[instrument]
            OWT_data = OWT(
                ds.where(ds.clearwater == 1),
                instrument,
                OWT_vectors,
                agm=True,
            )
        varname = instrument + "_agm_owt"
        if varname in ds.data_vars:
            ds[varname] = xr.where(~np.isnan(OWT_data), OWT_data, ds[varname])
        else:
            ds[varname] = OWT_data
        del OWT_data
    return ds
