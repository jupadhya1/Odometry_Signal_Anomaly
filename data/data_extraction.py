import json
import logging
import multiprocessing as mp
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from importlib import reload
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from minio import Minio
from tqdm import tqdm

from typing import Dict

import boto3

from ..config.config import CONFIG
from .preprocess import is_file_processed, log_processed_files, replace_element


def get_diagnostics_df_from_bytes(bytes_data, cols=None, nrows=None, rename_cols: Dict[str, str] = None, fillna_method='ffill'):
    """Returns dataframe object from a bytes object
    representing a csv file extracted from a zip file.

    Parameters
    ----------
    bytes_data : bytes
        data as bytes
    cols : List, optional
        list of columns to be processed, by default None
    nrows : int, optional
        number of rows, by default None
    rename_cols : Dict, optional
        rename columns name, by default None
    fillna_method : str, optional
        method to fill na values, by default 'ffill'

    Returns
    -------
    pd.DataFrame
        dataframe
    """


    # string representation of the data
    if not isinstance(bytes_data, str):
        s = str(bytes_data, 'utf-8')

    if rename_cols is None:
        rename_cols = {
            'MSTEP_A_axle1RawSpeed': 'axle1RawSpeed',
            'MSTEP_A_axle2RawSpeed': 'axle2RawSpeed'
        }
    # reading the file content to get the header as a string,
    # to check if fields defined in features_dtypes_dict exists in header
    header_proxy = s[:10000].split(",")

    cols_corrected = cols.copy()
    # rename axle variables if needed
    if "MSTEP_A_axle1RawSpeed" not in header_proxy:
        cols_corrected = replace_element(
            cols_corrected, "MSTEP_A_axle1RawSpeed", "axle1RawSpeed")
    if "MSTEP_A_axle2RawSpeed" not in header_proxy:
        cols_corrected = replace_element(
            cols_corrected, "MSTEP_A_axle2RawSpeed", "axle2RawSpeed")

    data = StringIO(s)
    df = pd.read_csv(data,
                     nrows=nrows,
                     usecols=cols_corrected)

    # rename unnamed variables
    df = df.rename({"Unnamed: 0": "TimeStamp",
                    "Unnamed: 1": "Record Number"},
                   axis=1)

    # rename axle variables if needed
    if "MSTEP_A_axle1RawSpeed" in df.columns:
        df = df.rename({"MSTEP_A_axle1RawSpeed": "axle1RawSpeed"}, axis=1)
    if "MSTEP_A_axle2RawSpeed" in df.columns:
        df = df.rename({"MSTEP_A_axle2RawSpeed": "axle2RawSpeed"}, axis=1)

    # Replacing * by previous numerical value, since it means "no change in value"
    df.replace("*", np.nan, inplace=True)
    df.fillna(method=fillna_method, inplace=True)

    return df


def process_file(zip_file: str, input_bucket: str, output_bucket: str, minio_client, columns, nrows=None, output_csv_name_prefix: str='filtered'):
    """Download zip file from minio bucket, read all the diagnostic files and 
    write again over minio bucket

    Parameters
    ----------
    zip_file : str
        zip file path respect to minio
    input_bucket : str
        minio bucket where zip file exists
    output_bucket : str
        minio bucket where output should be put
    minio_client : [type]
        minio client
    columns : [type]
        columns to be picked
    nrows : int, optional
        number of rows, by default None
    output_csv_name_prefix : str, optional
        intermediate minio directory path, by default 'filtered'

    Returns
    -------
    str
        process status
    """
    
    if is_file_processed(zip_file):
        logging.info(f"{zip_file} is already processed.")
        return "processed"

    data = minio_client.get_object(input_bucket, zip_file)

    try:
        zf = zipfile.ZipFile(BytesIO(data.read()))
    except zipfile.BadZipFile as bzf:
        logging.info(f"{zip_file} is a bad zip file.")
        log_processed_files(zip_file)
        return "bad zip file"

    archived_files = zf.namelist()
    diagnostic_files = [
        n for n in archived_files if "Diagno" in n and n.endswith("csv")]

    dfs = []
    for filename in diagnostic_files:
        try:
            data = zf.read(filename)
        except FileNotFoundError:
            logging.error(f"Did not find {filename} in zip file {zip_file}.")
        else:
            df = get_diagnostics_df_from_bytes(bytes_data=data,
                                               cols=columns,
                                               nrows=nrows)
            if df is None:
                logging.info(f"{zip_file} is skipped.")
                return "problem in diagnostic file"
            else:
                dfs.append(df)

    df_filt = pd.concat(dfs)
    file_info = df_filt.set_index(pd.to_datetime(
        df_filt["TimeStamp"])).sort_index().iloc[0]

    vid = file_info["VehicleID"]
    ymd = file_info.name.strftime("%Y-%m-%d")
    output_csv_name = f"{output_csv_name_prefix}/{vid}_{ymd}.csv"

    minio_client.write_df_to_minio(df_filt, output_bucket, output_csv_name)
    log_processed_files(zip_file)
    return "success"
