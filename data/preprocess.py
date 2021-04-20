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
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minio import Minio
from tqdm import tqdm

import boto3


def get_files(bucket_name: str, boto_client, file_type: str = 'zip', prefix=None, list_objects_type='list_objects_v2'):
    """get list of files (zip/csv/pkl) from minio bucekt

    Parameters
    ----------
    bucket_name : str
        minio bucket as input 
    boto_client : [type]
        s3 client
    file_type : str, optional
        type/format of file, by default 'zip'
    prefix : str, optional
        minio bucket object path, by default None
    list_objects_type : str, optional
        object type, by default 'list_objects_v2'

    Returns
    -------
    List
        list of files having train sensor data
    """

    paginator = boto_client.get_paginator(list_objects_type)

    if prefix is not None:
        operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}
    else:
        operation_parameters = {'Bucket': bucket_name}

    files = list()
    for page in paginator.paginate(**operation_parameters):
        if "Contents" in page.keys():
            for obj in page["Contents"]:
                if file_type == 'zip':
                    if "/Output Zip Files/" in obj["Key"] and obj["Key"].endswith(".zip"):
                        files.append(obj["Key"])
                elif file_type in ['csv', 'pkl']:
                    files.append(obj["Key"])

    return files


def log_processed_files(text, logfile="processedfiles.csv"):
    """log processed zip file which is a daywise train data

    Parameters
    ----------
    text : str
        file path
    logfile : str, optional
        log filename, by default "processedfiles.csv"
    """    
    if os.path.exists(logfile):
        is_processed = False
        with open(logfile, "r") as file:
            for line in file:
                if text in line:
                    is_processed = True
                    break

        if not is_processed:  # adding file if it is not processed
            with open(logfile, "a") as file:
                file.write(f"{text}\n")  # append missing data

    else:  # creating file if it doesn't exist
        with open(logfile, "w") as file:
            file.write(f"{text}\n")  # append missing data


def is_file_processed(text, logfile="processedfiles.csv"):
    """check if file is already processed or not

    Parameters
    ----------
    text : str
        file path
    logfile : str, optional
        log filename, by default "processedfiles.csv"

    Returns
    -------
    bool
        return True if file exists or False
    """    
    if not os.path.exists(logfile):
        return False

    filelist = pd.read_csv(logfile, header=None).values.flatten()
    if text in filelist:
        return True
    else:
        return False


def replace_element(items, old, new):
    return [new if item == old else item for item in items]
