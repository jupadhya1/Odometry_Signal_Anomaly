import logging
import os
import tempfile
import uuid
import pickle
from typing import List, Union
from io import BytesIO, StringIO
from zipfile import ZipFile

import pandas as pd
from tqdm import tqdm


def change_axle_speed(df: pd.DataFrame, divide_by: Union[int, float]=10):
    """change axle spped by a ratio

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    divide_by : Union[int, float], optional
        ratio value , by default 10

    Returns
    -------
    pd.DataFrame
        dataframe
    """    
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    df = df.set_index("TimeStamp")
    df = df.sort_index()

    df["axle1RawSpeed"] /= divide_by
    df["axle2RawSpeed"] /= divide_by
    return df


def get_filename_from_event_info(event: pd.Series, csv_name_prefix: str):
    """get filename from event information

    Parameters
    ----------
    event : pd.Series
        an event
    csv_name_prefix : str
        intermediate minio bucket object path

    Returns
    -------
    str
        filename
    """    
    vid = int(event["VehicleID"])
    date = pd.to_datetime(event["Timestamp"]).strftime("%Y-%m-%d")
    return f"{csv_name_prefix}/{vid}_{date}.csv"


def get_windows_files(df: pd.DataFrame, csv_name_prefix: str='filtered'):
    """get window file name

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    csv_name_prefix : str, optional
        intermediate minio bucket object path, by default 'filtered'

    Returns
    -------
    List
        list of files name
    """    
    windows_files = [get_filename_from_event_info(
        event, csv_name_prefix) for _, event in df.iterrows()]
    return windows_files


def create_window_event(files: List[str],
                        input_bucket: str,
                        output_bucket: str,
                        minio_client,
                        min_window_size: int = 1,
                        ouput_dir: str = 'events1min',
                        window_event_bucket: str = None,
                        window_events_file: str = None,
                        csv_name_prefix: str = 'filtered',
                        change_speed_by: Union[int, float] = 10,
                        train_zero_speed_ratio: float = 0.4,
                        datetime_limit: str = "2020-03-31",
                        filename_include: str = "nominal"):
    """create window event files corresponding to a train data datewise

    Parameters
    ----------
    files : List[str]
        number of train files datewise
    input_bucket : str
        minio bucket from where train files as datewise are choosing    
    output_bucket : str
        minio bucket where output should be put
    minio_client : MinioClient  
        minio client
    min_window_size : int, optional
        window size to create event, by default 1
    ouput_dir : str, optional
        minio bucket object path where output should be put, by default 'events1min'
    window_event_bucket : str, optional
        minio bucket where window event file exist, by default None
    window_events_file : str, optional
        minio object file path of window event, by default None
    csv_name_prefix : str, optional
        intermediate minio bucket object path, by default 'filtered'
    change_speed_by : Union[int, float], optional
        by which value speed should get change, by default 10
    train_zero_speed_ratio : float, optional
        minimum ratio to consider this event time interval, by default 0.4
    datetime_limit : str, optional
        datetime limit, by default "2020-03-31"
    filename_include : str, optional
        intermediate event file name, by default "nominal"

    Returns
    -------
    pd.DataFrame
        combined events data a particular window size
    """    
    df_events = []

    for file_name in tqdm(files):

        temp_pkl_path = tempfile.mkdtemp()

        vid = int(file_name.split("/")[-1].split("_")[0])
        dt = file_name.split("/")[-1].split("_")[1].strip(".csv")

        df = minio_client.get_dataframe(input_bucket, file_name)
        df = change_axle_speed(df, divide_by=change_speed_by)
        
        # treating files listed in white paper (labelled)
        if window_event_bucket is not None and window_events_file is not None:
            windows_df = pd.read_csv(window_events_file)
            windows_files = get_windows_files(windows_df, csv_name_prefix=csv_name_prefix)
        else:
            windows_files = []

        if file_name in windows_files:
            events = windows_df[(windows_df["VehicleID"] == vid) & (pd.to_datetime(
                windows_df["Timestamp"]).apply(lambda x: x.date()) == pd.to_datetime(dt))]

            for _, event in events.iterrows():

                begin = pd.to_datetime(event["Timestamp"])
                end = begin + pd.Timedelta(minutes=min_window_size)
                df = df[begin:end]

                if df.shape[0] == 0:
                    continue

                ts = pd.to_datetime(event["Timestamp"]).strftime(
                    "%Y-%m-%d--%H-%M-%S%f")
                if df.shape[0] != 0:
                    uid = uuid.uuid4()
                    df_event['minute_id'] = uid.hex 
                    dump_pickle(df_event, f"{temp_pkl_path}/{vid}_{ts}.pkl")
                    # minio_client.save_event_to_minio(
                    #     df, event, f"{temp_pkl_path}/{vid}_{ts}.pkl", output_bucket)
                    df_event = postprocess(df_event)
                    df_events.append(df_event)
                    print(df_event)
        # treating files for nominal
        else:
            if datetime_limit is not None or (datetime_limit is not None and datetime_limit != 'all'):
                if pd.to_datetime(dt) > pd.to_datetime(datetime_limit):
                    logging.info(f"Skip window event creation for time {dt}")
                    continue

            begin = df.index.min()
            end = begin + pd.Timedelta(minutes=min_window_size)

            while(end < df.index.max()):
                df_event = df[begin:end].copy()
                ts = end.strftime("%Y-%m-%d--%H-%M-%S%f")

                if 0 in df_event["trainSpeed"].value_counts().keys():
                    if df_event["trainSpeed"].value_counts()[0]/df_event["trainSpeed"].shape[0] > train_zero_speed_ratio:
                        begin = end
                        end = begin + pd.Timedelta(minutes=min_window_size)
                        continue
                    else:

                        if df.shape[0] != 0:
                            uid = uuid.uuid4()
                            df_event['minute_id'] = uid.hex
                            dump_pickle(df_event, f"{temp_pkl_path}/{filename_include}/{vid}_{ts}.pkl")
                            # minio_client.save_event_to_minio(
                            #     df_event, None, f"{temp_pkl_path}/{filename_include}/{vid}_{ts}.pkl", output_bucket)
                            df_event = postprocess(df_event)
                            df_events.append(df_event)
                        begin = end
                        end = begin + pd.Timedelta(minutes=min_window_size)
                        continue
                else:

                    if df.shape[0] != 0:
                        uid = uuid.uuid4()
                        df_event['minute_id'] = uid.hex
                        dump_pickle(df_event, f"{temp_pkl_path}/{filename_include}/{vid}_{ts}.pkl")
                        # minio_client.save_event_to_minio(
                        #     df_event, None,  f"{temp_pkl_path}/{filename_include}/{vid}_{ts}.pkl", output_bucket)
                        df_event = postprocess(df_event)
                        df_events.append(df_event)
                    begin = end
                    end = begin + pd.Timedelta(minutes=min_window_size)
                
                logging.info(f"Dump pickle file as {temp_pkl_path}/{filename_include}/{vid}_{ts}.pkl")

        zip_path = f"{os.path.basename(file_name).split('.')[0]}.zip"
    
    df = pd.concat(df_events)
    write_zip_file(temp_pkl_path, f"/tmp/{zip_path}")
    minio_client.fput_object(output_bucket,  f"{ouput_dir}/{zip_path}", f"/tmp/{zip_path}")
    logging.info(f"Dump pickled zip file to minio bucket {output_bucket} as {ouput_dir}/{zip_path}")
    return df_events


def postprocess(df: pd.DataFrame):
    """some postprocessing to the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        dataframe

    Returns
    -------
    pd.DataFrame
        return dataframe
    """         
    formt = '%Y-%m-%d %H:%M:%S.%f'
    df['vehicle_timestamp'] = pd.to_datetime(df.index, format=formt, utc=False)
    df['seconds'] = df['vehicle_timestamp'].apply(lambda x: x.strftime("%S"))
    df = df.drop_duplicates(subset='seconds', keep='first')
    df.drop('vehicle_timestamp', axis=1, inplace=True)
    return df


def write_zip_file(file_dir, save_path: str = None):
    """craete a zip fiel having having directory files

    Parameters
    ----------
    file_dir : str
        directory to be zipped 
    save_path : str, optional
        file path where to save zip file by default it get saved in /tmp directory, by default None

    Returns
    -------
    str
        output zip file path
    """    
    zip_file_path = save_path if save_path is not None else f"/tmp/tempzip_{uuid.uuid4()}.zip"
    with ZipFile(zip_file_path, 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(file_dir):
            for filename in subfolders+ filenames:
                source = os.path.join(folderName, filename)
                dest = source[len(file_dir):].lstrip(os.sep)
                zipObj.write(source, dest)
    return zip_file_path


def dump_pickle(data, file_path):
    """write any data to pickle file

    Parameters
    ----------
    data : anytype
        could be any serializable data
    file_path : str
        file path where to save file
    """ 
    base_dir = os.path.dirname(file_path)
    makedirs(base_dir)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)    


def makedirs(directory):
    """create a directory if it odes not exists

    Parameters
    ----------
    directory : str
        directory path
    """ 
    if not os.path.exists(directory):
        os.makedirs(directory)