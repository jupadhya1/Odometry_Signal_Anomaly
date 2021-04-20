import logging
import pickle
import json
from io import BytesIO, StringIO
from typing import Union, Dict, List
import pandas as pd

from minio import Minio
from minio.error import MinioException


class MinioClient(Minio):
    """ A minio client adding extra features in existing minio client

    Parameters
    ----------
    Minio ([type]): the actual client
    """

    def create_bucket_if_not_exist(self, bucket_name: str):
        """ check if bucket exists otherwise create it

        Parameters
        ----------
        bucket_name (str): name of bucket (s3 bucket)
        """
        if not self.bucket_exists(bucket_name):
            self.make_bucket(bucket_name)

    def write_df_to_minio(self, df: pd.DataFrame, bucket_name: str, outname: str):
        """ writes pandas dataframe to minio bucket with given object path 

        Parameters
        ----------
        df (pd.DataFrame): pandas dataframe
        bucket_name (str): name of bucket (s3 bucket)
        outname (str): bucket object path (with file .csv) to which file should be saved
        """        
        logging.info(
            f"writing dataframe to {outname} in minio bucket {bucket_name}")
        csv_bytes = df.to_csv().encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)

        self.create_bucket_if_not_exist(bucket_name)
        self.put_object(bucket_name,
                        outname,
                        data=csv_buffer,
                        length=len(csv_bytes),
                        content_type='application/csv')

        logging.info(
            f"Dump dataframe to minio bucket {bucket_name} as {outname}")

    def write_dict_to_minio(self, dict_obj: Dict, bucket_name: str, outname: str):
        """ writes dictionary to minio bucket with given object path as json object

        Parameters
        ----------
        dict_obj (Dict): python dicitonary
        bucket_name (str): name of bucket (s3 bucket)
        outname (str): bucket object path (with file .json) to which file should be saved
        """        
        logging.info(
            f"writing json to {outname} in minio bucket {bucket_name}")
        json_bytes = json.dumps(dict_obj).encode('utf-8')
        json_buffer = BytesIO(json_bytes)

        self.create_bucket_if_not_exist(bucket_name)
        self.put_object(bucket_name,
                        outname,
                        data=json_buffer,
                        length=len(json_bytes),
                        content_type='application/json')

        logging.info(
            f"Dump json to minio bucket {bucket_name} as {outname}")

    def write_df_as_xls(self, df: pd.DataFrame, bucket_name: str, outname: str):
        """write dataframe to minio bucket as xls file

        Parameters
        ----------
        df (pd.DataFrame): pandas dataframe
        bucket_name (str): name of bucket (s3 bucket)
        outname (str): bucket object path (with file .xls) to which file should be saved
        """        
        logging.info(
            f"writing dataframe to {outname} in minio bucket {bucket_name}")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)

        self.create_bucket_if_not_exist(bucket_name)
        self.put_object(bucket_name,
                        outname,
                        data=csv_buffer,
                        length=len(csv_bytes),
                        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        logging.info(
            f"Dump dataframe to minio bucket {bucket_name} as {outname}")

    def save_event_to_minio(self, df: pd.DataFrame, metadata, outname: str, bucket_name: str):
        """save pickel object to minio bucket 

        Parameters
        ----------
        df (pd.DataFrame): pandas dataframe
        metadata ([type]): metadata which to be saved
        outname (str): bucket object path (with file .pkl) to which file should be saved
        bucket_name (str): name of bucket (s3 bucket)
        """        
        logging.info(
            f"writing event dataframe as pickle object to {outname} in minio bucket {bucket_name}")

        pickle_byte_obj = pickle.dumps({"data": df, "metadata": metadata})

        self.create_bucket_if_not_exist(bucket_name)
        self.put_object(bucket_name,
                        outname,
                        data=BytesIO(pickle_byte_obj),
                        length=len(pickle_byte_obj))

        logging.info(
            f"Dump event dataframe as pickle to minio bucket {bucket_name} as {outname}")

    def get_bytes_data(self, bucket_name: str, filename: str):
        """read data as bytes from minio bucket

        Parameters
        ----------
        bucket_name (str): name of bucket (s3 bucket)
        filename (str): bucket object path (with any extension e.g .csv, .xls, .pkl)

        Returns:
            bytes: data as bytes
        """        
        try:
            logging.info(f"Getting {filename} from minio bucket {bucket_name}")
            data = self.get_object(bucket_name, filename)
        except:
            logging.error(
                f"Failed to Getting {filename} from minio bucket {bucket_name}")
            raise MinioException

        data = data.read()
        bytes_data = BytesIO(data)
        return bytes_data

    def get_dataframe(self, bucket_name: str, filename: str, index_col: Union[str, bool, int] = None):
        """Read files from bucket as dataframe 

        Parameters
        ----------
        bucket_name (str): name of bucket (s3 bucket)
        filename (str): bucket object path (with any extension e.g .csv, .xls, .pkl)
        index_col (Union[str, bool, int], optional): Column name or column index to place that column as a index. Defaults to None.

        Returns:
            pd.DataFrame: dataframe after reading from minio bucket
        """        
        bytes_data = self.get_bytes_data(bucket_name, filename)
        df = pd.read_csv(bytes_data, index_col=index_col)
        logging.info(
            f"Successfully got {filename} as dataframe from minio bucket {bucket_name}")
        return df

    def get_pickle(self, bucket_name: str, filename: str):
        """Read files from bucket as pickle object 

        Parameters
        ----------
            bucket_name (str): name of bucket (s3 bucket)
            filename (str): bucket object path (with extension .pkl)

        Returns:
            pickel: data as pickle object
        """        
        bytes_data = self.get_bytes_data(bucket_name, filename)
        pkl_data = pickle.load(bytes_data)["data"]
        logging.info(
            f"Successfully got {filename} as pickel from minio bucket {bucket_name}")
        return pkl_data

    def get_dict(self, bucket_name: str, filename: str):
        """Read files from bucket as dict  

        Parameters
        ----------
        bucket_name (str): name of bucket (s3 bucket)
        filename (str): bucket object path (with extension .json)

        Returns:
        ----------
            pickel: data as dict object
        """        
        bytes_data = self.get_bytes_data(bucket_name, filename)
        data = json.load(bytes_data)
        logging.info(
            f"Successfully got {filename} as dict from minio bucket {bucket_name}")
        return data
