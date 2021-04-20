import logging
from typing import List, Dict
# import psycopg2
from sqlalchemy import create_engine
import pandas as pd


class PostgreSqlClient():
    """postgres sql client to write dataframe to a table
    """    
    def __init__(self, host: str, database: str, user: str = None, password: str = None):
        if host and database:
            self.host = host
            if 'http' in host:
                self.host = self.host.replace("http://", "")
            if 'https' in self.host:
                self.host = self.host.replace("https://", "")
            self.host = self.host.rstrip("/")
            self.database = database
            self.user = user
            self.password = password
            self.engine = create_engine(
                f'postgresql://{user}:{password}@{host}/{database}')
        else:
            raise ValueError("Info not provided.")

    def write_df_to_table(self, df: pd.DataFrame, table_name: str):
        """write dataframe to postgres sql table

        Parameters
        ----------
        df : pd.DataFrame
            dataframe
        table_name : str
            table name
        """        
        logging.info(f"writing dataframe to {table_name}")
        df.to_sql(table_name, self.engine, method="multi",
                  if_exists="append", chunksize=500)

        logging.info(f"Dump dataframe to postgre table {table_name}")

    def read_table_as_df(self, table_name: str):
        """read table as dataframe

        Parameters
        ----------
        table_name : str
            table name

        Returns
        -------
        pd.DataFrame
            dataframe
        """        
        df = pd.read_sql_query(
            f""" select * from "{table_name}" """, con=self.engine)
        return df


def getType(value):
    if isinstance(value, str):
        return "char(500)"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "numeric"
    if isinstance(value, bool):
        return "boolean"
