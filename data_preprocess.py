import pandas as pd
from zipfile import ZipFile

def zip_extract(path:str):
    zip_ref = ZipFile('archive(1).zip')
    zip_ref.extractall()
    zip_ref.close()      

def clean_data(dataframe:pd.DataFrame):
    dataframe.FILENAME = dataframe.FILENAME.astype(str)
    dataframe.IDENTITY = dataframe.IDENTITY.astype(str)
    dataframe = dataframe[dataframe['IDENTITY']!='UNREADABLE']
    df_clean = dataframe.dropna()
    return df_clean
