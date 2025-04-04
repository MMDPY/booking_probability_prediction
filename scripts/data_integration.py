# data_integration.py is the first actor in this excercise. It loads the data into a CSV compatible format and stores it as well.

import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# TASK 1
def load_data(file_path: str) -> pd.DataFrame:
    '''
    Load a Parquet file into a Pandas DataFrame, optimizing column data types.
    
    Parameters:
    file_path (str): The file path to the Parquet file.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the loaded data.
    '''
    print("=" * 21 + "Data Integration Running . . ." + "=" * 21)
    
    # Read the Parquet file into a PyArrow Table
    table = pq.read_table(file_path)
    
    # Remove metadata and cast to fixed schema
    fixed_schema = table.schema.remove_metadata()
    fixed_table = table.cast(fixed_schema)
    
    # Convert to a Pandas DataFrame
    df = fixed_table.to_pandas()
    
    # Optimize data types for memory efficiency
    # Convert float64 columns to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Convert int64 columns to int32
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    
    return df


def save_csv(df: pd.DataFrame, output_path: str):
    '''
    Save the given DataFrame to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    output_path (str): The file path where the CSV will be saved.
    '''
    df.to_csv(output_path, index=False)  # Avoid saving index as a separate column
