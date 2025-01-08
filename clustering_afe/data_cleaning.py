# data_cleaning.py
import pandas as pd
import numpy as np

def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that have only one distinct value.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame, excluding columns with a single distinct value.
    """
    
    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(col, axis=1, inplace=True)
    return df

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values:
      - Median for numeric columns
      - Mode for categorical/object columns

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with possible missing values.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with missing values imputed.
    """
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
    return df

def convert_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns to boolean if they contain only {0,1} or {True,False}.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with certain columns converted to bool dtype.
    """
    
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals == {0, 1} or unique_vals == {True, False}:
            df[col] = df[col].astype(bool)
    return df

def winsorize_outliers(df: pd.DataFrame):
    """
    Winsorizes numeric columns in the DataFrame by clamping values
    below the 1st percentile (p1) and above the 99th percentile (p99).
    Returns:
      - df: The winsorized DataFrame (copy of the original)
    """
    
    numeric_cols = df.select_dtypes(include=np.number).columns

    p1 = df[numeric_cols].quantile(0.01)
    p99 = df[numeric_cols].quantile(0.99)

    df[numeric_cols] = df[numeric_cols].clip(lower=p1, upper=p99, axis=1)

    return df
