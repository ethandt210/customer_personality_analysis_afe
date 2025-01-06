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
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].nunique() == 1:
            df_copy.drop(col, axis=1, inplace=True)
    return df_copy

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
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            median_val = df_copy[col].median()
            df_copy[col].fillna(median_val, inplace=True)
        else:
            mode_val = df_copy[col].mode()[0]
            df_copy[col].fillna(mode_val, inplace=True)
    return df_copy

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
    df_copy = df.copy()
    for col in df_copy.columns:
        unique_vals = set(df_copy[col].dropna().unique())
        if unique_vals == {0, 1} or unique_vals == {True, False}:
            df_copy[col] = df_copy[col].astype(bool)
    return df_copy

def winsorize_outliers(df: pd.DataFrame):
    """
    Winsorizes numeric columns in the DataFrame by clamping values
    below the 1st percentile (p1) and above the 99th percentile (p99).
    Returns:
      - df_copy: The winsorized DataFrame (copy of the original)
      - outlier_mask: Boolean mask (DataFrame) indicating where outliers occurred
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=np.number).columns

    p1 = df_copy[numeric_cols].quantile(0.01)
    p99 = df_copy[numeric_cols].quantile(0.99)

    df_copy[numeric_cols] = df_copy[numeric_cols].clip(lower=p1, upper=p99, axis=1)

    return df_copy
