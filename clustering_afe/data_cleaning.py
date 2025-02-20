import pandas as pd
import numpy as np

def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that have only one distinct value and logs the dropped columns.
    """
    initial_cols = df.shape[1]
    df = df.loc[:, df.nunique() > 1]
    dropped_cols = initial_cols - df.shape[1]
    print(f"[LOG] Dropped {dropped_cols} single-value columns.")
    return df

def impute_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Imputes missing values and logs before/after missing count.
    """
    missing_before = df.isnull().sum().sum()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    missing_after = df.isnull().sum().sum()
    print(f"[LOG] Imputed missing values. Before: {missing_before}, After: {missing_after}.")
    return df

def convert_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns to boolean if they contain only {0,1} or {True,False}, logs conversion.
    """
    bool_cols = []
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals == {0, 1} or unique_vals == {True, False}:
            df[col] = df[col].astype(bool)
            bool_cols.append(col)
    print(f"[LOG] Converted {len(bool_cols)} columns to boolean: {bool_cols}")
    return df

def winsorize_outliers(df: pd.DataFrame, lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    Winsorizes numeric columns by clamping values below lower_percentile and above upper_percentile.
    Logs outliers before and after winsorization.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_counts = {}
    
    for col in numeric_cols:
        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)
        outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        outliers_after = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = (outliers_before, outliers_after)
    
    for col, (before, after) in outlier_counts.items():
        print(f"[LOG] Column {col}: Outliers before={before}, after={after}")
    
    return df, outlier_counts
