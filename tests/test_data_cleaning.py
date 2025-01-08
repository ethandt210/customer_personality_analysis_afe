# tests/test_data_cleaning.py

import pytest
import pandas as pd
from clustering_afe.data_cleaning import (
    drop_single_value_columns,
    impute_missing_values,
    convert_to_boolean,
    winsorize_outliers
)

def test_drop_single_value_columns():
    df = pd.DataFrame({
        'A': [1, 1, 1],   # single-value column
        'B': [1, 2, 3],   # multi-value column
        'C': [0, 0, 0],   # single-value column
    })
    result = drop_single_value_columns(df)
    assert 'A' not in result.columns
    assert 'C' not in result.columns
    assert 'B' in result.columns

def test_impute_missing_values():
    df = pd.DataFrame({
        'Numeric': [1, None, 3],
        'Categorical': ['A', None, 'A']
    })
    result = impute_missing_values(df)
    # Check that no Nones remain
    assert result['Numeric'].isna().sum() == 0
    assert result['Categorical'].isna().sum() == 0

def test_convert_to_boolean():
    df = pd.DataFrame({
        'Boolish0': [0, 0, 1, 1],
        'Boolish1': [False, True, False, True],
        'Numeric': [10, 20, 30, 40]
    })
    result = convert_to_boolean(df)
    # Check dtypes
    assert result['Boolish0'].dtype == 'bool'
    assert result['Boolish1'].dtype == 'bool'
    assert result['Numeric'].dtype != 'bool'

def test_winsorize_outliers():
    df = pd.DataFrame({
        'A': [1, 2, 999],
        'B': [10, 10, 10],
    })
    result = winsorize_outliers(df) 
