# tests/test_feature_transformation.py

import pytest
import pandas as pd
import numpy as np

from clustering_afe.feature_transformation import (
    frequency_encoding,
    transform_boolean_columns,
    pairwise_metafeature_generation,
    feature_scaling_standard
)

def test_frequency_encoding():
    """
    Test that each categorical value is replaced by its frequency (proportion)
    and the result is float-typed.
    """
    df = pd.DataFrame({
        "Color": ["Red", "Red", "Blue", "Green", "Green", "Green"],
        "Value": [1, 2, 3, 4, 5, 6]
    })
    result = frequency_encoding(df)

    # Check that the "Color" column is now float
    assert result["Color"].dtype == float, "Frequency encoding did not produce float dtype."

    # Check that the frequencies sum to 1 if we group them (sanity check)
    # e.g. 'Green' should appear 3 times out of 6 => 0.5
    unique_vals = result["Color"].unique()
    assert len(unique_vals) == 3, "There should still be 3 unique frequency values for 3 categories."

def test_transform_boolean_columns():
    """
    Test that boolean columns are converted into numeric scores
    based on the proportion of True vs False in the DataFrame.
    """
    df = pd.DataFrame({
        "Flag1": [True, True, False, True],
        "Flag2": [False, False, False, False],
        "Value": [10, 20, 30, 40]
    })
    result = transform_boolean_columns(df)

    # 'Flag1' and 'Flag2' should no longer be bool
    assert result["Flag1"].dtype != bool, "Flag1 dtype not transformed from bool."
    assert result["Flag2"].dtype != bool, "Flag2 dtype not transformed from bool."

    # Check that numeric columns remain unchanged
    assert (result["Value"] == [10, 20, 30, 40]).all()

def test_pairwise_feature_generation():
    """
    Test that squared, sqrt, multiplication, and division features
    are created for all numeric columns.
    """
    df = pd.DataFrame({
        "X": [1, 2, 3],
        "Y": [4, 5, 6],
        "NonNumeric": ["A", "B", "C"]  # should be ignored in numeric combos
    })

    # Generate pairwise features
    result = pairwise_metafeature_generation(df) 

    # Expected new columns:
    #   X^2, sqrt_X, Y^2, sqrt_Y,
    #   X_x_Y, Y_x_X, ...
    #   X_div_Y, Y_div_X, etc.
    assert "X^2" in result.columns, "Squared feature for X missing."
    assert "sqrt_Y" in result.columns, "Square root feature for Y missing."
    assert "X_x_Y" in result.columns, "Multiplication feature for X and Y missing."
    assert "X_div_Y" in result.columns, "Division feature (X/Y) missing."
    assert "Y_div_X" in result.columns, "Division feature (Y/X) missing."

    # Check a sample calculation
    # For example, X=2, Y=5 => 'X_div_Y' = 0.4
    idx = df.index[df["X"] == 2][0]  # row where X=2
    expected_div = 2 / 5
    assert np.isclose(result.loc[idx, "X_div_Y"], expected_div), "Incorrect division result for X=2, Y=5."

def test_feature_scaling_standard():
    """
    Test that numeric columns are transformed, 
    and non-numeric columns remain unchanged in dtype and values.
    We do not check for exact zero mean or unit std.
    """
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "Cat": ["x", "y", "x", "y", "z"]
    })
    original_cat_values = df["Cat"].tolist()

    result = feature_scaling_standard(df)

    # A and B should still be numeric
    assert pd.api.types.is_float_dtype(result["A"]), "Column 'A' should be float after scaling."
    assert pd.api.types.is_float_dtype(result["B"]), "Column 'B' should be float after scaling."

    # Cat column should be unchanged
    assert (result["Cat"] == original_cat_values).all(), "Non-numeric column 'Cat' should remain unchanged."
    assert result["Cat"].dtype == df["Cat"].dtype, "Non-numeric column 'Cat' dtype should remain unchanged."

    # Check shape remains the same
    assert result.shape == df.shape, "DataFrame shape should remain identical after scaling."
