import pandas as pd
import numpy as np
from itertools import combinations, permutations
from sklearn.preprocessing import StandardScaler


# prompt: create a function for frequency encoding with argument receiving df_input, it should replace the value in the original column directly

def frequency_encoding(df):
    """
    Performs frequency encoding on all categorical (object/category) columns in the DataFrame.
    Each unique value is replaced by its frequency (proportion) within that column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with categorical columns frequency-encoded.
    """
    df_copy = df.copy()

    for column in df_copy.select_dtypes(include=['object', 'category']).columns.tolist():

      # Calculate frequency of each value in the column
      frequency_map = df_copy[column].value_counts(normalize=True).to_dict()

      # Replace values in the original column with their frequencies
      df_copy[column] = df_copy[column].map(frequency_map)

    return df_copy

def transform_boolean_columns(df):
    """
    Transforms boolean columns into numeric "importance" scores:
      - For each boolean column, calculates the proportion of True vs False.
      - True is replaced by (1 - proportion of True), False by (1 - proportion of False).
      - This will give the rarer class with higher score.

    Example:
      If 30% of values are True, then True -> 0.70, False -> 0.30

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing boolean columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where boolean columns have been converted to numeric scores.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include='bool'):
        total = len(df_copy)
        true_count = df_copy[col].sum()
        true_prop = true_count / total
        false_prop = 1 - true_prop

        df_copy[col] = df_copy[col].apply(lambda x: (1 - true_prop) if x else (1 - false_prop))

    return df_copy

def pairwise_feature_generation(df):
    """
    Generates interaction features from all numeric columns in the DataFrame:
      - Squared terms (column^2)
      - Square roots (sqrt_column)
      - Multiplications (colA_x_colB)
      - Divisions (colA_div_colB), using permutations so both colA/colB and colB/colA

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with original columns plus additional interaction features.
    """

    df_copy = df.copy()

    # Identify numeric columns
    numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()

    interaction_data = {}

    # 1. Squared terms
    for feat in numeric_cols:
        interaction_data[f"{feat}^2"] = df_copy[feat] ** 2

    # 2. Square roots
    for feat in numeric_cols:
        interaction_data[f"sqrt_{feat}"] = df_copy[feat] ** 0.5  # or np.sqrt(df_copy[feat].abs())

    # 3. Multiplications (combinations)
    for feat_a, feat_b in combinations(numeric_cols, 2):
        interaction_data[f"{feat_a}_x_{feat_b}"] = df_copy[feat_a] * df_copy[feat_b]

    # 4. Divisions (permutations)
    for feat_a, feat_b in permutations(numeric_cols, 2):
        interaction_name = f"{feat_a}_div_{feat_b}"
        with np.errstate(divide='ignore', invalid='ignore'):
            division_result = np.where(
                df_copy[feat_b] != 0,
                df_copy[feat_a] / df_copy[feat_b],
                0
            )
        interaction_data[interaction_name] = division_result

    interaction_df = pd.DataFrame(interaction_data, index=df_copy.index)

    # Concatenate new features
    return pd.concat([df_copy, interaction_df], axis=1)

def feature_scaling_standard(df):
    """
    Applies standard scaling (z-score normalization) to all numeric columns:
      z = (x - mean) / std

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric features.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where numeric columns are scaled with StandardScaler.
    """

    df_copy = df.copy()
    numerical_cols = df_copy.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])

    return df_copy