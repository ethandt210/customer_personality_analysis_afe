import pandas as pd
import numpy as np
from itertools import combinations, permutations
from sklearn.preprocessing import StandardScaler

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
    

    for column in df.select_dtypes(include=['object', 'category']).columns.tolist():

      # Calculate frequency of each value in the column
      frequency_map = df[column].value_counts(normalize=True).to_dict()

      # Replace values in the original column with their frequencies
      df[column] = df[column].map(frequency_map)

    return df

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
    
    for col in df.select_dtypes(include='bool'):
        total = len(df)
        true_count = df[col].sum()
        true_prop = true_count / total
        false_prop = 1 - true_prop

        df[col] = df[col].apply(lambda x: (1 - true_prop) if x else (1 - false_prop))

    return df

def pairwise_metafeature_generation(df):
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

    

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    interaction_data = {}

    # 1. Squared terms
    for feat in numeric_cols:
        interaction_data[f"{feat}^2"] = df[feat] ** 2

    # 2. Square roots
    for feat in numeric_cols:
        interaction_data[f"sqrt_{feat}"] = df[feat] ** 0.5  # or np.sqrt(df[feat].abs())

    # 3. Multiplications (combinations)
    for feat_a, feat_b in combinations(numeric_cols, 2):
        interaction_data[f"{feat_a}_x_{feat_b}"] = df[feat_a] * df[feat_b]

    # 4. Divisions (permutations)
    for feat_a, feat_b in permutations(numeric_cols, 2):
        interaction_name = f"{feat_a}_div_{feat_b}"
        with np.errstate(divide='ignore', invalid='ignore'):
            division_result = np.where(
                df[feat_b] != 0,
                df[feat_a] / df[feat_b],
                0
            )
        interaction_data[interaction_name] = division_result

    interaction_df = pd.DataFrame(interaction_data, index=df.index)

    # Concatenate new features
    return pd.concat([df, interaction_df], axis=1)

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

    
    numerical_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df