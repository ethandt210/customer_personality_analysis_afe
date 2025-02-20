import re
import numpy as np
import pandas as pd

# Data Cleaning
from .data_cleaning import (
    drop_single_value_columns,
    impute_missing_values,
    convert_to_boolean,
    winsorize_outliers
)

# GPT Transformation
from .gpt_transformation import (
    config_client,
    build_prompt_from_df,
    call_gpt_for_transformation
)

# Feature Transformation
from .feature_transformation import (
    frequency_encoding,
    transform_boolean_columns,
    pairwise_metafeature_generation,
    feature_scaling_standard
)

# Feature Reduction
from .feature_reduction import ant_colony_optimization_search


def data_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the essential data cleaning steps in a typical sequence:
      1) Drop single-value columns
      2) Impute missing values
      3) Convert columns with {0,1} or {True,False} to boolean
      4) Winsorize outliers at p1/p99

    Parameters
    ----------
    df : pd.DataFrame
        The input raw data

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame
    """
    print("=========================[STEP 1]: CLEANING DATA...=========================\n")

    # 1) Drop single-value columns
    df = drop_single_value_columns(df)
    # 2) Impute missing
    df = impute_missing_values(df)
    # 3) Convert to boolean
    df = convert_to_boolean(df)
    # 4) Winsorize outliers (clamp p1/p99)
    df = winsorize_outliers(df)

    return df


def execute_gpt_code_snippets(df: pd.DataFrame, gpt_code: str) -> pd.DataFrame:
    """
    Executes any <start_code>...<end_code> blocks from GPT on the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to transform.
    gpt_code : str
        The raw GPT response containing code snippets.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame after running code snippets.
    """
    code_snippets = re.findall(r"<start_code>\n(.*?)\n<end_code>", gpt_code, re.DOTALL)
    if not code_snippets:
        print("No <start_code>...<end_code> blocks found in GPT response.")
        return df

    local_scope = {"df": df.copy(), "pd": pd, "np": np}

    for snippet in code_snippets:
        try:
            print(f"Executing Code: {snippet}\n")
            exec(snippet, {}, local_scope)
        except Exception as e:
            print(f"Error executing GPT code snippet: {e}")

    return local_scope["df"]


def gpt_transform_pipeline(df: pd.DataFrame, api_key: str, use_checklist: bool = True) -> pd.DataFrame:
    """
    1) Build a prompt based on the current DataFrame’s attributes 
       (optionally with a GPT-generated checklist).
    2) Call GPT to generate Python code that transforms the DataFrame.
    3) Execute that code on the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input data
    api_key : str
        Your OpenAI API key for GPT transformations.
    use_checklist : bool, optional
        If True, calls GPT to produce a "checklist" before building the final prompt.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.
    """
    print("=========================[STEP 2]: CALLING GPT...=========================\n")

    # Configure GPT client
    config_client(api_key)

    # Create prompt from df
    prompt = build_prompt_from_df(df, use_checklist=use_checklist)

    # Call GPT
    gpt_response = call_gpt_for_transformation(prompt)

    # Run the generated code snippets
    df = execute_gpt_code_snippets(df, gpt_response)

    return df


def meta_feature_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example advanced transformations:
      1) Frequency encoding for categorical columns
      2) Transform boolean columns to numeric weighting
      3) Pairwise feature generation (squared, sqrt, products, divisions)
      4) Standard scaling (z-score)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame
    """
    print("=========================[STEP 3]: TRANSFORMING META-FEATURES...=========================\n")

    # 1) Frequency encode categorical columns
    df = frequency_encoding(df)
    # 2) Convert boolean columns to numeric weighting
    df = transform_boolean_columns(df)
    # 3) Generate pairwise interactions
    df = pairwise_metafeature_generation(df)
    # 4) Apply standard scaling
    df = feature_scaling_standard(df)

    return df


def feature_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature transformations but without pairwise meta-feature generation:
      1) Frequency encoding for categorical columns
      2) Transform boolean columns to numeric weighting
      3) Standard scaling (z-score)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame
    """
    print("=========================[STEP 3]: TRANSFORMING FEATURES...=========================\n")

    # 1) Frequency encode categorical columns
    df = frequency_encoding(df)
    # 2) Convert boolean columns to numeric weighting
    df = transform_boolean_columns(df)
    # 3) Apply standard scaling
    df = feature_scaling_standard(df)

    return df


def feature_reduction(df: pd.DataFrame):
    """
    Example: use Ant Colony Optimization to find a subset of features 
    that yields good clustering performance (CHI vs DBI).
    Then reduce `df` to only those columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df_reduced : pd.DataFrame
        DataFrame with reduced set of best features
    best_feats : list
        List of best features found by ACO
    best_score : float
        Best score found by the optimization
    best_k : int
        The 'k' chosen by the optimization (e.g., # of clusters)
    """
    print("=========================[STEP 4]: PERFORMING FEATURE SEARCH...=========================\n")

    best_feats, best_score, best_k = ant_colony_optimization_search(df)
    final_cols = [col for col in best_feats if col in df.columns]
    df_reduced = df[final_cols]

    return df_reduced, best_feats, best_score, best_k


def run_pipeline(
    df: pd.DataFrame,
    api_key: str,
    use_gpt: bool = True,
    do_metafeature_engineering: bool = True,
    do_aco: bool = True
) -> pd.DataFrame:
    """
    Master method that calls each step in a typical sequence:
      1) Data Cleaning
      2) (Optional) GPT transformations
      3) (Optional) Meta-Feature transformations or simpler Feature transformations
      4) (Optional) Feature reduction (Ant Colony)
      5) Return final DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        The raw data to process.
    api_key : str
        Your OpenAI API key.
    use_gpt : bool, optional
        Whether to run GPT-based transformations.
    do_metafeature_engineering : bool, optional
        Whether to run the advanced meta-feature transformations (pairwise).
        If False, runs simpler transformations (no pairwise).
    do_aco : bool, optional
        Whether to run the ant_colony_optimization_search for feature selection.

    Returns
    -------
    pd.DataFrame
        The fully processed DataFrame after all transformations.
    """
    # 1) Data Cleaning
    df = data_cleaning_pipeline(df)

    # 2) GPT transformations
    if use_gpt:
        df = gpt_transform_pipeline(df, api_key, use_checklist=True)

    # 3) Feature transformations
    if do_metafeature_engineering:
        df = meta_feature_transform(df)
    else:
        df = feature_transform(df)

    # 4) Feature reduction with Ant Colony
    if do_aco:
        df, best_features, best_score, best_k = feature_reduction(df)
        # Optionally, you might want to log or store best_features, etc.
        # print("Best ACO Features:", best_features)

    print("=========================[DONE]=========================")
    return df
