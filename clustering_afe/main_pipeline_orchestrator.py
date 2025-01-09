# pipeline.py

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


class automated_feature_engineering:
    """
    A master orchestrator that unifies data cleaning, GPT transformations,
    feature engineering, and feature reduction in one pipeline.
    """

    def __init__(self, df: pd.DataFrame, api_key: str):
        """
        Initialize the pipeline with a DataFrame and an OpenAI API key.
        The DataFrame is stored internally, and the GPT client is configured.
        
        Parameters
        ----------
        df : pd.DataFrame
            The raw data you want to process end to end.
        api_key : str
            Your OpenAI API key for GPT transformations.
        """
        # Keep original DF untouched
        # store transformations in self.transformed_df
        self.original_df = df
        self.transformed_df = df.copy()  
        config_client(api_key)

        self.meta_info = {} # Stores informations of the processed data

    # -------------------------------------------------------------------------
    # Data Cleaning Steps
    # -------------------------------------------------------------------------
    def clean_data(self) -> "automated_feature_engineering":
        """
        Runs the essential data cleaning steps in a typical sequence:
          1) Drop single-value columns
          2) Impute missing values
          3) Convert columns with {0,1} or {True,False} to boolean
          4) Winsorize outliers at p1/p99

        Returns
        -------
        self : AutomatedPipeline
            (For method chaining)
        """
        print("=========================[STEP 1]: CLEANING DATA...=========================\n")

        # 1) Drop single-value columns
        self.transformed_df = drop_single_value_columns(self.transformed_df)
        # 2) Impute missing
        self.transformed_df = impute_missing_values(self.transformed_df)
        # 3) Convert to boolean
        self.transformed_df = convert_to_boolean(self.transformed_df)
        # 4) Winsorize outliers (clamp p1/p99)
        self.transformed_df = winsorize_outliers(self.transformed_df)

        return self

    # -------------------------------------------------------------------------
    # GPT Transformation
    # -------------------------------------------------------------------------
    def gpt_transform(self, use_checklist: bool = True) -> "automated_feature_engineering":
        """
        1) Build a prompt based on the current DataFrame’s attributes 
           (optionally with a GPT-generated checklist).
        2) Call GPT to generate Python code that transforms `self.df`.
        3) Execute that code on `self.df`.

        Parameters
        ----------
        use_checklist : bool, optional
            If True, calls GPT to produce a "checklist" before building the final prompt.

        Returns
        -------
        self : AutomatedPipeline
        """
        print("=========================[STEP 2]: CALLING GPT...=========================\n")

        prompt = build_prompt_from_df(self.transformed_df, use_checklist=use_checklist)

        code_response = call_gpt_for_transformation(prompt)

        self.run_code_blocks(code_response)

        return self

    def run_code_blocks(self, gpt_code: str):
        """
        A small helper that executes the <start_code>...<end_code> blocks 
        from GPT on self.df. 
        (We replicate the logic from your existing snippet, but keep it in the pipeline.)
        """
        import re, copy
        import numpy as np

        df_local = self.transformed_df.copy()
        code_snippets = re.findall(r"<start_code>\n(.*?)\n<end_code>", gpt_code, re.DOTALL)

        if not code_snippets:
            print("No <start_code>...<end_code> blocks found in GPT response.")
            return

        local_scope = {"df": df_local, "pd": pd, "np": np}
        for snippet in code_snippets:
            try:
                print(f"Executing Code: {snippet}\n")
                exec(snippet, {}, local_scope)
            except Exception as e:
                print(f"Error executing GPT code snippet: {e}")

        self.transformed_df = local_scope["df"]

    # -------------------------------------------------------------------------
    # Feature Transformations
    # -------------------------------------------------------------------------
    def meta_feature_transform(self) -> "automated_feature_engineering":
        """
        Example advanced transformations:
          1) Frequency encoding for categorical columns
          2) Transform boolean columns to numeric importance
          3) Pairwise feature generation (squared, sqrt, products, divisions)
          4) Standard scaling (z-score)

        Returns
        -------
        self : automated_feature_engineering
        """
        print("=========================[STEP 3]: TRANSFORMING META-FEATURES...=========================\n")

        # 1) Frequency encode categorical columns
        self.transformed_df = frequency_encoding(self.transformed_df)
        # 2) Convert boolean columns to numeric weighting
        self.transformed_df = transform_boolean_columns(self.transformed_df)
        # 3) Generate pairwise interactions
        self.transformed_df = pairwise_metafeature_generation(self.transformed_df)
        # 4) Apply standard scaling
        self.transformed_df = feature_scaling_standard(self.transformed_df)

        return self
    
    def feature_transform(self) -> "automated_feature_engineering": # Perform feature transformation but without Pairwise Metafeature generation
        """
        Example advanced transformations:
          1) Frequency encoding for categorical columns
          2) Transform boolean columns to numeric importance
          3) Standard scaling (z-score)

        Returns
        -------
        self : automated_feature_engineering
        """
        print("=========================[STEP 3]: TRANSFORMING FEATURES...=========================\n")

        # 1) Frequency encode categorical columns
        self.transformed_df = frequency_encoding(self.transformed_df)
        # 2) Convert boolean columns to numeric weighting
        self.transformed_df = transform_boolean_columns(self.transformed_df)
        # 3) Apply standard scaling
        self.transformed_df = feature_scaling_standard(self.transformed_df)

        return self

    # -------------------------------------------------------------------------
    # Feature Reduction
    # -------------------------------------------------------------------------
    def feature_reduction(self) -> "automated_feature_engineering":
        """
        Example: use Ant Colony Optimization to find a subset of features 
        that yields good clustering performance (CHI vs DBI).
        The pipeline can optionally store or log the best subset, 
        then reduce `self.df` to only those columns.

        Returns
        -------
        self : AutomatedPipeline
        """
        print("=========================[STEP 4]: PERFORMING FEATURE SEARCH...=========================\n")

        best_feats, best_score, best_k = ant_colony_optimization_search(self.transformed_df)

        # Store meta info
        self.meta_info["best_features"] = best_feats
        self.meta_info["best_fitness"] = best_score
        self.meta_info["best_k"] = best_k

        # Optionally reduce self.df to those best_feats
        final_cols = [col for col in best_feats if col in self.transformed_df.columns]
        self.transformed_df = self.transformed_df[final_cols]

        return self

    # -------------------------------------------------------------------------
    # Master Orchestrator
    # -------------------------------------------------------------------------
    def run_pipeline(self, use_gpt=True, do_metafeature_engineering=True, do_aco=True) -> pd.DataFrame:
        """
        Master method that calls each step in a typical sequence:
          1) Data Cleaning
          2) (Optional) GPT transformations
          3) (Optional) Meta-Feature transformations
          4) (Optional) Feature reduction (Ant Colony)
          5) Return final DataFrame

        Parameters
        ----------
        use_gpt : bool, optional
            Whether to run GPT-based transformations.
        do_feature_engineering : bool, optional
            Whether to run frequency encoding, boolean weighting, etc.
        do_aco : bool, optional
            Whether to run the ant_colony_optimization_search for feature selection.

        Returns
        -------
        pd.DataFrame
            The fully processed DataFrame after all transformations.
        """
        # 1) Data Cleaning
        self.clean_data()

        # 2) GPT transformations
        if use_gpt:
            self.gpt_transform(use_checklist=True)

        # 3) Feature transformations
        if do_metafeature_engineering:
            self.meta_feature_transform()
        else:
            self.feature_transform()

        # 4) Feature reduction with Ant Colony
        if do_aco:
            self.feature_reduction()

        print("=========================[DONE]=========================")
        # Return the final DataFrame
        return self.transformed_df
