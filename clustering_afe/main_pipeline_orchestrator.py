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
    pairwise_feature_generation,
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
        self.df = df.copy()
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
        self.df = drop_single_value_columns(self.df)
        # 2) Impute missing
        self.df = impute_missing_values(self.df)
        # 3) Convert to boolean
        self.df = convert_to_boolean(self.df)
        # 4) Winsorize outliers (clamp p1/p99)
        self.df = winsorize_outliers(self.df)

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

        # Build the prompt from self.df
        prompt = build_prompt_from_df(self.df, use_checklist=use_checklist)
        # Ask GPT to produce transformation code
        code_response = call_gpt_for_transformation(prompt)

        # Execute the code blocks from GPT on self.df
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

        df_local = self.df.copy()
        code_snippets = re.findall(r"<start_code>\n(.*?)\n<end_code>", gpt_code, re.DOTALL)

        if not code_snippets:
            # Possibly GPT returned no code blocks, handle gracefully
            print("No <start_code>...<end_code> blocks found in GPT response.")
            return

        local_scope = {"df": df_local, "pd": pd, "np": np}
        for snippet in code_snippets:
            try:
                print(f"Executing Code: {snippet}\n")
                exec(snippet, {}, local_scope)
            except Exception as e:
                print(f"Error executing GPT code snippet: {e}")

        self.df = local_scope["df"]

    # -------------------------------------------------------------------------
    # Feature Transformations
    # -------------------------------------------------------------------------
    def feature_transforms(self) -> "automated_feature_engineering":
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
        print("=========================[STEP 3]: TRANSFORMING FEATURES...=========================\n")

        # 1) Frequency encode categorical columns
        self.df = frequency_encoding(self.df)
        # 2) Convert boolean columns to numeric weighting
        self.df = transform_boolean_columns(self.df)
        # 3) Generate pairwise interactions
        self.df = pairwise_feature_generation(self.df)
        # 4) Apply standard scaling
        self.df = feature_scaling_standard(self.df)

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

        best_feats, best_score, best_k = ant_colony_optimization_search(self.df)

        # (Optional) Store meta info
        self.meta_info["best_features"] = best_feats
        self.meta_info["best_fitness"] = best_score
        self.meta_info["best_k"] = best_k

        # Optionally reduce self.df to those best_feats
        final_cols = [col for col in best_feats if col in self.df.columns]
        self.df = self.df[final_cols]

        return self

    # -------------------------------------------------------------------------
    # Master Orchestrator
    # -------------------------------------------------------------------------
    def run_pipeline(self, use_gpt=True, do_feature_engineering=True, do_aco=True) -> pd.DataFrame:
        """
        Master method that calls each step in a typical sequence:
          1) Data Cleaning
          2) (Optional) GPT transformations
          3) (Optional) Feature transformations
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
        if do_feature_engineering:
            self.feature_transforms()

        # 4) Feature reduction with Ant Colony
        if do_aco:
            self.feature_reduction()

        print("=========================[DONE]=========================")
        # Return the final DataFrame
        return self.df
