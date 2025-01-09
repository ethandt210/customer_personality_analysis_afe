# component_normalization.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def component_normalization(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """
    Performs PCA on the input DataFrame with up to n_components=10,
    then calculates an IQR-based ratio for each PCA component.
    Components deemed 'too skewed' based on the ratio cutoff are excluded.
    It will return 3 'valid' PCA components as a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame (already preprocessed/scaled if needed).
    n_components : int, optional
        Number of initial PCA components to compute. Default is 10.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 3 valid PCA components.
    """

    pca = PCA(n_components=n_components)
    pca_fit = pca.fit_transform(df)

    col_names = [f"comp{i+1}" for i in range(n_components)]
    pca_ds = pd.DataFrame(pca_fit, columns=col_names)

    Q1 = pca_ds.quantile(0.25)
    Q3 = pca_ds.quantile(0.75)
    IQR = Q3 - Q1
    ranges = pca_ds.max() - pca_ds.min()

    iqr_ratios = ranges / IQR.replace(0, np.nan)
    median_iqr_ratio = iqr_ratios.median()
    std_iqr_ratio = iqr_ratios.std()
    cutoff = median_iqr_ratio + std_iqr_ratio

    valid_cols = iqr_ratios[iqr_ratios <= cutoff].index.tolist()

    # If none pass the cutoff, fallback to the 3 with the lowest ratio
    if not valid_cols:
        valid_cols = iqr_ratios.sort_values().index.tolist()[:3]

    valid_cols = valid_cols[:3]
    df_pca = pca_ds[valid_cols]

    return df_pca
