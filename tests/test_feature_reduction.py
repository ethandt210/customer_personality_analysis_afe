# tests/test_feature_reduction.py

import pytest
import pandas as pd
import numpy as np
from clustering_afe.feature_reduction import ant_colony_optimization_search

def test_ant_colony_optimization_search_expand_columns():
    """
    Provide a DataFrame with >= 10 columns, allowing the ACO method to 
    sample up to 10 features safely without raising a 'Cannot take a larger 
    sample than population...' error.
    """

    df = pd.DataFrame(
        np.random.randn(5, 12), #need more than 10 columns to pass
        columns=[f"feat{i}" for i in range(1, 13)]
    )

    # Minimal ACO parameters for quick test
    best_feats, best_fitness, best_k = ant_colony_optimization_search(
        df,
        n_ants=2,
        max_iter=2,
        cluster_range=(2, 3),
        w_chi=1.0,
        w_dbi=1.0
    )

    # Basic validations
    assert len(best_feats) > 0, "ACO should pick at least one feature."
    assert isinstance(best_fitness, float), "best_fitness should be float."
    assert best_k in [2, 3], "best_k must be within defined cluster range."