from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import streamlit as st

def ant_colony_optimization_search(
    df: pd.DataFrame,
    n_ants: int = 20,
    max_iter: int = 100,
    cluster_range: tuple = (2, 5),
    w_chi: float = 1.0,
    w_dbi: float = 100.0,
    random_state: int = 0
):
    """
    Ant Colony Optimization (ACO) for feature selection with CHI and DBI as evaluation metrics.

    This function attempts to find a "best" subset of features by:
      1) Randomly selecting subsets of features (ants).
      2) Clustering (KMeans) within a range of possible cluster counts.
      3) Evaluating each subset based on:
         - Calinski-Harabasz Index (CHI), which we want to maximize.
         - Davies-Bouldin Index (DBI), which we want to minimize.
      4) Aggregating these metrics into a single fitness score = (w_chi * CHI) - (w_dbi * DBI).
      5) Updating pheromones on features in subsets with high fitness.
      6) Repeating for multiple iterations to converge to a strong feature subset.

    Parameters
    ----------
    df : pd.DataFrame
        The input data, typically numeric columns only. Non-numeric columns should be encoded or removed.
    n_ants : int, optional
        Number of ants (subsets to evaluate each iteration). Default is 20.
    max_iter : int, optional
        Maximum number of ACO iterations. Default is 100.
    cluster_range : tuple, optional
        The range of cluster counts (min_k, max_k) for KMeans. Default is (2, 5).
    w_chi : float, optional
        Weight for the Calinski-Harabasz score. Default is 1.0.
    w_dbi : float, optional
        Weight for the Davies-Bouldin score. Default is 1.0.

    Returns
    -------
    best_subset : list
        The best subset of features found by ACO.
    best_fitness : float
        The highest fitness score achieved.
    best_k : int
        The cluster size (within cluster_range) that yielded the best fitness for the best subset.

    Notes
    -----
    - By default, the number of features to select is set to 5% of the total columns, 
      but if 5% is less than 10, we enforce a minimum of 10 features.
    - Each ant randomly selects features with probability weighted by current pheromones.
    - After evaluating each subset, pheromones are updated to encourage selecting features that led to better fitness.
    """
    max_features = max(int(0.05 * len(df.columns)), min(10, len(df.columns)))

    n_features = df.shape[1]
    pheromones = np.ones(n_features) / n_features
    best_subset = []
    best_fitness = float('-inf')
    best_k = None

    numpy_random = np.random.default_rng(seed=random_state)

    st.subheader("🔍 ACO Optimization Process")
    
    # Create a scrollable container with a fixed height
    with st.container():
        aco_log = st.markdown("<div style='height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;' id='log-container'></div>", unsafe_allow_html=True)

    log_history = ""  # Store logs as a string

    for iteration in range(max_iter):
        subsets = []
        fitness_scores = []
        cluster_counts = []

        log_entry = f"<b>Iteration {iteration+1}/{max_iter}</b> - "

        for _ in range(n_ants):
            selected_features = numpy_random.choice(
                df.columns, max_features, replace=False, p=pheromones / sum(pheromones)
            )
            subsets.append(selected_features)

            cluster_data = df[list(selected_features)]
            df_reduced = PCA(n_components=3).fit_transform(cluster_data)
            best_local_fitness = float('-inf')
            best_local_k = None

            for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_reduced)
                chi_score = calinski_harabasz_score(df_reduced, kmeans.labels_)
                dbi_score = davies_bouldin_score(df_reduced, kmeans.labels_)
                fitness = (w_chi * chi_score) - (w_dbi * dbi_score)

                if fitness > best_local_fitness:
                    best_local_fitness = fitness
                    best_local_k = n_clusters

            fitness_scores.append(best_local_fitness)
            cluster_counts.append(best_local_k)

        max_fitness = max(fitness_scores)
        best_k_for_iteration = cluster_counts[fitness_scores.index(max_fitness)]
        
        log_entry += f"Best Score: {max_fitness:.4f}, Best k - cluster: {best_k_for_iteration}"

        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_subset = subsets[fitness_scores.index(max_fitness)]
            best_k = best_k_for_iteration
            log_entry += f" 🎉 <b>New Best Found!</b>"

            # Show selected features in a collapsible section
            log_entry += f"""<br><details><summary>🔹 <b>Selected Features</b> (Click to Expand)</summary>
                             <pre>{best_subset}</pre></details>"""

        log_history += log_entry + "<hr>"   # Keep history of all logs

        # Update the container dynamically
        aco_log.markdown(f"<div style='height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>{log_history}</div>", unsafe_allow_html=True)

        for i, feature in enumerate(df.columns):
            feature_in_subsets = [feature in subset for subset in subsets]
            pheromones[i] += np.sum([fitness for fitness, selected in zip(fitness_scores, feature_in_subsets) if selected])

    return best_subset, best_fitness, best_k

