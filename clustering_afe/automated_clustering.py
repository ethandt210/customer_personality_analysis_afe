# automated_clustering.py

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from .component_normalization import component_normalization
from .visualize_clustering import visualize_clustering

class automated_clustering:
    """
    Orchestrates component normalization (PCA with IQR ratio filtering), 
    KMeans clustering on PCA data, and 3D visualization in a single class.
    """

    def __init__(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be used for clustering. Should be preprocessed/scaled as needed.
        """
        self.df = df  # original DataFrame
        self.df_pca = None  # will hold up to 3 valid PCA components
        self.chi_score = None
        self.dbi_score = None

    def run_component_normalization(self, n_components=10):
        """
        Runs PCA on self.df, filters 'skewed' components with IQR logic, 
        and stores up to 3 valid components in self.df_pca.
        """
        self.df_pca = component_normalization(self.df, n_components=n_components)
        return self.df_pca

    def cluster_pca_kmeans(self, n_clusters, random_state=0):
        """
        Runs KMeans on the PCA DataFrame (self.df_pca) and assigns 
        the same cluster labels to the original DataFrame (self.df).

        Parameters
        ----------
        n_clusters : int
            Number of clusters for KMeans.
        random_state : int, optional
            Random seed for reproducibility, by default 0.

        Returns
        -------
        self.df_pca : pd.DataFrame
            PCA DataFrame with an added 'cluster' label column.
        (chi_score, dbi_score) : tuple of floats
            Calinski-Harabasz and Davies-Bouldin scores for the clustering.
        """
        if self.df_pca is None:
            raise ValueError(
                "self.df_pca is None. Please call run_component_normalization() first."
            )

        kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans_model.fit_predict(self.df_pca.values)

        self.df_pca["cluster"] = labels

        self.chi_score = calinski_harabasz_score(self.df_pca.values, labels)
        self.dbi_score = davies_bouldin_score(self.df_pca.values, labels)

        return self.df_pca, (self.chi_score, self.dbi_score)

    def visualize_clusters(self, cluster_col="cluster", chart_title="Clustering Result"):
        """
        Uses the visualize_clustering function to plot the 3D PCA data with cluster labels.

        Parameters
        ----------
        cluster_col : str, optional
            Name of the column in self.df_pca containing cluster IDs, by default 'Cluster'.
        """
        if self.df_pca is None:
            raise ValueError("No PCA data found. Call run_component_normalization() first.")
        if cluster_col not in self.df_pca.columns:
            raise ValueError(f"Column '{cluster_col}' not found in self.df_pca.")

        visualize_clustering(
            pca_df=self.df_pca,
            x_col="comp1",
            y_col="comp2",
            z_col="comp3",
            cluster_col=cluster_col,
            chart_title=chart_title
        )
