# visualize_clustering.py

import plotly.graph_objects as go
import pandas as pd

def visualize_clustering(pca_df: pd.DataFrame, 
                         x_col: str = "comp1", 
                         y_col: str = "comp2", 
                         z_col: str = "comp3", 
                         cluster_col: str = "cluster") -> None:
    """
    Creates a 3D scatter plot of PCA or dimension-reduced data with cluster labels using Plotly.

    Parameters
    ----------
    pca_df : pd.DataFrame
        A DataFrame containing at least 3 columns for x, y, z coordinates (e.g., 'comp1', 'comp2', 'comp3')
        plus a column for cluster labels (default 'Clusters').
    x_col : str, optional
        The column name representing the X-axis (default 'comp1').
    y_col : str, optional
        The column name representing the Y-axis (default 'comp2').
    z_col : str, optional
        The column name representing the Z-axis (default 'comp3').
    cluster_col : str, optional
        The column name representing cluster labels (default 'Clusters').

    Returns
    -------
    None
        Displays an interactive 3D scatter plot in a notebook or script.
    """

    color_map = {
        0: "rgb(23, 190, 207)",
        1: "rgb(255, 127, 14)",
        2: "rgb(44, 160, 44)",
        3: "rgb(214, 39, 40)",
        4: "rgb(148, 103, 189)",
        5: "rgb(140, 86, 75)",
        6: "rgb(227, 119, 194)",
        7: "rgb(127, 127, 127)",
    }

    data = []
    if cluster_col not in pca_df.columns:
        raise ValueError(f"The DataFrame must contain the cluster label column '{cluster_col}'.")

    for cluster_id, group in pca_df.groupby(cluster_col):
        scatter = go.Scatter3d(
            mode="markers",
            name=f"Cluster {cluster_id}",
            x=group[x_col],
            y=group[y_col],
            z=group[z_col],
            marker=dict(
                size=4,
                color=color_map.get(cluster_id, "rgb(128, 128, 128)"),
                opacity=1,
                line=dict(width=0.5, color="DarkSlateGrey")
            )
        )
        data.append(scatter)

    layout = go.Layout(
        title=dict(
            text="Clustering Visualization in 3D",
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title=f"{x_col}", zeroline=False),
            yaxis=dict(title=f"{y_col}", zeroline=False),
            zaxis=dict(title=f"{z_col}", zeroline=False),
            aspectmode="cube"  # ensures equal scaling across axes
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            xanchor="center"
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
