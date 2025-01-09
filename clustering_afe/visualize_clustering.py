# visualize_clustering.py

import plotly.graph_objects as go
import pandas as pd

def visualize_clustering(pca_df: pd.DataFrame, 
                         x_col: str = "comp1", 
                         y_col: str = "comp2", 
                         z_col: str = "comp3", 
                         cluster_col: str = "cluster",
                         chart_title: str = "Clustering Result") -> None:
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
    chart_title: str, optional
        Title of the chart

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

    comp_cols = [col for col in pca_df.columns if col.startswith("comp")]
    if len(comp_cols) < 3:
        raise ValueError(
            f"Need at least 3 'comp*' columns in df_pca for a 3D plot. Found: {comp_cols}"
        )
    comp_cols.sort()
    x_col, y_col, z_col = comp_cols[:3]

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
            text=chart_title,
            x=0.5,  
            y=0.95,  
            xanchor='center',
            yanchor='top',
            font=dict(size=20)  
        ),
        scene=dict(
            xaxis=dict(title="Component 1", zeroline=False),
            yaxis=dict(title="Component 2", zeroline=False),
            zaxis=dict(title="Component 3", zeroline=False),
            aspectmode="cube",
        ),
        width=700, 
        height=700,  
        margin=dict(l=0, r=0, b=0, t=100),  
        legend=dict(
            font=dict(size=12),  
            itemsizing='constant',  
            orientation="h",  
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
