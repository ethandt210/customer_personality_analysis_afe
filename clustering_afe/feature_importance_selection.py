import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.express as px
import itertools

def select_important_features(original_df, cluster_col="cluster", top_n=6):
    """
    Identifies the most important features using a Decision Tree + SHAP values.
    
    Parameters
    ----------
    original_df : pd.DataFrame
        The dataset before PCA transformation.
    cluster_col : str, optional
        Column name for cluster labels (default is 'cluster').
    top_n : int, optional
        Number of top features to select (default is 6).
    
    Returns
    -------
    List[str]
        The names of the top important features.
    """
    df_plot = original_df.copy()
    
    cat_cols = [col for col in df_plot.select_dtypes(include=['object', 'category']).columns if col != cluster_col]
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_plot[col] = le.fit_transform(df_plot[col])
        label_encoders[col] = dict(zip(le.classes_, range(len(le.classes_))))
    
    X = df_plot.drop(columns=[cluster_col])
    y = df_plot[cluster_col]
    
    clf = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
    clf.fit(X, y, cat_features=[X.columns.get_loc(col) for col in cat_cols if col in X.columns])
    
    explainer = shap.Explainer(clf)
    shap_values = explainer.shap_values(X)
    
    shap_values = np.array(shap_values)
    shap_values_agg = np.mean(np.abs(shap_values), axis=2)
    
    feature_importance = shap_values_agg.mean(axis=0)
    
    if feature_importance.shape[0] != len(X.columns):
        raise ValueError(f"Mismatch: SHAP values {feature_importance.shape[0]} != Features {len(X.columns)}")
    
    feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
    
    top_features = feature_importance_df.head(min(top_n, len(feature_importance_df)))['feature'].tolist()
    
    return top_features

def plot_feature_importance_parallel(original_df, cluster_col="cluster", top_n=5):
    """
    Plots a Parallel Coordinates chart with the most important features and returns the figure object.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object for Streamlit rendering.
    """
    selected_features = select_important_features(original_df, cluster_col, top_n)
    selected_features.append(cluster_col)
    
    df_plot = original_df[selected_features].copy()
    cat_cols = [col for col in df_plot.select_dtypes(include=['object', 'category']).columns if col != cluster_col]
    label_mappings = {}
    
    scaler = MinMaxScaler()
    for col in selected_features:
        if col in cat_cols:
            le = LabelEncoder()
            df_plot[col] = le.fit_transform(df_plot[col])
            label_mappings[col] = dict(zip(le.transform(le.classes_), le.classes_))
    
    features_to_scale = [col for col in selected_features if col != cluster_col]
    df_plot[features_to_scale] = scaler.fit_transform(df_plot[features_to_scale])
    
    df_plot[cluster_col] = df_plot[cluster_col].astype(int)
    df_plot[cluster_col] = df_plot[cluster_col].apply(lambda x: f"Cluster {x}")
    sorted_cluster_labels = sorted(df_plot[cluster_col].unique(), key=lambda x: int(x.split(' ')[1]))
    df_plot[cluster_col] = pd.Categorical(df_plot[cluster_col], categories=sorted_cluster_labels, ordered=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    parallel_coordinates(df_plot, class_column=cluster_col, colormap='viridis', alpha=0.35, ax=ax)
    ax.set_title("Parallel Coordinates Plot of Top Important Features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: int(labels[i].split(' ')[1]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right')
    
    for line in ax.get_legend().get_lines():
        line.set_linewidth(3)
        line.set_alpha(1.0)
    
    return fig

def plot_feature_correlation_matrix(df: pd.DataFrame, cluster_col: str = "cluster", top_n = 5):
    """
    Creates a correlation matrix-like scatter plot with all selected features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the selected original features and cluster labels.
    selected_features : list
        A list of the top selected features (exactly 5).
    cluster_col : str, optional
        The column name representing cluster labels (default 'cluster').

    Returns
    -------
    plt.Figure
        A matplotlib figure object containing the correlation matrix-like plot.
    """

    selected_features = select_important_features(df, cluster_col, top_n)
    # selected_features = ['Total_Spending','Total_Purchases','Income','Campaign_Acceptance_Rate','Proportion_MntWines']
    selected_features.append(cluster_col)
    
    df_plot = df[selected_features].copy()
    cat_cols = [col for col in df_plot.select_dtypes(include=['object', 'category']).columns if col != cluster_col]
    for col in selected_features:
        if col in cat_cols:
            le = LabelEncoder()
            df_plot[col] = le.fit_transform(df_plot[col])
    
    g = sns.pairplot(
        df_plot,
        hue=cluster_col,
        diag_kind="hist",
        palette="viridis",
        plot_kws={"alpha": 0.6},
        corner=True
    )

    g.fig.suptitle("Feature Correlation Matrix", y=1.08, fontsize=14) 
    g.fig.set_size_inches(8, 6)

    for ax in g.axes.flat:
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=30, ha="right")
            ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=30, ha="right")
            ax.tick_params(axis="x", labelrotation=30)
            ax.tick_params(axis="y", labelrotation=30) 

    return g.fig

    # selected_features = select_important_features(df, cluster_col, top_n)
    # selected_features.append(cluster_col)

    # df_plot = df[selected_features].copy()
    # cat_cols = [col for col in df_plot.select_dtypes(include=['object', 'category']).columns if col != cluster_col]
    # for col in selected_features:
    #     if col in cat_cols:
    #         le = LabelEncoder()
    #         df_plot[col] = le.fit_transform(df_plot[col])

    # num_features = len(selected_features) - 1
    # fig, axes = plt.subplots(num_features, num_features, figsize=(4*num_features, 4*num_features))

    # for i, feature_x in enumerate(selected_features[:-1]):
    #     for j, feature_y in enumerate(selected_features[:-1]):
    #         ax = axes[i, j]

    #         if i == j:
    #             sns.histplot(data=df_plot, x=feature_x, hue=cluster_col, kde=True, ax=ax, palette="viridis")
    #         else:
    #             sns.kdeplot(
    #                 data=df_plot,
    #                 x=feature_y,
    #                 y=feature_x,
    #                 hue=cluster_col,
    #                 fill=True,
    #                 ax=ax,
    #                 palette="viridis",
    #                 alpha=0.5
    #             )

    #         if i < num_features - 1:
    #             ax.set_xlabel('')
    #         else:
    #             ax.set_xlabel(feature_y, fontsize=12)

    #         if j > 0:
    #             ax.set_ylabel('')
    #         else:
    #             ax.set_ylabel(feature_x, fontsize=12)

    #         ax.tick_params(axis='x', rotation=30)
    #         ax.tick_params(axis='y', rotation=30)

    # plt.tight_layout()
    # fig.suptitle("KDE Feature Correlation Matrix", y=1.02, fontsize=16)

    # return fig


