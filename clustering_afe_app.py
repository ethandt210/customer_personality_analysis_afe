import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time

# Importing classes and functions from local repository
from clustering_afe.automated_clustering import automated_clustering
from clustering_afe.visualize_clustering import visualize_clustering
from clustering_afe.main_pipeline_function import meta_feature_transform, feature_reduction, feature_transform
from clustering_afe.feature_importance_selection import select_important_features, plot_feature_importance_parallel
from clustering_afe.data_cleaning import drop_single_value_columns, impute_missing_values, convert_to_boolean, winsorize_outliers

st.set_page_config(page_title='Automated Feature Engineering & Clustering', layout='wide')
st.title('📊 Automated Feature Engineering & Clustering Framework')

with st.expander("📖 **Documentation & Usage Guide**", expanded=True):
    st.markdown("""
    ## 💡 **Framework Overview**
    This web-based tool allows users to perform automated feature engineering and clustering analysis, including:
    - **Data Cleaning**: Handle missing values, remove single-value columns, and standardize data.
    - **Feature Engineering**: Perform transformations, meta-feature generation, and feature selection using ACO.
    - **Clustering Analysis**: PCA for dimensionality reduction and KMeans clustering with performance metrics.
    - **Visualization**: Interactive 3D plots and parallel coordinates plots for interpretability.

    ## 🚀 **How to Use**
    1. Upload your dataset in `.csv` format.
    2. Select processing options from the sidebar.
    3. Click **Run Analysis** and explore the outputs.
    4. Download the processed data using the provided button at the end.
    """)

# Sidebar options
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

use_gpt = st.sidebar.toggle("Use GPT for transformation", value=False)
do_metafeature = st.sidebar.toggle("Perform Metafeature Generation", value=True)
do_aco = st.sidebar.toggle("Use ACO for Feature Selection", value=True)

# Expandable section for detailed parameter settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    # Slider for outlier thresholds
    st.markdown("#### Outlier Winsorization Thresholds")
    lower_threshold = st.slider("Lower percentile threshold:", min_value=0.0, max_value=25.0, value=1.0, step=1.0)
    upper_threshold = st.slider("Upper percentile threshold:", min_value=75.0, max_value=100.0, value=99.0, step=1.0)

    st.markdown("### Ant Colony Optimization Parameters")
    n_ants = st.number_input("ACO: Number of Ants", min_value=5, max_value=50, value=20)
    max_iter = st.number_input("ACO: Max Iterations", min_value=10, max_value=500, value=100)
    w_chi = st.number_input("ACO: Weight for CHI", min_value=0.01, max_value=5.0, value=1.0)
    w_dbi = st.number_input("ACO: Weight for DBI", min_value=0.1, max_value=1000.0, value=100.0)

    

gpt_api_key = None
if use_gpt:
    gpt_api_key = st.sidebar.text_input("Enter GPT API Key", type="password")

# Run button to execute the pipeline
if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")
    if st.sidebar.button("Run Analysis"):
        df = pd.read_csv(uploaded_file)

        # Progress for Data Processing Pipeline
        with st.spinner("Running Data Processing & Feature Engineering..."):
            progress_text = "Starting data cleaning..."
            progress_bar = st.progress(0, text=progress_text)

            # Step 1: Data Cleaning
            st.subheader("1️⃣ Data Cleaning")
            progress_text = "🔄 Cleaning data..."
            progress_bar.progress(20, text=progress_text)
            time.sleep(1)

            # 1) Drop single-value columns
            st.write("\n🔄 Dropping single-value columns...")
            before_cols = df.shape[1]
            df_not_single_value = drop_single_value_columns(df)
            after_cols = df_not_single_value.shape[1]
            st.write(f"Removed {before_cols - after_cols} single-value columns. Remaining columns: {after_cols}")

            # 2) Impute missing values
            st.write("\n🔄 Imputing missing values...")
            missing_before = df_not_single_value.isnull().sum().sum()
            df_is_not_null = impute_missing_values(df_not_single_value)
            missing_after = df_is_not_null.isnull().sum().sum()
            st.write(f"Missing values before: {missing_before}, after: {missing_after}")

            # 3) Convert columns to boolean
            st.write("\n🔄 Converting eligible columns to boolean...")
            bool_before = df_is_not_null.select_dtypes(include=[bool]).shape[1]
            df_bool = convert_to_boolean(df_is_not_null)
            bool_after = df_bool.select_dtypes(include=[bool]).shape[1]
            st.write(f"Converted {bool_after - bool_before} columns to boolean.")

            # 4) Winsorize outliers with adjustable thresholds
            st.write("\n🔄 Winsorizing outliers with adjustable thresholds...")
            df_non_outlier, outlier_counts = winsorize_outliers(df_bool, lower_threshold/100, upper_threshold/100)
            for col, (before, after) in outlier_counts.items():
                st.write(f"Column {col}: Outliers before={before}, after={after}")


            st.subheader("2️⃣ Feature Transformation")
            # Step 2: GPT Transformation
            progress_val = 40
            if use_gpt:
                progress_text = "🔄 Applying GPT transformation..."
                progress_bar.progress(progress_val, text=progress_text)
                time.sleep(1)
                from clustering_afe.gpt_transformation import config_client, build_prompt_from_df, call_gpt_for_transformation, get_df_attribute
                st.write("\n🔄 Transforming Data Contextually using GPT...")
                gpt_df = df_non_outlier.copy()
                config_client(gpt_api_key)
                prompt = build_prompt_from_df(gpt_df, use_checklist=True)
                gpt_code = call_gpt_for_transformation(prompt)
                code_snippets = re.findall(r"<start_code>\n(.*?)\n<end_code>", gpt_code, re.DOTALL)

                if not code_snippets:
                    st.error("No <start_code>...<end_code> blocks found in GPT response.")

                local_scope = {"df": gpt_df, "pd": pd, "np": np}
                for i, snippet in enumerate(code_snippets):
                    try:
                        max_lines = 10  # Adjust based on how much you want to show
                        snippet_lines = snippet.strip().split("\n")
                        displayed_code = "\n".join(snippet_lines[:max_lines])
                        if len(snippet_lines) > max_lines:
                            displayed_code += "\n... (truncated)"

                        st.markdown(f"#### **Executing Code Snippet {i+1}/{len(code_snippets)}**")
                        st.code(displayed_code, language='python')

                        # ✅ Execute the code snippet
                        exec(snippet, {}, local_scope)

                        st.success(f"✅ Snippet {i+1} executed successfully.")
                    except Exception as e:
                        st.error(f"❌ Error executing snippet {i+1}: {e}")
                gpt_df = local_scope["df"]
                st.write("After GPT Transformation:", gpt_df.head())
                download_df = gpt_df.copy()
            else:
                gpt_df = df_non_outlier.copy()
                download_df = gpt_df.copy()

            # Step 3: Feature Transformations
            progress_val = 60
            if do_metafeature:
                progress_text = "🔄 Generating meta-features..."
                progress_bar.progress(progress_val, text=progress_text)
                time.sleep(1)
                st.write("\n🔄 Encoding, Scaling, and Transforming Features...")
                transformed_df = meta_feature_transform(gpt_df)
            else:
                progress_text = "🔄 Transforming features..."
                progress_bar.progress(progress_val, text=progress_text)
                time.sleep(1)
                st.write("\n🔄 Encoding, Scaling, and Transforming Features...")
                transformed_df = feature_transform(gpt_df)

            # Step 4: Feature Reduction using ACO
            if do_aco:
                progress_val = 80
                progress_text = "🔄 Performing Ant Colony Optimization for feature selection..."
                progress_bar.progress(progress_val, text=progress_text)
                time.sleep(1)
                st.write("\n🔄 Selecting Feature using Ant Colony Optimization...")
                aco_selected_df, best_feat, best_score, best_k = feature_reduction(transformed_df)
                df_processed = aco_selected_df.copy()
            else:
                df_processed = transformed_df.copy()
                best_k = 3

            progress_bar.progress(100, text="✅ Feature Engineering Complete!")
            st.success("Feature Engineering Pipeline Completed Successfully!")

        st.write("Processed Dataset:", df_processed.head())

        st.write(f"Optimal number of clusters (k): {best_k}")
        # Clustering with PCA + KMeans
        with st.spinner("Running PCA and Clustering..."):
            st.subheader("3️⃣ Clustering using PCA & KMeans")
            clustering_model = automated_clustering(df_processed)
            df_pca = clustering_model.run_component_normalization()
            df_pca, (chi_score, dbi_score) = clustering_model.cluster_pca_kmeans(n_clusters=best_k)

        # Attach Cluster Labels
        st.write("Clustered Component Data:", df_pca.head())

        # Display CHI and DBI scores
        st.subheader("📊Clustering Metrics")
        st.success(f"**Calinski-Harabasz Index (CHI):** {chi_score:.2f}")
        st.success(f"**Davies-Bouldin Index (DBI):** {dbi_score:.2f}")

        with st.spinner("Generating 3D Clustering Plot..."):
            st.subheader("4️⃣ 3D Clustering Plot")
            fig = visualize_clustering(df_pca, cluster_col='cluster')

            st.plotly_chart(fig, use_container_width=True)


            labeled_df = download_df.copy()
            labeled_df['cluster'] = KMeans(n_clusters=best_k).fit_predict(df_pca)

        # Feature Selection and Parallel Coordinates Plot
        with st.spinner("Selecting important features and generating parallel coordinates plot..."):
            st.subheader("Feature Importance using CatBoost & Parallel Coordinates Plot")
            top_features = select_important_features(labeled_df, cluster_col='cluster', top_n=5)
            st.write(f"Top {5} Important Features:", top_features)

            st.markdown("### Cluster Movement Through Important Features")
            fig = plot_feature_importance_parallel(labeled_df, cluster_col="cluster", top_n=5)
            st.pyplot(fig)        

        # Download Results
        st.download_button(
            label="⬇️ Download Clustered & Transformed Data",
            data=download_df.to_csv(index=False).encode('utf-8'),
            file_name="clustered_data.csv",
            mime='text/csv',
        )
        st.success("All processes completed successfully.")