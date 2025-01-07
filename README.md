# **customer_personality_analysis_afe**
**Automated Feature Engineering Framework for Customer Personality Analysis in the Retail Industry**

## Overview
This repository provides a one-stop solution for **data cleaning**, **GPT-powered transformations**, **feature engineering**, and **feature reduction**—all tuned specifically for **Clustering tasks**, useful for **Customer Personality Analysis** in a retail setting.

With a single orchestrator pipeline, users can:
- Drop or impute problematic columns,
- Automatically generate Python code via GPT to transform their DataFrame,
- Engineer advanced features (e.g., frequency encoding, pairwise interactions, scaling),
- Perform Ant Colony Optimization (ACO) for feature reduction.

## Key Features
1. **Data Cleaning**
   - Drop single-value columns
   - Impute missing values (median/mode)
   - Convert binary columns to boolean
   - Winsorize outliers at 1st/99th percentile

2. **GPT Transformations**
   - Summarize DataFrame schema and produce a “checklist”
   - Generate dynamic Python code in `<start_code> ... <end_code>` blocks
   - Execute GPT-generated transformations automatically on your DataFrame

3. **Feature Engineering**
   - Frequency encode categorical columns
   - Transform boolean columns to numeric “weights”
   - Create pairwise interactions (squared, sqrt, products, divisions)
   - Standard scaling for numerical features

4. **Feature Reduction**
   - **Ant Colony Optimization** to pick an optimal subset of features
   - Evaluate subset quality using Calinski-Harabasz (CHI) and Davies-Bouldin (DBI)

5. **Orchestrator Pipeline**
   - A single class (`AutomatedPipeline`) that ties all steps into a `.run_pipeline()` call
   - Configurable toggles to skip or include GPT transformations, advanced feature ops, or ACO-based reduction

## Installation
1. **Clone or Download** this repo:
   ```bash
   git clone https://github.com/ethandt210/customer_personality_analysis_afe.git
   cd customer_personality_analysis_afe
   ```
2. **Install via** `pip`:
   ```bash
   pip install -e .
   ```

## Usage Example
```python
import pandas as pd
from clustering_afe import automated_feature_engineering

# Suppose you have a CSV file
df_raw = pd.read_csv("customer_marketing_data.csv")

# Provide your OpenAI API key to enable GPT transformations
my_api_key = "sk-YourOpenAIKeyHere"

# Create the pipeline
afe = automated_feature_engineering(df_raw, my_api_key)

# Run the entire pipeline:
#   1) Data Cleaning
#   2) GPT transformations
#   3) Feature transformations
#   4) Feature reduction (ACO)
df_final = afe.run_pipeline(
    use_gpt=True, 
    do_feature_engineering=True, 
    do_aco=True
)

print("Final DataFrame Shape:", df_final.shape)
print("Selected Features:", pipeline.meta_info.get("best_features"))
```
