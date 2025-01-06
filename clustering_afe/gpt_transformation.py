import pandas as pd
from openai import OpenAI



def config_client(api_key: str):
    """
    Configures a global OpenAI client using the provided API key.
    Must be called before any GPT-based functions (e.g., gpt_checklist, call_gpt_for_transformation).

    Parameters
    ----------
    api_key : str
        The user-provided OpenAI API key.

    Returns
    -------
    None
        Sets the global 'client' variable to an OpenAI instance.
    """

    global client
    client = OpenAI(api_key=api_key)

def get_df_attribute(df: pd.DataFrame) -> str:
    """
    Summarizes each column in the DataFrame by listing:
      - Data type
      - NaN frequency (as a percentage)
      - Number of distinct values
      - Up to 10 sample values (or less if fewer unique values exist)

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be analyzed.

    Returns
    -------
    str
        A descriptive string of columns and their sample data, suitable for GPT prompts.
    """
    attr_str = ""
    df_head = df.head(10)

    for col in df.columns:
        nan_freq = float("%.2g" % (df[col].isna().mean() * 100))
        distinct_values = df[col].nunique()

        if distinct_values < 10:
            samples = df[col].unique().tolist()
        else:
            samples = df_head[col].tolist()

        if pd.api.types.is_float_dtype(df[col]):
            samples = [round(x, 2) for x in samples]

        attr_str += (
            f"{col} ({df[col].dtype}): "
            f"NaN-freq [{nan_freq}%], "
            f"Count Distinct: {distinct_values}, "
            f"Samples {samples}\n"
        )
    return attr_str

def gpt_checklist(prompt, model= "gpt-4o", temperature = 0.25):
    """
    Calls GPT to generate a 'checklist' analyzing the DataFrame's attributes.

    The system instructions guide GPT to return a list of potential attributes
    (demographics, purchase behavior, etc.) that might exist in the data. The user
    prompt is typically the summary from get_df_attribute().

    Parameters
    ----------
    prompt : str
        The user (or system) prompt describing the DataFrame attributes.
    model : str, optional
        The GPT model to use, by default "gpt-4o".
    temperature : float, optional
        Sampling temperature, by default 0.25.

    Returns
    -------
    str
        GPT's textual response (the checklist).
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
            "content":
            '''
            Return this checklist everytime you get a prompt, assess carefuly whether the `df` description that were given to you has any of these attributes:
                ##Checklist##
                1. User Demographics
                Age (calculated from birth year or date of birth)
                Gender
                Marital Status
                Education Level
                Income (absolute values or income brackets)
                Employment Status
                Household Composition (e.g., number of kids, teens, or adults)

                2. Geographic Information
                Country
                Region/State/City
                Urban/Rural Indicator
                Distance from Store (if applicable)
                Geo-based segmentation (e.g., coastal vs. inland customers)

                3. Purchase Behavior
                Total Spending (sum across all categories)
                Spending by Category (e.g., food, beverages, electronics)
                Proportion of Spending by Category
                Frequency of Purchases (e.g., monthly, yearly)
                Average Purchase Value
                Median Purchase Value
                Maximum/Minimum Purchase Value
                Seasonality Trends (e.g., more purchases during holidays)
                Top Categories (most frequently purchased product types)

                4. Channel Behavior
                Number of Purchases per channel category
                Channel Preference Index (e.g., online vs. store purchases)
                Web Visit Frequency (e.g., number of visits per month)
                Store Visit Frequency (if applicable)

                5. Campaign Responses
                Total Campaigns Accepted
                Campaign Acceptance Rate
                Last Campaign Accepted (recency of campaign interaction)
                Response to Individual Campaigns (binary indicators or counts)
                Average Time Between Campaign Acceptance

                6. Interaction Metrics
                Recency (days since last interaction/purchase)
                Tenure (days since first interaction or enrollment)
                Loyalty Indicator (e.g., frequent buyer program participation)
                Number of Complaints
                Average Time Between Interactions

                7. Derived Financial Metrics
                Customer Lifetime Value (CLV)
                Average Revenue Per User (ARPU)
                Revenue per Purchase
                Profit Margin (if cost data is available)
                Discounts Used (total or percentage of purchases with discounts)

                8. Product Preferences
                Product Types Bought (e.g., food, wine, electronics)
                Product Diversity (number of unique categories purchased from)
                Favorite Products (highest spending categories/products)
                Product Loyalty (repeated purchases of the same product)

                9. Temporal Features
                Month of First Purchase
                Month of Last Purchase
                Weekday/Weekend Preference
                Morning/Afternoon/Evening Purchase Trends
                Holidays vs. Non-Holiday Spending Trends

                10. Custom Behavioral Indicators
                Churn Prediction Indicator (based on purchase inactivity)
                Segmentation Clusters (e.g., RFM segmentation: Recency, Frequency, Monetary value)
                Customer Tier (e.g., high-value, medium-value, low-value)
                Predictive Indicators (e.g., likelihood to respond to a campaign)

                If you think the metrics are available from the `df` then flag it next to the item in this checklist.
                e.g.:
                Age (calculated from birth year or date of birth) --> [!AVAILABLE!]. Need to transform from column ["birth_year"] in dataframe `df`
                Gender --> Not Available
                Marital Status --> Not Available
                Education Level --> Available. There are redundant values. Need to perform Binning and Grouping ["education"] in dataframe `df`
                Income (absolute values or income brackets) --> AVAILABLE. Directly from column ["Income"] in dataframe `df`

                Make sure all the item in the list are checked, no matter their availability. If it's Available, but need to be transformed first use the format [!AVAILABLE!], if it's available directly from a column just use AVAILABLE

            '''
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = temperature
    )
    return completion.choices[0].message.content 

def build_prompt_from_df(df: pd.DataFrame, use_checklist = True):
    """
    Builds a final GPT prompt combining DataFrame attributes, optional GPT-generated
    checklist, and sample code instructions.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to describe in the prompt.
    use_checklist : bool, optional
        Whether to call GPT to generate a checklist from 'df' attributes, by default True.
    data_description_unparsed : str, optional
        Additional unparsed text about the dataset (not used in code, but can be appended),
        by default "No Descriptions Available, Proceed Accordingly".

    Returns
    -------
    str
        A combined prompt string ready for GPT transformation instructions.
    """
    samples = get_df_attribute(df)

    if use_checklist:
        checklist = gpt_checklist(samples)
    else:
        checklist = "No Checklist Provided"

    return f"""
            The dataframe `df` is loaded and in memory. Columns are also named attributes.

            Columns in `df` (true feature dtypes listed here):
            "{samples}"

            Number of (rows) in dataset: {int(len(df))}

            Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.

            Don't forget our objective is to analyze customer's personality, so make sure to focus on those attributes.

            Now take a look at this checklist and for every features that are AVAILABLE but need to be transformed, do the operations accordingly:
            {checklist}

            Here are some sample codes that you should follow as a guideline:
            Follow these Code Formatting:

            For Added Columns:
            <start_code>
            # (Feature name and description)
            # Usefulness: (Explain why this feature adds real-world knowledge for clustering tasks in CPA in the retail industry.)
            # Input samples: (Provide three samples of the input columns used in the following code.)
            # Example:
            # ‘dt_birthdate’: ['2000-01-01', '1985-06-15', NaN]
            <end_code>

            Pandas code to get a user's preference in channels from df
            <start_code>
            channel_columns = ['num_channel_1_purchase', 'num_channel_2_purchase', 'num_channel_3_purchase', 'num_channel_4_purchase']
            df['mkt_channel'] = df[channel_columns].idxmax(axis=1).str.replace('num', '').str.replace('purchase', '')
            <end_code>

            Feature: Finding Customer Age
            Usefulness: Age is a critical demographic feature that aids in clustering users based on life stages and preferences. Make sure to count the age based on today's date
            <start_code>
            # Input samples: 'dt_birthdate': ['2000-01-01', '1985-06-15', NaN]
            df['age'] = (pd.to_datetime('today') - pd.to_datetime(df['dt_birthdate'], format='%d-%m-%Y')).dt.days // 365
            <end_code>

            Codeblock for Binning and Grouping a Feature's Value:
            <start_code>
            # Explanation: The column 'Education' is grouped because it is contains the same information in different values.
            df['Education']=df['Education'].replace({{'Basic':"Undergraduate","High School":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"}})
            <end_code>

            Codeblock for Skipping a Feature:
            <start_code>
            # Skipped: No columns matching the desired criteria (e.g., birthdate) were found in the dataframe.
            pass
            <end_code>

            Codeblock for Dropping a Column:
            drop columns that are too distinctively unique such as dates and User's ID or have been transformed into new ones
            <start_code>
            # Explanation: The column 'category_name' is dropped because it is redundant with aggregated features generated elsewhere.
            df.drop(columns=['user_id', 'category_name_1', 'num_spending_a', 'num_spending_b'], inplace=True)
            <end_code>

            Each codeblock ends with <end_code> and starts with <start_code>
          """


def call_gpt_for_transformation(prompt: str, model= "gpt-4o", temperature=0.25) -> str:
    """
    Calls GPT to generate code that transforms the DataFrame, based on the prompt.

    Parameters
    ----------
    prompt : str
        The user (or system) prompt describing how GPT should transform the DataFrame.
    model : str, optional
        The GPT model to use, by default "gpt-4o".
    temperature : float, optional
        Sampling temperature, by default 0.25.

    Returns
    -------
    str
        GPT's textual response, which should contain <start_code>...<end_code> blocks of Python code.
    """

    if client is None: # Making sure client is configured
        raise ValueError("OpenAI client is not configured. Call config_client(api_key) first.")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
            "content":
            '''
            You are a Python expert and data scientist specialized on Customer Personality Analysis in the Retail Industry.

            Your specific task is to process a Pandas DataFrame named `df`
            You will receive the detail attributes of each feature in `df` and a list that you will follow the instructions from the user's prompt

            Your role is limited to inferring and grouping attributes without performing further calculations or modeling tasks, as other methods will handle those tasks.
            You answer and respond only by writing Python code as a string based on inferred column names from a Pandas DataFrame (`df`). The generated code should follow these rules:

            - The output should be a Python code snippet as a string that implements the rules described above.
            - Use `df` as the variable name for the DataFrame in the generated code.
            - Make sure to follow the code format by starting with <start_code> and ending it with <end_code>
            The output should be multiple code blocks, each generating exactly one useful column that groups, aggregates, bins existing features. Each code block should adhere to the above formats, providing clarity and utility for clustering tasks

            MAKE SURE TO DO BOTH PHASE

            PHASE 1:

            Instructions for the Model, follow these instructions carefully and do not miss any of them:
            - Carefully infer column roles based on the given descriptions and dataset context.
            - Ensure the code is modular and adheres to the formats provided.
            - If a categorical column has values that are similar in definition (such as Single == Alone) based on your judgement, then group them into values that are more commonly used.
            - If you're creating a new column based on an also newly created column, make sure the order you write the column is correct, do not calculate based on a column that has not been defined earlier.

            PHASE 2:
            Finally, make sure to follow these rules after you're done with the PHASE 1:
            - If a categorical or object column is too distinctively unique (e.g. more than 10 distinct values) drop the column
            - If a column has been transformed into a new one, DROP the OLD COLUMNS (e.g.: num_spent_a + num_spent_b = total_spent, then drop the num_spent_a & num_spent_b)
            - All columns that have been transformed into a new feature must be dropped, that way there are no redundant values between all features.


            '''
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = temperature
    )
    return completion.choices[0].message.content