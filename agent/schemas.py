from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field

# ─── 1. Orchestration & Intent Metadata Schemas ───

class SubQuestion(BaseModel):
    """A single decomposed data question paired with its required execution category."""
    question: str = Field(
        ..., 
        description="The self-contained, specific data question with all pronouns resolved."
    )
    target_category: Literal[
        "SPECIALIZED_ANALYTICS_AND_MODELING", 
        "SCENARIO_SIMULATION_AND_WHAT_IF", 
        "VISUALIZATION_AND_PLOTTING", 
        "BASIC_SQL_AGGREGATION"
    ] = Field(
        ..., 
        description=(
            "The analytical category required to answer this question. "
            "RULES: If the prompt mentions regression, forecasting, clustering, PCA, Random Forest, or unit economics, MUST use 'SPECIALIZED_ANALYTICS_AND_MODELING'. "
            "If it asks 'what if', assumes new values, or simulates changes, MUST use 'SCENARIO_SIMULATION_AND_WHAT_IF'. "
            "ONLY use 'BASIC_SQL_AGGREGATION' for simple data retrieval, counts, sums, or averages."
        )
    )

class DecomposedQuestions(BaseModel):
    """The broken-down data queries based on the user's prompt, enriched with routing categories."""
    questions: List[SubQuestion] = Field(
        description="A list of specific, actionable, and categorized data queries. Max 5."
    )

# -------------------- ANALYSIS SCHEMAS --------------------

class execute_sql_query_tool(BaseModel):
    """
    Queries the Databricks database using PostgreSQL syntax. Because this is PostgreSQL, you MUST wrap all column names in double quotes to preserve exact capitalization. 
    Do NOT use this tool if a more specific tool is available for the user's request:
    - Use calculate_unit_economics_tool for CPA or CLV questions.
    - Use run_scenario_planning_tool for ANY what-if analysis, simulating changes to variables, or holding variables constant.
    - Use regression/forecasting/clustering tools for modeling.
    Do NOT attempt to write complex SQL window functions, regressions, or simulations manually if a tool exists for it.
    """
    sql_query: str = Field(
        ..., 
        description="The raw PostgreSQL query to execute."
    )

class run_ols_regression_tool(BaseModel):
    """
    Performs an Ordinary Least Squares (OLS) multiple regression. cannot perform non-linear regression. 
    Use this when the user asks to analyze the relationship, correlation, or impact of multiple independent numerical variables on a dependent target variable.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    dependent_variable: str = Field(
        ..., 
        description="The exact column name of the target numerical variable to predict (the Y variable)."
    )
    
    independent_variables: List[str] = Field(
        ..., 
        description="A list of exact column names for the numerical predictor variables (the X variables)."
    )

class run_arima_forecasting_tool(BaseModel):
    """
    Performs ARIMA time series forecasting grouped by Activation_Year and Activation_Month. 
    Use this when the user asks to predict future values based on historical trends.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    value_column: str = Field(
        ..., 
        description="The exact column name of the numerical variable to forecast (e.g., mcf, sac, temp_Id)."
    )
    
    aggregation: Optional[Literal["SUM", "AVG", "COUNT"]] = Field(
        default=None,
        description="The SQL aggregation function to apply to the value_column per month (e.g., SUM for totals, AVG for averages, COUNT for volume)."
    )
    
    steps: Optional[int] = Field(
        default=5, 
        description="The number of future months to forecast (default is 5)."
    )
    
    p: Optional[int] = Field(
        default=1, 
        description="The ARIMA model's autoregressive order (default is 1)."
    )
    
    d: Optional[int] = Field(
        default=1, 
        description="The ARIMA model's differencing order (default is 1)."
    )
    
    q: Optional[int] = Field(
        default=1, 
        description="The ARIMA model's moving average order (default is 1)."
    )

class run_pca_tool(BaseModel):
    """
    Performs Principal Component Analysis (PCA) to reduce dimensionality and find the underlying variance/patterns in a set of features. 
    Use this to identify which combinations of variables explain the most variance in the dataset.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of exact column names to include in the PCA."
    )
    
    n_components: Optional[int] = Field(
        default=None, 
        description="The number of principal components to compute. If omitted, computes components for all features."
    )

class run_kmeans_clustering_tool(BaseModel):
    """
    Performs K-Means clustering to group data into distinct segments based on feature similarities. 
    Use this to discover customer segments, group similar behaviors, or identify natural groupings in the data.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of exact column names to use for clustering."
    )
    
    n_clusters: Optional[int] = Field(
        default=3, 
        description="The number of clusters (k) to create. Default is 3."
    )

class run_random_forest_tool(BaseModel):
    """
    Trains a Random Forest machine learning model to predict a target variable based on multiple features. 
    Use this to find non-linear relationships, classify outcomes, or determine the importance/impact of various features.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    target_variable: str = Field(
        ..., 
        description="The exact column name of the target variable to predict."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of exact column names for the predictor variables."
    )
    
    task_type: Literal["regression", "classification"] = Field(
        ..., 
        description="Specify 'regression' if the target variable is numerical/continuous, or 'classification' if the target is categorical/discrete."
    )
    
    n_estimators: Optional[int] = Field(
        default=100, 
        description="The number of trees in the forest (default is 100)."
    )

class compare_monthly_metrics_tool(BaseModel):
    """
    Compares monthly marketing spend metrics against rolled-up acquisition metrics over time.
    """
    marketing_metric: str = Field(
        ..., 
        description="The specific column from the marketing table to sum (e.g., 'Total_Spend')."
    )
    
    acquisition_metric_func: str = Field(
        ..., 
        description="The SQL aggregation function to apply to the acquisition table (e.g., 'COUNT', 'SUM', 'AVG')."
    )
    
    acquisition_column: str = Field(
        ..., 
        description="The specific column from the acquisition table to aggregate. Use '*' if counting total customers."
    )

class calculate_unit_economics_tool(BaseModel):
    """
    marketing cost per acquistion (CPA), Lifetime Value (CLV), and CLV:CPA ratios by safely merging marketing spend and acquisition volumes. Use this whenever the user asks about unit economics, cost per acquisition, or marketing efficiency.
    """
    marketing_where_clause: Optional[str] = Field(
        default=None, 
        description="Optional. A PostgreSQL WHERE clause to filter the marketing spend table (e.g., '\"account\" = ''611010'''). Exclude the 'WHERE' keyword."
    )
    
    acquisition_where_clause: Optional[str] = Field(
        default=None, 
        description="Optional. A PostgreSQL WHERE clause to filter the acquisition table (e.g., '\"Sales_Channel\" = ''Direct'''). Exclude the 'WHERE' keyword."
    )

class ScenarioChange(BaseModel):
    """A single hypothetical change to a feature variable."""
    column_name: str = Field(
        ..., 
        description="The exact name of the feature column to modify (e.g., 'Marketing_Spend')."
    )
    new_value: float = Field(
        ..., 
        description="The new hypothetical numerical value for this column (e.g., 50000.0)."
    )

class run_scenario_planning_tool(BaseModel):
    """
    Performs statistical what-if scenario planning and simulations using OLS regression. 
    Use this tool whenever the user asks:
    - What would happen to a target variable (Z) if a feature (X) changes by a certain percentage or to a specific value.
    - Questions containing phrases like "what if", "assume X is", "increase/decrease by X%", or "hold Y constant".
    This tool automatically computes baseline averages, applies the hypothetical changes, holds specified control variables constant at their historical means, and returns expected predictions with 95% confidence intervals.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    target_variable: str = Field(
        ..., 
        description="The exact column name of the target variable to predict (the Z variable)."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of all relevant predictor columns to include in the model (both the variables being changed AND the variables being held constant)."
    )
    
    # Updated: Replaced dict[str, float] with List[ScenarioChange] to satisfy strict JSON Schema rules
    scenario_changes: List[ScenarioChange] = Field(
        ..., 
        description="A list of specific feature columns and their new hypothetical values."
    )
    
    confidence_level: Optional[float] = Field(
        default=0.95, 
        description="The statistical confidence level for the prediction interval (default is 0.95 for a 95% interval)."
    )



#----------------------------VISUALS SCHEMAS----------------------------

class generate_scatterplot_tool(BaseModel):
    """
    Generates an interactive scatterplot to explore relationships between two numerical variables.
    Supports querying a single table or automatically joining multiple tables.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"' or a list of tables to join."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis numerical variable."
    )
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis numerical variable."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to color-code and segment the scatter points."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause (exclude the 'WHERE' keyword)."
    )
    include_trendline: Optional[bool] = Field(
        default=False,
        description="Set to True to overlay an Ordinary Least Squares (OLS) trendline."
    )

class generate_barchart_tool(BaseModel):
    """
    Generates a bar chart to compare aggregated numerical values across categorical groups or time periods.
    Automatically handles pre-aggregation (SUM, AVG, COUNT) to ensure clean visualizations.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"'."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis (usually categorical or dates)."
    )
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis numerical value to measure."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to group or color-code side-by-side bars."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause to filter data before plotting."
    )
    aggregation: Optional[Literal["SUM", "AVG", "COUNT", "MAX", "MIN", "NONE"]] = Field(
        default="SUM",
        description="The aggregation function applied to the Y-axis variable per X-axis group. Default is SUM."
    )

class generate_histogram_tool(BaseModel):
    """
    Generates a histogram with an executive box-plot marginal to visualize data distributions, spread, and outliers.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name of the numerical variable to analyze."
    )
    n_bins: Optional[int] = Field(
        default=None, 
        description="Optional. The number of frequency bins to divide the data into."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to overlay multiple distribution cohorts."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause."
    )

class generate_linechart_tool(BaseModel):
    """
    Generates a continuous line chart to visualize trends over time or sequences.
    Automatically groups duplicate timestamps and sorts chronologically to prevent erratic line jumps.
    """
    TABLE_NAME: Union[str, List[str]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis time, date, or sequential sequence."
    )
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis numerical variable."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to plot multiple colored trend lines simultaneously."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause."
    )
    aggregation: Optional[Literal["SUM", "AVG", "COUNT", "MAX", "MIN", "NONE"]] = Field(
        default="SUM",
        description="The aggregation applied if multiple records share the same X-axis timestamp. Default is SUM."
    )