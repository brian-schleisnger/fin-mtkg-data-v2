from typing import List, Optional, Literal

from pydantic import BaseModel, Field

# ─── Orchestration Schemas ───
class DecomposedQuestions(BaseModel):
    """The broken-down data queries based on the user's prompt."""
    questions: List[str] = Field(
        description="A list of specific, actionable data queries. Max 5."
    )

# -------------------- ANALYSIS SCHEMAS --------------------

class execute_sql_query_tool(BaseModel):
    """
    Queries the Databricks database using PostgreSQL syntax. Because this is PostgreSQL, you MUST wrap all column names in double quotes to preserve exact capitalization. 
    Do NOT use this tool if a more specific tool is available for the user's request (e.g., use calculate_unit_economics_tool for CPA/CLV, use regression/forecasting tools for modeling).
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
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
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
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
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
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
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
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
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
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
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
    Performs what-if scenario planning by predicting a target variable (Z) when specific features (X) 
    are changed to hypothetical values, while holding all other control variables (Y) constant at their historical averages.
    Use this when the user asks questions like 'What would happen to revenue if marketing spend was $50k?' or 'If X is [value] while holding Y constant, what is Z?'.
    """
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
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



#----------------------------VISUALS SCHEMAAS----------------------------

class AxisConfig(BaseModel):
    table_name: str = Field(
        ..., 
        description="The exact SQL table name."
    )
    
    column_name: str = Field(
        ..., 
        description="The raw column name."
    )

class generate_scatterplot_tool(BaseModel):
    """
    Generates a scatterplot. Can pull variables from the same table or across two different tables.
    """
    x_config: AxisConfig = Field(
        ..., 
        description="The table and column configuration for the X-axis metric."
    )
    
    y_config: AxisConfig = Field(
        ..., 
        description="The table and column configuration for the Y-axis metric."
    )
    
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column to color-code the scatter points."
    )
    
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional SQL WHERE clause (exclude the 'WHERE' keyword)."
    )
    
    include_trendline: Optional[bool] = Field(
        default=None
    )

class generate_barchart_tool(BaseModel):
    """
    Generates a bar chart to compare numerical values across categorical variables. 
    Use this when the user asks for a bar chart or wants to compare totals/averages across different groups.
    """
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis (usually categorical)."
    )
    
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis (numerical value to measure)."
    )
    
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional. The exact column name to use for color-coding or grouping the bars."
    )
    
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional. A standard PostgreSQL WHERE clause to filter the data."
    )

class generate_histogram_tool(BaseModel):
    """
    Generates a histogram to visualize the distribution of a single numerical variable. 
    Use this when the user asks about the distribution, spread, or frequency of a specific metric.
    """
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    x_column: str = Field(
        ..., 
        description="The exact column name of the numerical variable to analyze."
    )
    
    n_bins: Optional[int] = Field(
        default=None, 
        description="Optional. The number of bins to divide the data into."
    )
    
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional. The exact column name to use for color-coding the distribution by group."
    )
    
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional. A standard PostgreSQL WHERE clause to filter the data."
    )

class generate_linechart_tool(BaseModel):
    """
    Generates a line chart to visualize trends over time or across a continuous sequence. 
    Use this when the user asks for a line chart or wants to see how a metric changes over time.
    """
    TABLE_NAME: str = Field(
        ..., 
        description="The exact SQL-safe table name to query, e.g., '\"sandbox\".\"acquisition_data_no_id\"' or '\"sandbox\".\"dbs_marketing_spend_sync\"'."
    )
    
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis (usually a time, date, or continuous sequence variable)."
    )
    
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis (numerical variable)."
    )
    
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional. The exact column name to use for color-coding multiple lines."
    )
    
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional. A standard PostgreSQL WHERE clause to filter the data."
    )