import json
import os
from pathlib import Path
import ssl
from typing import Any, Dict

from databricks.sdk import WorkspaceClient
import pandas as pd
import plotly.express as px
import mlflow
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sqlalchemy as sa
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

YEARLY_WACC = 0.1
MONTHLY_WACC = (1 + YEARLY_WACC) ** (1 / 12) - 1

# ─── Configuration ───────────────────────────────────────────────
MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres" 

# Initialize the SDK Client (auto-authenticates using the App's Service Principal)
w = WorkspaceClient()

# ─── Table Relationships ─────────────────────────────────────────
TABLE_RELATIONSHIPS = {
    (
        '"sandbox"."dbs_marketing_spend_sync"', 
        '"sandbox"."acquisition_data_v3"'
    ): (
        ' "sandbox"."dbs_marketing_spend_sync"."year" = "sandbox"."acquisition_data_v3"."Activation_Year" '
        'AND "sandbox"."dbs_marketing_spend_sync"."month" = "sandbox"."acquisition_data_v3"."Activation_Month" '
    )
}

def get_join_clause(table_a: str, table_b: str) -> str:
    """Returns the correct ON clause regardless of the order the tables are passed."""
    return TABLE_RELATIONSHIPS.get((table_a, table_b)) or TABLE_RELATIONSHIPS.get((table_b, table_a))

# ─── Helper Functions ────────────────────────────────────────────
@st.cache_resource
def get_db_engine():
    auth_headers = w.config.authenticate()
    current_user = w.current_user.me().user_name
    auth_token = auth_headers["Authorization"].split(" ")[1]
    db_url = f"postgresql+pg8000://{current_user}:{auth_token}@{PGHOST}:5432/{PGDATABASE}"
    ssl_context = ssl.create_default_context()
    return sa.create_engine(db_url, connect_args={"ssl_context": ssl_context})

def run_sql_query(query: str) -> pd.DataFrame:
    engine = get_db_engine()
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)


def raw_llm_call(messages: list, tools: list = None, require_json: bool = False) -> dict:
    """Handles standard and tool-calling requests using the SDK to auto-manage tokens."""
    
    payload = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1500
    }
    if tools:
        payload["tools"] = tools
    if require_json:
        payload['response_format'] = {'type': 'json_object'}

    # The SDK natively handles headers, authentication, and token refreshes
    response = w.api_client.do(
        method="POST", 
        path=f"/serving-endpoints/{MODEL}/invocations", 
        body=payload
    )
    
    usage = response.get("usage", {})
    if usage:
        # Initialize in session_state if it doesn't exist yet
        for metric in ["total_tokens", "prompt_tokens", "completion_tokens"]:
            if metric not in st.session_state:
                st.session_state[metric] = 0
                
        # Aggregate the tokens for the current session
        st.session_state["total_tokens"] += usage.get("total_tokens", 0)
        st.session_state["prompt_tokens"] += usage.get("prompt_tokens", 0)
        st.session_state["completion_tokens"] += usage.get("completion_tokens", 0)

    return response["choices"][0]["message"]


# ─── Tool Definition ─────────────────────────────────────────────
def execute_sql_query_tool(sql_query: str) -> dict:
    """Executes SQL directly. Orchestrator now handles the retry logic."""
    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: Query executed successfully, but returned 0 rows.", "data": None}
        
        # Limit rows converted to text to prevent blowing up the LLM context window
        csv_text = df.head(100).to_csv(index=False)
        return {"text": f"Success. Showing top 100 rows:\n{csv_text}", "data": df}
        
    except Exception as e:
        return {"text": f"Error executing SQL: {str(e)}", "data": None}
    
def run_ols_regression_tool(TABLE_NAME, dependent_variable: str, independent_variables: list) -> dict:
    """
    Sub-agent tool: Fetches specific numerical columns and runs an OLS multiple regression.
    """
    columns_to_fetch = [dependent_variable] + independent_variables
    # NEW: Strip any existing quotes the LLM might have added, then forcefully wrap in double quotes
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in columns_to_fetch]
    columns_str = ", ".join(safe_columns)
    
    # We query more rows here than the SQL tool to ensure a valid sample size for regression
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    
    try:
        # Fetch the data using your existing helper
        df = run_sql_query(sql_query)
        
        # Drop rows with missing values to prevent the regression from crashing
        df = df.dropna(subset=columns_to_fetch)
        
        # Ensure we have enough data points to run a valid regression
        if df.empty or len(df) <= len(independent_variables):
            return "Error: Not enough valid data points to perform regression."
            
        # 2. Define target (Y) and features (X)
        Y = pd.to_numeric(df[dependent_variable])
        
        # Convert all independent variables to numeric, coercing errors to NaN, then drop again if needed
        X = df[independent_variables].apply(pd.to_numeric, errors='coerce')
        
        # Add a constant (intercept) to the model, which is required for standard OLS
        X = sm.add_constant(X)
        
        # 3. Fit the OLS model
        model = sm.OLS(Y, X).fit()
        
        # 4. Return the statistical summary as a string for the LLM to interpret
        return {"text": model.summary().as_text(), "data": model}
        
    except Exception as e:
        return f"Regression Error: {e}"
    
def run_arima_forecasting_tool(TABLE_NAME, value_column: str, aggregation: str = "SUM", steps: int = 5, p: int = 1, d: int = 1, q: int = 1) -> dict:
    """
    Sub-agent tool: Fetches historical data grouped by Activation_Year and Activation_Month, 
    and forecasts future periods using an ARIMA model.
    """
    safe_value = '"{}"'.format(value_column.replace('"', ''))
    
    # Protect against SQL injection and ensure a valid aggregation function
    agg_func = aggregation.upper() if aggregation.upper() in ["SUM", "AVG", "COUNT"] else "SUM"
    
    # Group by Year and Month, ensuring chronological order for the time series
    sql_query = f"""
        SELECT 
            "Activation_Year", 
            "Activation_Month", 
            {agg_func}({safe_value}) AS target_value 
        FROM {TABLE_NAME} 
        WHERE "Activation_Year" IS NOT NULL AND "Activation_Month" IS NOT NULL
        GROUP BY "Activation_Year", "Activation_Month" 
        ORDER BY "Activation_Year" ASC, "Activation_Month" ASC
    """
    
    try:
        df = run_sql_query(sql_query)
        
        # Clean the target values
        df['target_value'] = pd.to_numeric(df['target_value'], errors='coerce')
        df = df.dropna(subset=['target_value'])
        
        if df.empty or len(df) < 10:
            return "Error: Not enough historical monthly data points (minimum 10 required) to perform ARIMA forecasting."
            
        # Parse series data (already ordered chronologically by SQL)
        series = df['target_value'].values
        
        # Fit ARIMA model
        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
        
        # Generate future forecasts
        forecast = model_fit.forecast(steps=steps)
        
        # Build text summary for the LLM
        result_text = f"ARIMA({p},{d},{q}) Forecasting Results for {agg_func} of {value_column}:\n"
        result_text += f"Based on {len(series)} months of historical data, here are the predictions for the next {steps} months:\n"
        
        for i, val in enumerate(forecast, start=1):
            result_text += f"  • Month +{i}: {val:.4f}\n"
            
        return {"text": result_text, "data": model_fit}
        
    except Exception as e:
        return f"ARIMA Forecasting Error: {e}"
    

def run_random_forest_tool(TABLE_NAME, target_variable: str, feature_variables: list, task_type: str = "regression", n_estimators: int = 100) -> Dict[str, Any]:
    """
    Sub-agent tool: Fetches columns, preprocesses data, and runs a Random Forest model.
    Returns a dictionary containing the LLM-readable text result and the trained model object.
    """
    columns_to_fetch = [target_variable] + feature_variables
    
    # TODO: Add explicit validation of `columns_to_fetch` against your database schema here
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in columns_to_fetch]
    columns_str = ", ".join(safe_columns)
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 100000"
    
    try:
        df = run_sql_query(sql_query)
        
        if df.empty or len(df) <= len(feature_variables):
            return {"text": "Error: Not enough data points.", "model": None}
            
        # Coerce target to numeric if regression
        if task_type.lower() == "regression":
            df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')
            
        # Handle categorical features by creating dummy variables (One-Hot Encoding)
        # We don't blindly coerce features to numeric; we let pd.get_dummies handle strings
        df = pd.get_dummies(df, columns=[col for col in feature_variables if df[col].dtype == 'object'], drop_first=True)
        
        # Update feature variables list after dummy creation
        current_features = [col for col in df.columns if col != target_variable]
        
        # Single, clean dropna step
        df = df.dropna(subset=[target_variable] + current_features)
        
        if len(df) < 10:
            return {"text": "Error: Data size too small after cleaning to train a valid model.", "model": None}
            
        X = df[current_features]
        y = df[target_variable]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model with sensible regularization to prevent severe overfitting
        if task_type.lower() == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=7, min_samples_leaf=3, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            result_text = f"Random Forest Regression Results (n_estimators={n_estimators}):\n"
            result_text += f"Model Test R-squared: {r2_score(y_test, preds):.4f}\n"
            result_text += f"Model Test MSE: {mean_squared_error(y_test, preds):.4f}\n\n"
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=7, min_samples_leaf=3, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            result_text = f"Random Forest Classification Results (n_estimators={n_estimators}):\n"
            result_text += f"Model Test Accuracy: {accuracy_score(y_test, preds):.4f}\n"
            result_text += f"Classification Report:\n{classification_report(y_test, preds)}\n\n"
            
        # Feature importances
        importances = model.feature_importances_
        feat_imp = sorted(zip(current_features, importances), key=lambda x: x[1], reverse=True)
        
        result_text += "Feature Importances (higher is more impactful):\n"
        # Only show top 10 if there are many dummy variables to save LLM context window
        for feat, imp in feat_imp[:10]:
            result_text += f"  • {feat}: {imp:.4f}\n"
            
        return {"text": result_text, "model": model}
        
    except Exception as e:
        # Consider logging the full traceback here for debugging
        return {"text": f"Random Forest Error: {e}", "model": None}
    
def run_pca_tool(TABLE_NAME, feature_variables: list, n_components: int = None) -> Dict[str, Any]:
    """
    Sub-agent tool: Fetches columns, standardizes data, and runs Principal Component Analysis (PCA).
    Returns a dictionary containing the LLM-readable text result and the trained PCA object.
    """
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in feature_variables]
    columns_str = ", ".join(safe_columns)
    
    # Using the same row limit strategy as the Random Forest tool
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 100000"
    
    try:
        df = run_sql_query(sql_query)
        
        if df.empty or len(df) < 2:
            return {"text": "Error: Not enough data points fetched to perform PCA.", "model": None}
            
        # 1. Handle categorical features by creating dummy variables
        df = pd.get_dummies(df, columns=[col for col in feature_variables if df[col].dtype == 'object'], drop_first=True)
        
        # 2. Drop any rows with missing values
        df = df.dropna()
        
        current_features = df.columns.tolist()
        
        if len(df) < 2 or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning and encoding to perform PCA.", "model": None}
        
        # 3. Standardize the columns (Crucial for PCA so large magnitude features don't dominate)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Cap n_components to the maximum mathematically possible if the LLM requests too many
        max_components = min(len(df), len(current_features))
        if n_components is None or n_components > max_components:
            actual_components = max_components
        else:
            actual_components = n_components
            
        # 4. Fit PCA
        pca = PCA(n_components=actual_components)
        pca.fit(scaled_data)
        
        # 5. Build text summary for the LLM
        result_text = f"PCA Results (n_components={actual_components}):\n"
        explained_variance = pca.explained_variance_ratio_
        
        result_text += "Explained Variance Ratio per Component:\n"
        for i, var in enumerate(explained_variance):
            result_text += f"  • PC{i+1}: {var:.4f} ({(var*100):.1f}%)\n"
        result_text += f"Total Explained Variance: {sum(explained_variance):.4f} ({(sum(explained_variance)*100):.1f}%)\n\n"
        
        # Extract feature loadings to explain *what* makes up each component
        # We limit the output to the top 2 components and only significant loadings to save LLM context
        components_to_show = min(2, actual_components)
        result_text += "Top Feature Loadings (absolute magnitude > 0.3) for primary components:\n"
        
        for i in range(components_to_show):
            result_text += f"  PC{i+1} Signficant Loadings:\n"
            loadings = pca.components_[i]
            
            # Match loadings to feature names and sort by absolute impact
            feat_loadings = sorted(zip(current_features, loadings), key=lambda x: abs(x[1]), reverse=True)
            for feat, load in feat_loadings:
                if abs(load) > 0.3:
                    result_text += f"    - {feat}: {load:.4f}\n"
        
        return {"text": result_text, "model": pca}
        
    except Exception as e:
        return {"text": f"PCA Error: {e}", "model": None}
    
def run_kmeans_clustering_tool(TABLE_NAME, feature_variables: list, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Sub-agent tool: Fetches columns, standardizes data, and runs K-Means clustering.
    Returns a dictionary containing the LLM-readable text result and the trained KMeans object.
    """
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in feature_variables]
    columns_str = ", ".join(safe_columns)
    
    # Using the same row limit strategy as the Random Forest and PCA tools
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 100000"
    
    try:
        df = run_sql_query(sql_query)
        
        if df.empty or len(df) < n_clusters:
            return {"text": f"Error: Not enough data points fetched to perform {n_clusters}-means clustering.", "model": None}
            
        # 1. Handle categorical features by creating dummy variables
        df = pd.get_dummies(df, columns=[col for col in feature_variables if df[col].dtype == 'object'], drop_first=True)
        
        # 2. Drop any rows with missing values
        df = df.dropna()
        current_features = df.columns.tolist()
        
        if len(df) < n_clusters or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning and encoding to perform clustering.", "model": None}
        
        # 3. Standardize the columns (Crucial for distance-based clustering algorithms)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # 4. Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(scaled_data)
        
        # 5. Build text summary for the LLM
        df['Cluster'] = kmeans.labels_
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        result_text = f"K-Means Clustering Results (n_clusters={n_clusters}):\n"
        result_text += "Cluster Population Sizes:\n"
        for cluster_id, count in cluster_counts.items():
            result_text += f"  • Cluster {cluster_id}: {count} data points\n"
            
        result_text += "\nCluster Profiles (Standardized Centroids):\n"
        result_text += "Note: Values > 0 mean the cluster is above the dataset average for that feature, < 0 means below average.\n"
        
        centroids = kmeans.cluster_centers_
        
        # Limit to the most defining features per cluster to save LLM context
        for i in range(n_clusters):
            result_text += f"  Cluster {i} Defining Features (Top 5):\n"
            
            # Match standardized centroid values to feature names and sort by absolute magnitude
            feat_centroids = sorted(zip(current_features, centroids[i]), key=lambda x: abs(x[1]), reverse=True)
            
            for feat, val in feat_centroids[:5]:
                # Only show features that have a somewhat distinct deviation from the mean
                if abs(val) > 0.15: 
                    result_text += f"    - {feat}: {val:.4f}\n"
                    
        return {"text": result_text, "model": kmeans}
        
    except Exception as e:
        return {"text": f"K-Means Error: {e}", "model": None}
    
def generate_scatterplot_tool(x_config: dict, y_config: dict, category_column: str = None, where_clause: str = None, include_trendline: bool = False) -> dict:
    """Sub-agent tool: Generates an interactive Plotly scatterplot. Supports single-table or cross-table queries."""
    
    table_x = x_config["table_name"]
    col_x = x_config["column_name"].replace('"', '').split('.')[-1]
    
    table_y = y_config["table_name"]
    col_y = y_config["column_name"].replace('"', '').split('.')[-1]

    # Handle optional category column
    cat_select = ""
    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        # Defaulting category to table_x for simplicity in this snippet
        cat_select = f', {table_x}."{cat_clean}" AS category_col'

    # Single-Table Scenario
    if table_x == table_y:
        sql_query = f'SELECT "{col_x}" AS x, "{col_y}" AS y {cat_select} FROM {table_x}'
        if where_clause:
            sql_query += f" WHERE {where_clause}"
            
    # Cross-Table Scenario
    else:
        join_condition = get_join_clause(table_x, table_y)
        if not join_condition:
            return {"text": f"Error: No known relationship to join {table_x} and {table_y}.", "data": None}
            
        sql_query = f"""
            SELECT 
                {table_x}."{col_x}" AS x, 
                {table_y}."{col_y}" AS y 
                {cat_select}
            FROM {table_x}
            JOIN {table_y} ON {join_condition}
        """
        if where_clause:
            sql_query += f" WHERE {where_clause}"

    sql_query += " ORDER BY RANDOM() LIMIT 5000"

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned. Check filters or join conditions.", "data": None}

        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['x', 'y'])

        if df.empty:
            return {"text": "Error: Not enough valid numeric data to plot.", "data": None}

        t_line = "ols" if include_trendline else None
        
        if category_column:
            df['category_col'] = df['category_col'].astype(str)
            fig = px.scatter(df, x='x', y='y', color='category_col', trendline=t_line, title=f"{col_y} vs {col_x}")
        else:
            fig = px.scatter(df, x='x', y='y', trendline=t_line, title=f"{col_y} vs {col_x}")

        # Update axis labels to be descriptive
        fig.update_layout(xaxis_title=col_x, yaxis_title=col_y)

        return {
            "text": f"Successfully generated scatterplot for {col_y} vs {col_x}.", 
            "data": df, 
            "figure": fig
        }
    except Exception as e:
        return {"text": f"Scatterplot Error: {e}", "data": None}
    
def generate_barchart_tool(TABLE_NAME, x_column: str, y_column: str, category_column: str = None, where_clause: str = None) -> dict:
    """Sub-agent tool: Generates an interactive Plotly bar chart."""
    x_clean = x_column.replace('"', '').split('.')[-1]
    y_clean = y_column.replace('"', '').split('.')[-1]
    columns_to_fetch = [x_clean, y_clean]

    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        columns_to_fetch.append(cat_clean)

    safe_columns = ['"{}"'.format(col) for col in list(set(columns_to_fetch))]
    columns_str = ", ".join(safe_columns)
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    if where_clause:
        sql_query += f" WHERE {where_clause}"
    sql_query += " LIMIT 5000"

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for the bar chart.", "data": None}

        df[y_clean] = pd.to_numeric(df[y_clean], errors='coerce')
        df = df.dropna(subset=[x_clean, y_clean])

        if category_column:
            df[cat_clean] = df[cat_clean].astype(str)
            fig = px.bar(df, x=x_clean, y=y_clean, color=cat_clean, title=f"{y_clean} by {x_clean}")
        else:
            fig = px.bar(df, x=x_clean, y=y_clean, title=f"{y_clean} by {x_clean}")

        return {"text": f"Successfully generated bar chart for {y_clean} by {x_clean}.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Bar Chart Error: {e}", "data": None}


def generate_histogram_tool(TABLE_NAME, x_column: str, n_bins: int = None, category_column: str = None, where_clause: str = None) -> dict:
    """Sub-agent tool: Generates an interactive Plotly histogram to show data distributions."""
    x_clean = x_column.replace('"', '').split('.')[-1]
    columns_to_fetch = [x_clean]

    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        columns_to_fetch.append(cat_clean)

    safe_columns = ['"{}"'.format(col) for col in list(set(columns_to_fetch))]
    columns_str = ", ".join(safe_columns)
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    if where_clause:
        sql_query += f" WHERE {where_clause}"
    sql_query += " LIMIT 10000" # Slightly larger limit for distributions

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for the histogram.", "data": None}

        df[x_clean] = pd.to_numeric(df[x_clean], errors='coerce')
        df = df.dropna(subset=[x_clean])

        kwargs = {"x": x_clean, "title": f"Distribution of {x_clean}"}
        if n_bins:
            kwargs["nbins"] = n_bins
        if category_column:
            df[cat_clean] = df[cat_clean].astype(str)
            kwargs["color"] = cat_clean
            kwargs["barmode"] = "overlay" # Better default for overlapping distributions

        fig = px.histogram(df, **kwargs)

        return {"text": f"Successfully generated histogram for {x_clean}.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Histogram Error: {e}", "data": None}


def generate_linechart_tool(TABLE_NAME, x_column: str, y_column: str, category_column: str = None, where_clause: str = None) -> dict:
    """Sub-agent tool: Generates an interactive Plotly line chart, sorting by X-axis."""
    x_clean = x_column.replace('"', '').split('.')[-1]
    y_clean = y_column.replace('"', '').split('.')[-1]
    columns_to_fetch = [x_clean, y_clean]

    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        columns_to_fetch.append(cat_clean)

    safe_columns = ['"{}"'.format(col) for col in list(set(columns_to_fetch))]
    columns_str = ", ".join(safe_columns)
    
    # We order by the X column at the SQL level to ensure the line draws correctly chronologically/sequentially
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    if where_clause:
        sql_query += f" WHERE {where_clause}"
    sql_query += f" ORDER BY \"{x_clean}\" ASC LIMIT 5000"

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for the line chart.", "data": None}

        df[y_clean] = pd.to_numeric(df[y_clean], errors='coerce')
        df = df.dropna(subset=[x_clean, y_clean])

        if category_column:
            df[cat_clean] = df[cat_clean].astype(str)
            fig = px.line(df, x=x_clean, y=y_clean, color=cat_clean, title=f"Trend of {y_clean} over {x_clean}")
        else:
            fig = px.line(df, x=x_clean, y=y_clean, title=f"Trend of {y_clean} over {x_clean}")

        return {"text": f"Successfully generated line chart for {y_clean} over {x_clean}.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Line Chart Error: {e}", "data": None}
    
def compare_monthly_metrics_tool(marketing_metric: str, acquisition_metric_func: str = "COUNT", acquisition_column: str = "*") -> dict:
    """
    Aggregates acquisition data to a monthly grain and joins it with monthly marketing metrics
    to generate an interactive trend line chart.
    """
    marketing_clean = marketing_metric.replace('"', '').split('.')[-1]
    acq_clean = acquisition_column.replace('"', '').split('.')[-1] if acquisition_column != "*" else "*"
    func_clean = acquisition_metric_func.upper()
    
    # Validate aggregation function to prevent injection
    if func_clean not in ["COUNT", "SUM", "AVG"]:
        func_clean = "COUNT"

    sql_query = f"""
        SELECT 
            m."year" AS year,
            m."month" AS month,
            SUM(m."{marketing_clean}") AS marketing_total,
            {func_clean}(a."{acq_clean}") AS acquisition_total
        FROM "sandbox"."dbs_marketing_spend_sync" m
        LEFT JOIN "sandbox"."acquisition_data_v3" a 
            ON m."year" = a."Activation_Year" AND m."month" = a."Activation_Month"
        GROUP BY m."year", m."month"
        ORDER BY m."year" ASC, m."month" ASC
    """

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for the time-series comparison.", "data": None}

        # Create a unified date string for the X-axis
        df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01', errors='coerce')
        df = df.dropna(subset=['Date', 'marketing_total', 'acquisition_total'])

        # Melt the dataframe so Plotly can easily plot two lines with a legend
        df_melted = df.melt(id_vars=['Date'], value_vars=['marketing_total', 'acquisition_total'], 
                            var_name='Metric', value_name='Value')

        fig = px.line(df_melted, x='Date', y='Value', color='Metric', 
                      title=f"Monthly Trend: {marketing_clean} vs {func_clean} of {acq_clean}")

        return {
            "text": f"Successfully generated monthly comparison chart for {marketing_clean} and {func_clean} of {acq_clean}.", 
            "data": df, 
            "figure": fig
        }
    except Exception as e:
        return {"text": f"Monthly Comparison Error: {e}", "data": None}

def calculate_unit_economics_tool(marketing_where_clause: str = None, acquisition_where_clause: str = None) -> dict:
    """
    Sub-agent tool: Calculates Marketing cost per acquisition (CPA) and est. CLV:cpa ratios 
    by aggregating marketing spend and acquisition volumes at a monthly grain, 
    safely merged via Pandas.
    """
    # 1. Fetch Marketing Spend
    mkt_query = """
        SELECT 
            "year" AS year,
            "month" AS month,
            SUM("amount") AS total_spend
        FROM "sandbox"."dbs_marketing_spend_sync"
    """
    if marketing_where_clause:
        mkt_query += f" WHERE {marketing_where_clause}"
    mkt_query += ' GROUP BY "year", "month"'

    # 2. Fetch Acquisition Data (Activations & Net Present Value)
    acq_query = """
        SELECT 
            "Activation_Year" AS year,
            "Activation_Month" AS month,
            COUNT(*) AS total_activations,
            AVG("mcf") AS avg_mcf,
            AVG("Ve_Churn") AS avg_churn
        FROM "sandbox"."acquisition_data_v3"
    """
    if acquisition_where_clause:
        acq_query += f" WHERE {acquisition_where_clause}"
    acq_query += ' GROUP BY "Activation_Year", "Activation_Month"'

    try:
        df_mkt = run_sql_query(mkt_query)
        df_acq = run_sql_query(acq_query)

        if df_mkt.empty or df_acq.empty:
            return {"text": "Error: One or both tables returned no data for the specified filters.", "data": None}

        # 3. Clean and Merge safely in Pandas
        df_mkt['year'] = pd.to_numeric(df_mkt['year'], errors='coerce')
        df_mkt['month'] = pd.to_numeric(df_mkt['month'], errors='coerce')
        df_acq['year'] = pd.to_numeric(df_acq['year'], errors='coerce')
        df_acq['month'] = pd.to_numeric(df_acq['month'], errors='coerce')

        df_merged = pd.merge(df_mkt, df_acq, on=['year', 'month'], how='inner')

        if df_merged.empty:
            return {"text": "Error: Could not calculate CAC. No overlapping months found between the two datasets.", "data": None}

        # 4. Calculate Unit Economics
        df_merged['cpa'] = df_merged['total_spend'] / df_merged['total_activations']
        df_merged['clv'] = df_merged['avg_mcf'] /(MONTHLY_WACC + df_merged['avg_churn'])
        df_merged['clv_cpa_ratio'] = df_merged['clv'] / df_merged['cpa']

        # Clean up infinities if there were months with zero activations
        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Sort chronologically for the time series
        df_merged = df_merged.sort_values(by=['year', 'month'])

        # 5. Build Interactive Trend Chart
        df_merged['Date'] = pd.to_datetime(
            df_merged['year'].astype(int).astype(str) + '-' + 
            df_merged['month'].astype(int).astype(str) + '-01', 
            errors='coerce'
        )
        
        fig = px.line(
            df_merged, 
            x='Date', 
            y='cpa', 
            title='Blended Cost per Acquisition (CPA) Trend',
            markers=True,
            labels={'cpa': 'CPA ($)', 'Date': 'Activation Month'}
        )

        # 6. Generate Business Summary for the LLM
        overall_spend = df_merged['total_spend'].sum()
        overall_acq = df_merged['total_activations'].sum()
        blended_cpa = overall_spend / overall_acq if overall_acq > 0 else 0
        avg_clv = df_merged['clv'].mean()
        clv_cpa = avg_clv / blended_cpa if blended_cpa > 0 else 0
        
        text_output = (
            f"Unit Economics Summary:\n"
            f"  • Total Marketing Spend Analyzed: ${overall_spend:,.2f}\n"
            f"  • Total Activations: {overall_acq:,.0f}\n"
            f"  • Blended CPA: ${blended_cpa:,.2f}\n"
            f"  • Average CLV (NPV): ${avg_clv:,.2f}\n"
            f"  • Blended CLV:CPA Ratio: {clv_cpa:.2f}x\n\n"
            f"Note: Data is grouped by month. See the attached dataframe and chart for trend lines."
        )

        return {"text": text_output, "data": df_merged, "figure": fig}

    except Exception as e:
        return {"text": f"Unit Economics Calculation Error: {e}", "data": None}

# ─── Load Config & Map Dispatcher ────────────────────────────────
TOOLS_FILE_PATH = Path(__file__).parent.resolve() / "tool_config.json"
try:
    with TOOLS_FILE_PATH.open("r", encoding="utf-8") as f:
        TOOLS = json.load(f)
except Exception as e:
    st.error(f"Configuration Error loading {TOOLS_FILE_PATH.name}: {e}")
    TOOLS = []

TOOL_DISPATCHER = {
    "execute_sql_query_tool": execute_sql_query_tool,
    "run_ols_regression_tool": run_ols_regression_tool,
    "run_arima_forecasting_tool": run_arima_forecasting_tool,
    "run_random_forest_tool": run_random_forest_tool,
    "run_pca_tool": run_pca_tool,
    "run_kmeans_clustering_tool": run_kmeans_clustering_tool,
    "generate_scatterplot_tool": generate_scatterplot_tool,
    "generate_barchart_tool": generate_barchart_tool,
    "generate_histogram_tool": generate_histogram_tool,
    "generate_linechart_tool": generate_linechart_tool,
    "calculate_unit_economics_tool": calculate_unit_economics_tool
}