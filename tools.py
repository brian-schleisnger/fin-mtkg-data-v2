import json
import os
import ssl
from pathlib import Path
from typing import Any, Dict

from databricks.sdk import WorkspaceClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sqlalchemy as sa
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st


# ─── Configuration ───────────────────────────────────────────────
MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres" 
TABLE_NAME = '"sandbox"."acquisition_data_no_id"'

# Initialize the SDK Client (auto-authenticates using the App's Service Principal)
w = WorkspaceClient()

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
        # Returning the error string allows the outer orchestrator to see it and retry
        return {"text": f"Error executing SQL: {str(e)}", "data": None}
    
def run_ols_regression_tool(dependent_variable: str, independent_variables: list) -> dict:
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
    
def run_arima_forecasting_tool(value_column: str, aggregation: str = "SUM", steps: int = 5, p: int = 1, d: int = 1, q: int = 1) -> dict:
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
    

def run_random_forest_tool(target_variable: str, feature_variables: list, task_type: str = "regression", n_estimators: int = 100) -> Dict[str, Any]:
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
    
def run_pca_tool(feature_variables: list, n_components: int = None) -> Dict[str, Any]:
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
    "run_pca_tool": run_pca_tool
}