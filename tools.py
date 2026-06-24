import os
import json
import pandas as pd
import ssl
from pathlib import Path
import streamlit as st
import sqlalchemy as sa
from databricks.sdk import WorkspaceClient
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from typing import Dict, Any

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
    
def run_arima_forecasting_tool(time_column: str, value_column: str, steps: int = 5, p: int = 1, d: int = 1, q: int = 1) -> dict:
    """
    Sub-agent tool: Fetches chronological data and forecasts future periods using an ARIMA model.
    """
    # NEW: Clean and wrap the injected column names in double quotes
    safe_time = '"{}"'.format(time_column.replace('"', ''))
    safe_value = '"{}"'.format(value_column.replace('"', ''))
    
    # Order by the time column to ensure data is chronological for time series
    sql_query = f"""
        SELECT 
            DATE_TRUNC('month', {safe_time}) AS {safe_time}, 
            SUM({safe_value}) AS {safe_value} 
        FROM {TABLE_NAME} 
        GROUP BY DATE_TRUNC('month', {safe_time}) 
        ORDER BY {safe_time} ASC
    """
    
    try:
        df = run_sql_query(sql_query)
        
        # Drop missing values and ensure numeric casting
        df = df.dropna(subset=[time_column, value_column])
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        df = df.dropna(subset=[value_column])
        
        if df.empty or len(df) < 10:
            return "Error: Not enough historical data points (minimum 10 required) to perform ARIMA forecasting."
            
        # Parse series data
        series = df[value_column].values
        
        # Fit ARIMA model using the p, d, q parameters passed by the LLM (or defaults)
        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
        
        # Generate future forecasts
        forecast = model_fit.forecast(steps=steps)
        
        # Build a text summary for the LLM to read and translate into natural language
        result_text = f"ARIMA({p},{d},{q}) Forecasting Results:\n"
        result_text += f"Based on {len(series)} historical rows, here are the predictions for the next {steps} periods:\n"
        
        for i, val in enumerate(forecast, start=1):
            result_text += f"  • Period +{i}: {val:.4f}\n"
            
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
    "run_random_forest_tool": run_random_forest_tool
}