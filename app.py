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


st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

# ─── Configuration ───────────────────────────────────────────────
MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres" 
TABLE_NAME = '"sandbox"."acquisition_data_no_id"'

# Initialize the SDK Client (auto-authenticates using the App's Service Principal)
w = WorkspaceClient()


# ─── Schema Context ──────────────────────────────────────────────
DATA_DICTIONARY = f"""
Table Name: {TABLE_NAME}
This table contains marketing data. Key columns include:
- Acnt_Id (string - unique customer identifier)
- Activation_Date (timestamp - date the ccustomer activated service)
- Beacon_Score_10pt (string - credit score range)
- Core_Package (string - package for the user (e.g., americas top 120))
- Tactic (string - marketing tactic the customer came from)
- WA Churn (float - estimated months on by customer)
- sac (float - subscriber aquisition cost)
- NC_ARPU (float - new customer average total revenue)
- NC_COGS (float - new customer average cogs)
- mcf (float - estimated monthly cash flow)
- npv (float - customer net present value)
there are more columns in the table as well.
"""

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

# ─── RAG & Orchestration Helpers ─────────────────────────────────
def filter_schema(user_prompt: str) -> dict:
    """Filters the dictionary so the LLM isn't overwhelmed by irrelevant tables."""
    # In a larger app, use keyword matching here. For now, we return the whole dict.
    return DATA_DICTIONARY

def decompose_question(user_prompt: str, schema: dict) -> list:
    """Step 1: Breaks the user's prompt into specific data questions."""
    prompt = f"""You are a data strategist. Break the user's broad request down into specific, actionable data queries.
    Available Data Schema: {json.dumps(schema)}
    User Request: {user_prompt}

    RULES:
    1. ONLY generate data queries if the user is explicitly asking for data analysis, metrics, or insights.
    2. Do not generate more than five queries. 
    3. CRITICAL: Do NOT break down statistical models (like Regression, Random Forest, or ARIMA) into separate questions for their sub-metrics (e.g., coefficients, R-squared, p-values, residuals). Group all requirements for a single model into ONE unified question
    4. If the user is asking a general question, greeting you, or asking about your capabilities, return the user's exact prompt as a single item and do NOT generate data queries.
    
    Respond STRICTLY with a JSON object containing a 'questions' key mapped to a list of strings.
    Example: {{"questions": ["What is the sum of NC_COGS in 2025?", "What is the average NPV?"]}}"""
    
    msgs = [{"role": "user", "content": prompt}]
    response = raw_llm_call(msgs, require_json=True)
    
    try:
        parsed = json.loads(response.get("content", "{}"))
        return parsed.get("questions", [user_prompt]) # Fallback to original prompt if parsing fails
    except json.JSONDecodeError:
        return [user_prompt]

# ─── Tool Definition ─────────────────────────────────────────────
def execute_sql_query_tool(user_intent: str, schema: dict) -> dict:
    """Writes SQL, executes it, and features an internal auto-correction loop."""
    max_retries = 2
    error_msg = ""
    logs = []
    
    for attempt in range(max_retries):
        sql_system_prompt = f"""You are an expert Databricks SQL analyst. 
        Write a SQL query for this question: {user_intent}
        Schema: {json.dumps(schema)}
        Table to use: {TABLE_NAME}
        {f'PREVIOUS ERROR TO FIX: {error_msg}' if error_msg else ''}
        
        RULES: 
        1. Return ONLY raw SQL. No markdown formatting. 
        2. If returning raw, unaggregated row data, you MUST append LIMIT 100. If returning aggregations, statistics, counts, or grouped summaries (e.g., SUM, AVG, CORR, quantiles), DO NOT use a limit.
        3. CRITICAL: Because this is a PostgreSQL database, you MUST wrap all column names in double quotes to preserve exact capitalization (e.g., SELECT "Core_Package", AVG("sac") FROM...)."""
    
        # 1. Generate SQL
        msgs = [{"role": "user", "content": sql_system_prompt}]
    
        response_msg = raw_llm_call(msgs)
        sql_query = response_msg.get("content", "").replace("```sql", "").replace("```", "").strip()
    
        logs.append(f"Attempting SQL: {sql_query}")

        # 2. Execute SQL
        try:
            df = run_sql_query(sql_query)
            if df.empty:
                return "Query executed successfully, but returned 0 rows."
            # Return CSV string to the agent (scalability handled by the LIMIT 100 in prompt)
            return {"text": df.to_csv(index=False), "data": df, "logs": logs}
        except Exception as e:
            error_msg = str(e)
            logs.append(f"SQL Error caught: {error_msg}. Retrying...")
    return {"text": f"Failed after {max_retries} attempts. Last error: {error_msg}", "data": None, "logs": logs}
    
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
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    
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
    

# ─── Load Tool Schemas ───────────────────────────────────────────
# Get the absolute path to the tools_config.json file next to app.py
TOOLS_FILE_PATH = Path(__file__).parent.resolve() / "tool_config.json"

# Load the JSON into the TOOLS variable
try:
    # Path objects have their own .open() method!
    with TOOLS_FILE_PATH.open("r", encoding="utf-8") as f:
        TOOLS = json.load(f)
except FileNotFoundError:
    st.error(f"Configuration Error: Could not find '{TOOLS_FILE_PATH.name}'. Please ensure the file exists.")
    TOOLS = []
except json.JSONDecodeError as e:
    st.error(f"Configuration Error: Invalid JSON in {TOOLS_FILE_PATH.name}. Error: {e}")
    TOOLS = []

# Map string names from the LLM to actual Python functions
TOOL_DISPATCHER = {
    "execute_sql_query_tool": execute_sql_query_tool,
    "run_ols_regression_tool": run_ols_regression_tool,
    "run_arima_forecasting_tool": run_arima_forecasting_tool,
    "run_random_forest_tool": run_random_forest_tool
}

# ─── Agent Orchestration Loop ────────────────────────────────────
def run_agent_loop(user_prompt: str):
    """The main orchestrator chaining the workflow together across multiple tools."""
    st.session_state.run_log = []
    st.session_state.current_turn_dfs = []
    
    # 1. Filter Context
    relevant_schema = filter_schema(user_prompt)
    
    # 2. Decompose Intent
    with st.spinner("Decomposing question..."):
        sub_questions = decompose_question(user_prompt, relevant_schema)
        st.session_state.run_log.append(f"Sub-questions identified: {sub_questions}")
    
    # 3. Execute Tools per Question dynamically
    raw_outputs = []
    for sq in sub_questions:
        with st.spinner(f"Analyzing: '{sq}'..."):
            
            # Pass the full TOOLS array so the LLM can pick the right one
            msgs = [
                {"role": "system", "content": "You are a routing assistant. Select the appropriate tool to answer the user's question, or answer directly if no tool is needed."},
                {"role": "user", "content": sq}
            ]
            assistant_msg = raw_llm_call(msgs, tools=TOOLS)
            
            # ─── THE DYNAMIC DISPATCHER ───
            if assistant_msg.get("tool_calls"):
                for tool_call in assistant_msg["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    
                    st.session_state.run_log.append(f"Agent selected tool: {tool_name} with args: {args}")
                    
                    # Check if the tool exists in our dictionary
                    if tool_name in TOOL_DISPATCHER:
                        func = TOOL_DISPATCHER[tool_name]
                        
                        try:
                            # Special handling: SQL tool needs the dynamic schema for its auto-correct loop
                            if tool_name == "execute_sql_query_tool":
                                # Assuming the execute_sql_query_tool from the merged_app takes (question, schema)
                                result = func(user_intent=args.get("user_intent", sq), schema=relevant_schema)
                            else:
                                # Standard execution for OLS, ARIMA, and future tools using kwargs unpacking
                                result = func(**args)
                                
                            if isinstance(result, dict):
                                output_text = result.get("text", "")
                                
                                # Handle inner tool logs (like the SQL retry loop)
                                if result.get("logs"):
                                    st.session_state.run_log.extend(result["logs"])
                                    
                                # Safely inject models or dataframes into the UI state
                                if isinstance(result, dict):
                                    output_text = result.get("text", "")
                                
                                # Handle inner tool logs (like the SQL retry loop)
                                if result.get("logs"):
                                    st.session_state.run_log.extend(result["logs"])
                                    
                                # Safely inject models or dataframes into the UI state without evaluating truthiness
                                payload = result.get("data") if result.get("data") is not None else result.get("model")
                                
                                if payload is not None:
                                    st.session_state.current_turn_dfs.append(payload)
                            else:
                                # Fallback just in case a tool returns a raw string
                                output_text = str(result)

                            raw_outputs.append(f"Sub-question: {sq}\nTool Used: {tool_name}\nData: {output_text}")
                            
                        except Exception as e:
                            error_msg = f"Tool {tool_name} failed with error: {e}"
                            st.session_state.run_log.append(error_msg)
                            raw_outputs.append(error_msg)
                    else:
                        st.session_state.run_log.append(f"Warning: LLM hallucinated a non-existent tool '{tool_name}'")
            else:
                # The LLM decided it didn't need a tool for this specific sub-question
                raw_outputs.append(f"Sub-question: {sq}\nAnswer: {assistant_msg.get('content')}")

    # 4. Final Synthesis
    with st.spinner("Synthesizing final answer..."):
        synthesis_prompt = f"""You are a data insights assistant. 
        User's Original Prompt: {user_prompt}
        Raw Data Extracted across all tools: {raw_outputs}
        
        Synthesize the raw data into a clear, business-friendly summary answering the original prompt.
        If any tools failed or returned errors in the raw data, briefly mention what analysis could not be completed and why, alongside the successful insights."""
        
        final_msgs = st.session_state.messages + [{"role": "user", "content": synthesis_prompt}]
        final_response = raw_llm_call(final_msgs)
        
        # Save to memory
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append(final_response)
        
        return final_response.get("content", "")

# ─── UI ───────────────────────────────────────────────────────────
st.title("Acquisition Finance Agent - Phase 1")
st.caption("Ask questions, and the agent will autonomously decide when to query the data.")

# Initialize System Prompt & Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history (filtering out system/tool messages for a clean UI)
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the marketing data..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Trigger the agentic loop
            final_response = run_agent_loop(prompt)
            st.markdown(final_response)
            
            # Display execution context if the database was queried during this loop
            with st.expander("Agent Reasoning Log"):
                for log in st.session_state.run_log:
                    st.text(log)
                    
            if st.session_state.current_turn_dfs:
                with st.expander("View Raw Data Returned"):
                    for i, item in enumerate(st.session_state.current_turn_dfs):
                        st.write(f"**Result {i+1}**")
                        # Check if it's a pandas dataframe before trying to render it as one
                        if isinstance(item, pd.DataFrame):
                            st.dataframe(item)
                        # Check if it's a statsmodels object (they have a summary method)
                        elif hasattr(item, "summary"):
                            st.text(item.summary().as_text())
                        # Fallback for Scikit-Learn models or unknown objects
                        else:
                            st.write(str(item))
                    
        except Exception as e:
            st.error(f"Agent Orchestration Error: {e}")