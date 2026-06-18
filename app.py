import os
import json
import pandas as pd
import ssl
import streamlit as st
import sqlalchemy as sa
from databricks.sdk import WorkspaceClient
import statsmodels.api as sm

st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

# ─── Configuration ───────────────────────────────────────────────
MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
HTTP_PATH = os.environ.get('DATABRICKS_SQL_HTTP_PATH')
ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{MODEL}/invocations"
TABLE_NAME = "ai_dpm_np_sbx.sandbox.2025_fin_mktg_raw"


PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres" 
TABLE_NAME = '"sandbox"."2025_fin_mktg_raw"'

# Initialize the SDK Client (auto-authenticates using the App's Service Principal)
w = WorkspaceClient()


# ─── Schema Context ──────────────────────────────────────────────
SCHEMA_CONTEXT = f"""
Table Name: {TABLE_NAME}
This table contains marketing data. Key columns include:
- Acnt_Id (string)
- Pull_Date (timestamp)
- State (string)
- City (string)
- Sales_Channel (string)
- Activation_Plan (string)
- Promo_Cohort (string)
- Core_Package (string)
- Tfn (string)
- Tactic (string)
"""

# ─── Helper Functions ────────────────────────────────────────────
def run_sql_query(query: str) -> pd.DataFrame:
    """Connects to Lakebase Postgres using dynamic OAuth tokens."""
    
    # 1. Ask the SDK to generate the auth headers
    auth_headers = w.config.authenticate()
    
    # NEW: Get the active identity (Service Principal ID or User Email)
    current_user = w.current_user.me().user_name
    
    # 2. Extract just the raw token string from the dictionary
    auth_token = auth_headers["Authorization"].split(" ")[1]
    
    # 3. Build the SQLAlchemy Postgres connection string using the actual identity
    db_url = f"postgresql+pg8000://{current_user}:{auth_token}@{PGHOST}:5432/{PGDATABASE}"
    
    # 4. Create a default SSL context
    ssl_context = ssl.create_default_context()
    
    # 5. Pass the SSL context into the engine using connect_args
    engine = sa.create_engine(db_url, connect_args={"ssl_context": ssl_context})
    
    # 6. Execute the query
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)

def raw_llm_call(messages: list, tools: list = None) -> dict:
    """Handles standard and tool-calling requests using the SDK to auto-manage tokens."""
    
    payload = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1500
    }
    if tools:
        payload["tools"] = tools

    # The SDK natively handles headers, authentication, and token refreshes
    response = w.api_client.do(
        method="POST", 
        path=f"/serving-endpoints/{MODEL}/invocations", 
        body=payload
    )
    
    return response["choices"][0]["message"]

# ─── Tool Definition ─────────────────────────────────────────────
def execute_sql_query_tool(user_intent: str) -> str:
    """
    Sub-agent tool: Takes a natural language intent, writes SQL, executes it, 
    and returns the data as a CSV string.
    """
    sql_system_prompt = f"""You are an expert Databricks SQL analyst. 
    Convert the user's intent into a Databricks SQL query based on this schema:
    {SCHEMA_CONTEXT}
    
    RULES:
    1. ONLY return the raw SQL query. No markdown formatting.
    2. Always use the full table name: {TABLE_NAME}
    3. LIMIT queries to 100 rows maximum to ensure scalability.
    """
    
    # 1. Generate SQL
    msgs = [
        {"role": "system", "content": sql_system_prompt},
        {"role": "user", "content": user_intent}
    ]
    
    response_msg = raw_llm_call(msgs)
    sql_query = response_msg.get("content", "").replace("```sql", "").replace("```", "").strip()
    
    # Save SQL to session state so we can display it in the UI later
    st.session_state.current_turn_sql = sql_query

    # 2. Execute SQL
    try:
        df = run_sql_query(sql_query)
        st.session_state.current_turn_df = df # Save for UI rendering
        
        if df.empty:
            return "Query executed successfully, but returned 0 rows."
        
        # Return CSV string to the agent (scalability handled by the LIMIT 100 in prompt)
        return df.to_csv(index=False)
    
    except Exception as e:
        return f"Database Error executing query: {e}"
    
def run_ols_regression_tool(dependent_variable: str, independent_variables: list) -> str:
    """
    Sub-agent tool: Fetches specific numerical columns and runs an OLS multiple regression.
    """
    # 1. Build a dynamic SQL query to pull only the necessary columns
    columns_to_fetch = [dependent_variable] + independent_variables
    columns_str = ", ".join(columns_to_fetch)
    
    # We query more rows here than the SQL tool to ensure a valid sample size for regression
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} LIMIT 5000"
    
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
        return model.summary().as_text()
        
    except Exception as e:
        return f"Regression Error: {e}"

# Tool schema for the Agent loop
TOOLS = [{
        "type": "function",
        "function": {
            "name": "execute_sql_query_tool",
            "description": "Queries the Databricks marketing database. Use this ONLY when you need factual data to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_intent": {
                        "type": "string", 
                        "description": "A highly detailed natural language description of the exact data, metrics, or filters needed."
                    }
                },
                "required": ["user_intent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_ols_regression_tool",
            "description": "Performs an Ordinary Least Squares (OLS) multiple regression. Use this when the user asks to analyze the relationship, correlation, or impact of multiple independent numerical variables on a dependent target variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dependent_variable": {
                        "type": "string", 
                        "description": "The exact column name of the target numerical variable to predict (the Y variable)."
                    },
                    "independent_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of exact column names for the numerical predictor variables (the X variables)."
                    }
                },
                "required": ["dependent_variable", "independent_variables"]
            }
        }
    }
]

# ─── Agent Orchestration Loop ────────────────────────────────────
def run_agent_loop(user_prompt: str):
    """Native Python reasoning loop handling conversational memory and tool calls."""
    
    # Append user prompt to memory
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Reset UI side-effects for this turn
    st.session_state.current_turn_sql = None
    st.session_state.current_turn_df = None

    # Turn 1: Let the model evaluate history and decide if it needs a tool
    assistant_msg = raw_llm_call(st.session_state.messages, tools=TOOLS)
    st.session_state.messages.append(assistant_msg)

    # Check if the model decided to call a tool
    if assistant_msg.get("tool_calls"):
        for tool_call in assistant_msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            
            if tool_name == "execute_sql_query_tool":
                with st.spinner(f"Querying Database for: {args.get('user_intent')}..."):
                    tool_result = execute_sql_query_tool(args["user_intent"])
                    
            elif tool_name == "run_ols_regression_tool":
                with st.spinner(f"Running OLS Regression on {args.get('dependent_variable')}..."):
                    tool_result = run_ols_regression_tool(
                        args["dependent_variable"], 
                        args["independent_variables"]
                    )
            
            # Append tool execution results back to memory
            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": tool_result
            })
        
        # Turn 2: Model synthesizes the final answer using the tool data
        with st.spinner("Synthesizing answer..."):
            final_msg = raw_llm_call(st.session_state.messages)
            st.session_state.messages.append(final_msg)
            return final_msg.get("content", "")
    else:
        # Model answered purely from memory (e.g., conversational follow-up)
        return assistant_msg.get("content", "")

# ─── UI ───────────────────────────────────────────────────────────
st.title("Aquisition Finance Agent - Phase 1")
st.caption("Ask questions, and the agent will autonomously decide when to query the data.")

# Initialize System Prompt & Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful Databricks data assistant. Answer the user's questions clearly and concisely. If you need data from the database, use your provided tool."}
    ]

# Render chat history (filtering out system/tool messages for a clean UI)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant" and msg.get("content"):
        with st.chat_message("assistant"):
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
            if st.session_state.current_turn_sql:
                with st.expander("View Agent's Generated SQL Query"):
                    st.code(st.session_state.current_turn_sql, language="sql")
            if st.session_state.current_turn_df is not None:
                with st.expander("View Raw Data Result"):
                    st.dataframe(st.session_state.current_turn_df)
                    
        except Exception as e:
            st.error(f"Agent Orchestration Error: {e}")