import os
import json
import pandas as pd
import streamlit as st
import sqlalchemy as sa
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

# ─── Configuration ───────────────────────────────────────────────
MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
HTTP_PATH = os.environ.get('DATABRICKS_SQL_HTTP_PATH')
ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{MODEL}/invocations"
TABLE_NAME = "ai_dpm_np_sbx.sandbox.2025_marketing_raw"


PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres" 
TABLE_NAME = "2025_marketing_raw" # Update to whatever you named the synced table

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
    
    # 1. Ask the SDK to generate a fresh token for the current Service Principal identity
    auth_token = w.config.authenticate()
    
    # 2. Build the SQLAlchemy Postgres connection string
    # We use 'token' as the username and the generated OAuth token as the password
    db_url = f"postgresql+psycopg2://token:{auth_token}@{PGHOST}:5432/{PGDATABASE}?sslmode=require"
    
    engine = sa.create_engine(db_url)
    
    # 3. Execute the query
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
}]

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

    # Check if the model decided to call our SQL tool
    if assistant_msg.get("tool_calls"):
        for tool_call in assistant_msg["tool_calls"]:
            if tool_call["function"]["name"] == "execute_sql_query_tool":
                # Parse arguments and run the tool
                args = json.loads(tool_call["function"]["arguments"])
                with st.spinner(f"Querying Database for: {args.get('user_intent')}..."):
                    tool_result_csv = execute_sql_query_tool(args["user_intent"])
                
                # Append tool execution results back to memory
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "content": tool_result_csv
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