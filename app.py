import os
import pandas as pd
import streamlit as st
import requests
from databricks import sql

st.set_page_config(page_title="Dataset Q&A", page_icon="📊", layout="wide")

# ─── Configuration ───────────────────────────────────────────────
# Databricks API & Auth setup
MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
HTTP_PATH = os.environ.get('DATABRICKS_SQL_HTTP_PATH')
ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{MODEL}/invocations"

# The Delta Table you created
TABLE_NAME = "ai_dpm_np_sbx.sandbox.2025_marketing_raw"

# ─── Helper Functions ────────────────────────────────────────────
def run_sql_query(query: str) -> pd.DataFrame:
    """Connects to Databricks SQL Warehouse and executes a query."""
    # Remove the https:// from the host for the SQL connector
    hostname = DATABRICKS_HOST.replace("https://", "")
    
    with sql.connect(
        server_hostname=hostname,
        http_path=HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            # Fetch results as an Arrow table, then convert to pandas
            return cursor.fetchall_arrow().to_pandas()

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Sends a request to the Databricks Model Serving endpoint."""
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# ─── Schema Context ──────────────────────────────────────────────
# In a production app, you might query `DESCRIBE TABLE` dynamically, 
# but hardcoding the core columns saves an API call and speeds up the app.
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

# ─── System Prompts ──────────────────────────────────────────────
# Prompt 1: Text to SQL
SQL_SYSTEM_PROMPT = f"""You are an expert Databricks SQL analyst. 
Your job is to take a user's question and write a Databricks SQL query to answer it.
You have access to a table with this schema:
{SCHEMA_CONTEXT}

RULES:
1. ONLY return the raw SQL query.
2. Do NOT wrap the query in ```sql ``` markdown blocks.
3. Do NOT include any explanations or conversational text.
4. Always use the full table name: {TABLE_NAME}
5. LIMIT queries to 100 rows maximum to prevent browser crashes.
"""

# Prompt 2: Data to Natural Language
NL_SYSTEM_PROMPT = """You are a helpful data assistant. 
You will be provided with a user's question and the raw data results from a SQL query.
Your job is to explain the data results in plain, conversational English. 
Be concise, clear, and highlight the exact numbers requested.
"""

# ─── UI ───────────────────────────────────────────────────────────
st.title("📊 Ask Questions About Your Data")

with st.sidebar:
    st.header("🔧 Connection Debug")
    st.write(f"**Host Found:** {bool(DATABRICKS_HOST)}")
    st.write(f"**HTTP Path Found:** {bool(HTTP_PATH)}")
    st.write(f"**Token Found:** {bool(DATABRICKS_TOKEN)}")

# Chat interface state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message:
            with st.expander("View SQL Query"):
                st.code(message["sql"], language="sql")
        if "result_df" in message:
            st.dataframe(message["result_df"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the marketing data..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL query..."):
            try:
                # Step A: Get SQL from LLM
                sql_query = call_llm(SQL_SYSTEM_PROMPT, prompt)
                
                # Strip markdown just in case the LLM disobeys the prompt rules
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            except Exception as e:
                st.error(f"Error generating SQL: {e}")
                st.stop()

        with st.spinner("Querying Databricks SQL Warehouse..."):
            try:
                # Step B: Execute SQL against Delta Table
                result_df = run_sql_query(sql_query)
            except Exception as e:
                st.error(f"Error executing SQL: {e}")
                with st.expander("Attempted Query"):
                    st.code(sql_query, language="sql")
                st.stop()

        with st.spinner("Translating results..."):
            try:
                # Step C: Generate Natural Language Summary
                # Convert a sample of the dataframe to markdown/string for the LLM context
                data_context = result_df.head(10).to_markdown()
                nl_prompt = f"User Question: {prompt}\n\nData Results:\n{data_context}"
                
                nl_summary = call_llm(NL_SYSTEM_PROMPT, nl_prompt)
                
                # Display everything to the user
                st.markdown(nl_summary)
                with st.expander("View Generated SQL Query"):
                    st.code(sql_query, language="sql")
                st.dataframe(result_df)

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": nl_summary,
                    "sql": sql_query,
                    "result_df": result_df
                })

            except Exception as e:
                st.error(f"Error summarizing results: {e}")
