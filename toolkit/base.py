import json
import os
from pathlib import Path
import ssl

from databricks.sdk import WorkspaceClient
import instructor
import mlflow
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel
import sqlalchemy as sa
import streamlit as st


# ─── Configuration ───────────────────────────────────────────────
MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres" 

# Initialize the SDK Client
w = WorkspaceClient()
auth_headers = w.config.authenticate()
auth_token = auth_headers["Authorization"].split(" ")[1]
databricks_host = w.config.host

# 1. CREATE THE RAW CLIENT (For standard chat & native tool calling)
raw_client = OpenAI(
    api_key=auth_token,
    base_url=f"{databricks_host}/serving-endpoints"
)

# 2. CREATE THE INSTRUCTOR CLIENT (For Pydantic structured outputs)
instructor_client = instructor.from_openai(raw_client)

mlflow.openai.autolog()

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

# ─── Schema Context ──────────────────────────────────────────────
SCHEMA_CONFIG = {
    '"sandbox"."acquisition_data_v3"': "acquisition_data_dictionary.json",
    '"sandbox"."dbs_marketing_spend_sync"': "marketing_spend_dictionary.json"
}

DATA_DICTIONARY = {}

# Step up from /toolkit/base.py to the root project folder
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

for table_name, file_name in SCHEMA_CONFIG.items():
    dict_path = PROJECT_ROOT / "dictionaries" / file_name
    try:
        with dict_path.open("r", encoding="utf-8") as f:
            schema_data = json.load(f)
            schema_data["table_name"] = table_name 
            DATA_DICTIONARY[table_name] = schema_data
    except Exception as e:
        st.error(f"Error loading schema {file_name}: {e}")


# ─── Helper Functions ────────────────────────────────────────────
@st.cache_resource
def get_db_engine():
    current_user = w.current_user.me().user_name
    db_url = f"postgresql+pg8000://{current_user}:{auth_token}@{PGHOST}:5432/{PGDATABASE}"
    ssl_context = ssl.create_default_context()
    return sa.create_engine(db_url, connect_args={"ssl_context": ssl_context})

@mlflow.trace(name="run_sql_query")
def run_sql_query(query: str) -> pd.DataFrame:
    engine = get_db_engine()
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)

def llm_call(messages: list, response_model: BaseModel):
    """Replaces raw_llm_call for tasks that require strict JSON outputs."""
    return instructor_client.chat.completions.create(
        model=MODEL, # e.g., "databricks-dbrx-instruct"
        messages=messages,
        response_model=response_model, # Instructor handles the magic here
        max_retries=3 # Instructor automatically feeds Pydantic validation errors back to the LLM
    )

def get_join_clause(table_a: str, table_b: str) -> str:
    """Returns the correct ON clause regardless of the order the tables are passed."""
    return TABLE_RELATIONSHIPS.get((table_a, table_b)) or TABLE_RELATIONSHIPS.get((table_b, table_a))

def track_tokens(response):
    """Directly extracts token usage from a live OpenAI/Databricks SDK response object."""
    if hasattr(response, "usage") and response.usage:
        # Check both OpenAI (.prompt_tokens) and OTel/Databricks (.input_tokens) naming conventions
        prompt_t = getattr(response.usage, "prompt_tokens", 0) or getattr(response.usage, "input_tokens", 0)
        comp_t = getattr(response.usage, "completion_tokens", 0) or getattr(response.usage, "output_tokens", 0)
        total_t = getattr(response.usage, "total_tokens", 0) or (prompt_t + comp_t)
        
        st.session_state.prompt_tokens += prompt_t
        st.session_state.completion_tokens += comp_t
        st.session_state.total_tokens += total_t