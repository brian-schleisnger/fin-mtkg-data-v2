import json
import os
from pathlib import Path
import ssl

from databricks.sdk import WorkspaceClient
import instructor
import mlflow
from openai import OpenAI
import pandas as pd
import pg8000
from pydantic import BaseModel
import sqlalchemy as sa
import streamlit as st


# ─── Configuration ───────────────────────────────────────────────
class ModelConfig:
    ROUTING_MODEL = "databricks-gpt-5-4-nano"
    SYNTHESIS_MODEL = "databricks-gpt-5-4-nano"

def set_model_tier(tier: str):
    if tier == "Advanced (Thorough)":
        ModelConfig.ROUTING_MODEL = "databricks-gpt-5-4-nano"
        ModelConfig.SYNTHESIS_MODEL = "databricks-dbrx-instruct" # Or any frontier model you gain access to
    else:
        ModelConfig.ROUTING_MODEL = "databricks-gpt-5-4-nano"
        ModelConfig.SYNTHESIS_MODEL = "databricks-gpt-5-4-nano"
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres" 

# Initialize the SDK Client
w = WorkspaceClient()
databricks_host = w.config.host

def get_auth_token() -> str:
    """Dynamically fetches a fresh token from the Databricks SDK on every call."""
    auth_headers = w.config.authenticate()
    return auth_headers["Authorization"].split(" ")[1]

class DynamicOpenAIClient:
    """A proxy wrapper that ensures every API call uses a freshly rotated token."""
    def __getattr__(self, name):
        fresh_token = get_auth_token()
        client = OpenAI(
            api_key=fresh_token,
            base_url=f"{databricks_host}/serving-endpoints"
        )
        return getattr(client, name)

# 1. CREATE THE RAW CLIENT PROXY (Zero changes needed in app.py!)
raw_client = DynamicOpenAIClient()

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
    """
    Caches the SQLAlchemy engine pool, but delegates physical connection 
    creation to a dynamic function so Postgres always gets a fresh token.
    """
    ssl_context = ssl.create_default_context()
    
    def get_fresh_connection():
        fresh_token = get_auth_token()
        current_user = w.current_user.me().user_name
        return pg8000.connect(
            user=current_user,
            password=fresh_token,
            host=PGHOST,
            port=5432,
            database=PGDATABASE,
            ssl_context=ssl_context
        )
    
    # We pass an empty URL and bind the dynamic connection creator
    return sa.create_engine("postgresql+pg8000://", creator=get_fresh_connection)

@mlflow.trace(name="run_sql_query")
def run_sql_query(query: str) -> pd.DataFrame:
    engine = get_db_engine()
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)

def llm_call(messages: list, response_model: BaseModel, model_name: str = None):
    """Replaces raw_llm_call for tasks that require strict JSON outputs, tracking tokens accurately."""
    fresh_instructor_client = instructor.from_openai(
        OpenAI(
            api_key=get_auth_token(),
            base_url=f"{databricks_host}/serving-endpoints"
        )
    )
    # Use create_with_completion to get both the Pydantic model AND the raw OpenAI response
    model_res, raw_res = fresh_instructor_client.chat.completions.create_with_completion(
        model=model_name or ModelConfig.ROUTING_MODEL,
        messages=messages,
        response_model=response_model,
        max_retries=3
    )
    track_tokens(raw_res)  # Now decomposition tokens are accurately tracked!
    return model_res

def track_tokens(response):
    """Directly extracts token usage from a live OpenAI/Databricks SDK response object."""
    if hasattr(response, "usage") and response.usage:
        # Explicitly check modern input/output naming first, then fall back to prompt/completion
        input_t = getattr(response.usage, "input_tokens", 0) or getattr(response.usage, "prompt_tokens", 0)
        output_t = getattr(response.usage, "output_tokens", 0) or getattr(response.usage, "completion_tokens", 0)
        total_t = getattr(response.usage, "total_tokens", 0) or (input_t + output_t)
        
        if "input_tokens" in st.session_state:
            st.session_state.input_tokens += input_t
        if "output_tokens" in st.session_state:
            st.session_state.output_tokens += output_t
        if "total_tokens" in st.session_state:
            st.session_state.total_tokens += total_t

def get_join_clause(table_a: str, table_b: str) -> str:
    """Returns the correct ON clause regardless of the order the tables are passed."""
    return TABLE_RELATIONSHIPS.get((table_a, table_b)) or TABLE_RELATIONSHIPS.get((table_b, table_a))
