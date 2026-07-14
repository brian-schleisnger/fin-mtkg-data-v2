import copy
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

# !! ADD YOUR MODEL ENDPOINTS HERE !!
# Keys are the display names shown in the UI dropdown.
# Values are the Databricks serving endpoint names passed to the OpenAI-compatible API.
AVAILABLE_MODELS: dict[str, str] = {
    "GPT 5.4 Nano (Low)": "system.ai.gpt-5-4-nano",
    "Gemini Flash Lite (Low-Medium)": "system.ai.gemini-3-1-flash-lite",
    "Gemini Flash (Medium)": "system.ai.gemini-3-5-flash",
    "Claude Sonet (Medium-High)": "system.ai.claude-sonnet-5",
    "Claude Opus (High)": "system.ai.claude-opus-4-8"
}

class ModelConfig:
    # The single model used for all LLM calls (routing, decomposition, synthesis).
    # Updated at runtime by set_active_model() when the user picks from the sidebar.
    ACTIVE_MODEL: str = next(iter(AVAILABLE_MODELS.values()), "")

def set_active_model(display_name: str) -> None:
    """Updates the single active model endpoint based on the user's sidebar selection."""
    ModelConfig.ACTIVE_MODEL = AVAILABLE_MODELS.get(display_name, ModelConfig.ACTIVE_MODEL)
        
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
            base_url=f"{databricks_host}/ai-gateway/mlflow/v1"
        )
        
        # Intercept the chat module to sanitize inputs for Claude
        if name == "chat":
            class ChatProxy:
                @property
                def completions(self):
                    class CompletionsProxy:
                        def create(self, *args, **kwargs):
                            model = kwargs.get("model", "")
                            
                            # Claude strictly rejects 'strict: True' in tool definitions
                            if "claude" in model.lower() and "tools" in kwargs:
                                safe_tools = copy.deepcopy(kwargs["tools"])
                                for tool in safe_tools:
                                    if "function" in tool and "strict" in tool["function"]:
                                        del tool["function"]["strict"]
                                kwargs["tools"] = safe_tools
                                
                            return client.chat.completions.create(*args, **kwargs)
                    return CompletionsProxy()
            return ChatProxy()
            
        return getattr(client, name)

# 1. CREATE THE RAW CLIENT PROXY (Zero changes needed in app.py!)
raw_client = DynamicOpenAIClient()

mlflow.openai.autolog()

# ─── Conformed Dimension Mapping ─────────────────────────────────
# Keys ('year', 'month', 'day') are standardized conceptual dimensions.
# Values are the actual column names used inside that specific table.
TABLE_DIMENSIONS = {
    '"sandbox"."dbs_marketing_spend_sync"': {
        "year": "year",
        "month": "month",
    },
    '"sandbox"."acquisition_data_v3"': {
        "year": "Activation_Year",
        "month": "Activation_Month",
    },
    '"sandbox"."subcount_data_synced"': {
        "year": "Year",
        "month": "Month",
    }
}

# ─── Schema Context ──────────────────────────────────────────────
SCHEMA_CONFIG = {
    '"sandbox"."acquisition_data_v3"': "acquisition_data_dictionary.json",
    '"sandbox"."dbs_marketing_spend_sync"': "marketing_spend_dictionary.json",
    '"sandbox"."subcount_data_synced"': "subscriber_count_dictionary.json"
}

ALIASES = {
    '"sandbox"."acquisition_data_v3"': ['arpu', 'cogs', 'sac', 'churn', 'mcf', 'npv', 'activations', 'retention', 'revenue', 'cost of goods sold', 'subscriber acquisition cost', 'lifetime value', 'clv', 'profitability'],
    '"sandbox"."dbs_marketing_spend_sync"': ['marketing', 'spend', 'budget', 'cpa', 'tactic', 'digital', 'tv', 'cost per acquisition', 'ad spend', 'campaign', 'media', 'advertising'],
    '"sandbox"."subcount_data_synced"': ['subscribers', 'subscriber count', 'subscriber balance', 'gross adds', 'net adds', 'disconnects', 'churn rate', 'beginning subscribers', 'ending subscribers', 'local retail', 'sales partner', 'national retail', 'telco activations', 'indirect activations', 'direct activations', 'commercial activations', 'subscriber growth', 'subscriber base']
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
            schema_data["related_concepts"] = ALIASES.get(table_name, [])
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

def _is_gpt_model(model_name: str) -> bool:
    """Returns True if the endpoint name indicates a GPT model (OpenAI-native tool-calling)."""
    return "gpt" in model_name.lower()


def _extract_text_content(message) -> str:
    """
    Safely extracts the text string from a ChatCompletionMessage regardless of
    whether content is a plain string or a Gemini-style list of content blocks
    (e.g. [{'type': 'text', 'text': '...', 'thoughtSignature': '...'}]).
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Grab the first block with type='text' and return its text value
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return str(content)


def llm_call(messages: list, response_model: BaseModel, model_name: str = None):
    """
    Structured-output LLM call with cross-model compatibility.

    - GPT endpoints: uses instructor TOOLS mode (native tool-calling + JSON schema).
    - All other endpoints (Gemini, Claude, etc.): calls the raw client directly,
      extracts the text from Gemini's content-block list format, then parses with
      Pydantic. This avoids both the $defs/$ref tool-schema rejection AND the
      instructor parse failure caused by Gemini returning content as a list when
      extended thinking is enabled.
    """
    resolved_model = model_name or ModelConfig.ACTIVE_MODEL

    openai_client = OpenAI(
        api_key=get_auth_token(),
        # UPDATE THIS LINE:
        base_url=f"{databricks_host}/ai-gateway/mlflow/v1"
    )

    if _is_gpt_model(resolved_model):
        # GPT: let instructor handle everything via tool-calling
        fresh_instructor_client = instructor.from_openai(openai_client)
        model_res, raw_res = fresh_instructor_client.chat.completions.create_with_completion(
            model=resolved_model,
            messages=messages,
            response_model=response_model,
            max_retries=3
        )
        track_tokens(raw_res)
        return model_res
    else:
        # Non-GPT (Gemini, Claude, etc.): inject a plain-text JSON instruction.
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        json_instruction = (
            f"\n\nYou MUST respond with a single valid JSON object that strictly conforms "
            f"to this schema and nothing else — no markdown, no explanation:\n{schema_str}"
        )

        augmented_messages = list(messages)
        
        # Check if a system prompt already exists in the message history
        system_prompt_index = next((i for i, msg in enumerate(augmented_messages) if msg.get("role") == "system"), -1)

        if system_prompt_index != -1:
            # Safely append our JSON instructions to the EXISTING system prompt
            augmented_messages[system_prompt_index] = dict(augmented_messages[system_prompt_index])
            augmented_messages[system_prompt_index]["content"] += json_instruction
        else:
            # Create a brand new system prompt if none exists
            augmented_messages = [{"role": "system", "content": json_instruction.strip()}] + augmented_messages

        last_exc = None
        for attempt in range(3):
            raw_res = openai_client.chat.completions.create(
                model=resolved_model,
                messages=augmented_messages
            )
            track_tokens(raw_res)
            text = _extract_text_content(raw_res.choices[0].message).strip()

            # Strip markdown fences if the model wrapped the JSON anyway
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            try:
                return response_model.model_validate_json(text)
            except Exception as exc:
                last_exc = exc
                # Feed the error back so the model can self-correct on the next attempt
                augmented_messages = augmented_messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That response failed validation: {exc}. Please return only valid JSON."}
                ]

        raise last_exc

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
    """
    Dynamically generates a SQL ON clause by intersecting the shared 
    conformed dimensions between two tables.
    """
    dims_a = TABLE_DIMENSIONS.get(table_a)
    dims_b = TABLE_DIMENSIONS.get(table_b)
    
    if not dims_a or not dims_b:
        raise ValueError(f"One or both tables not found in TABLE_DIMENSIONS: {table_a}, {table_b}")
        
    # Find the overlapping dimensional concepts (e.g., {'year', 'month'})
    shared_concepts = set(dims_a.keys()) & set(dims_b.keys())
    
    if not shared_concepts:
        raise ValueError(f"No shared dimensions found between {table_a} and {table_b} to join on.")
        
    join_conditions = []
    for concept in shared_concepts:
        col_a = dims_a[concept]
        col_b = dims_b[concept]
        # Generates: "table_a"."col_a" = "table_b"."col_b"
        join_conditions.append(f'{table_a}."{col_a}" = {table_b}."{col_b}"')
        
    return " AND ".join(join_conditions)
