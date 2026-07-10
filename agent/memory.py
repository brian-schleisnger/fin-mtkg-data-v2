import json
import logging
from typing import List, Dict, Any, Optional
import uuid

from llmlingua import PromptCompressor
import pandas as pd
import streamlit as st
import tiktoken

# Set up logging for graceful fallbacks
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────
DEFAULT_TOKENIZER_MODEL = "gpt-4o"  # Uses standard cl100k_base or o200k_base encoding
LLMLINGUA_MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
DEFAULT_HISTORY_TOKEN_BUDGET = 300000
DEFAULT_SCHEMA_COMPRESSION_RATE = 0.6  # Compress by 40%, retain 60% of most critical tokens

@st.cache_resource(show_spinner="Loading semantic compression model...")
def get_prompt_compressor():
    """
    Initializes and caches the LLMLingua-2 PromptCompressor as a Streamlit singleton.
    Uses @st.cache_resource so the model is only loaded into Databricks cluster memory once.
    """
    try:
        
        # We explicitly enable llmlingua2 for faster, structured-data-friendly compression
        return PromptCompressor(
            model_name=LLMLINGUA_MODEL_NAME,
            use_llmlingua2=True,
            device_map="auto"  # Automatically uses Databricks GPU if available, else CPU
        )
    except ImportError:
        logger.warning("`llmlingua` library not installed. Falling back to uncompressed text.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LLMLingua: {e}. Falling back to uncompressed text.")
        return None


class ContextOptimizer:
    """
    Manages conversational memory pruning and semantic prompt compression for the agentic loop.
    """
    def __init__(self, tokenizer_model: str = DEFAULT_TOKENIZER_MODEL):
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Returns the exact number of tokens in a string."""
        if not text:
            return 0
        return len(self.encoding.encode(str(text)))

    def prune_history_by_budget(
        self, 
        messages: List[Dict[str, Any]], 
        max_tokens: int = DEFAULT_HISTORY_TOKEN_BUDGET
    ) -> List[Dict[str, Any]]:
        """
        Replaces arbitrary turn-slicing (`history[-6:]`) with exact token-budget pruning.
        Iterates backwards from newest to oldest messages, including them until the budget is hit.
        """
        if not messages:
            return []

        pruned_history = []
        current_token_count = 0

        # Iterate backwards from most recent message
        for msg in reversed(messages):
            # Only process user and assistant roles for history context
            role = msg.get("role", "")
            if role not in ["user", "assistant"]:
                continue

            content = str(msg.get("content", ""))
            # Calculate tokens for this message including role framing overhead (~4 tokens)
            msg_tokens = self.count_tokens(content) + 4

            if current_token_count + msg_tokens <= max_tokens:
                pruned_history.insert(0, msg)  # Prepend to maintain chronological order
                current_token_count += msg_tokens
            else:
                # If adding this message exceeds budget, stop traversing history
                break

        return pruned_history

    def compress_text(
        self, 
        text: str, 
        target_rate: float = 0.5, 
        context_instruction: str = "Preserve numbers, SQL columns, and table relationships."
    ) -> str:
        """
        Uses LLMLingua-2 to semantically compress verbose text (like raw tool outputs or logs)
        while strictly preserving numbers, entity names, and analytical context.
        """
        if not text or self.count_tokens(text) < 300:
            # Don't waste compute compressing short strings
            return text

        compressor = get_prompt_compressor()
        if not compressor:
            return text

        try:
            compressed_result = compressor.compress_prompt(
                prompt=text,
                rate=target_rate,
                force_tokens=["\n", "?", "!", ".", "=", "-", "_", "SELECT", "FROM", "WHERE", "GROUP BY"],
                drop_consecutively=True
            )
            return compressed_result.get("compressed_prompt", text)
        except Exception as e:
            logger.warning(f"Compression failed during execution: {e}. Returning original text.")
            return text

    def compress_schema_context(self, schema_dict: Dict[str, Any], target_rate: float = DEFAULT_SCHEMA_COMPRESSION_RATE) -> str:
        """
        Specialized compressor for JSON data dictionaries. Converts schema to string and compresses
        while protecting table names and key types.
        """
        raw_schema_str = json.dumps(schema_dict, indent=2)
        
        # If schema is relatively small (< 800 tokens), return as-is to avoid overhead
        if self.count_tokens(raw_schema_str) < 800:
            return raw_schema_str

        return self.compress_text(
            text=raw_schema_str,
            target_rate=target_rate,
            context_instruction="Retain all table names, column names, data types, and primary/foreign keys."
        )

    def format_history_for_prompt(self, messages: List[Dict[str, Any]], max_tokens: int = DEFAULT_HISTORY_TOKEN_BUDGET) -> str:
        """
        All-in-one helper: Prunes history by token budget, formats to a readable string,
        and compresses it if it's still dense.
        """
        pruned_msgs = self.prune_history_by_budget(messages, max_tokens=max_tokens)
        if not pruned_msgs:
            return "No previous conversation history."

        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in pruned_msgs if msg.get("content")
        ])

        # If even the pruned history is very dense (e.g., contains past dataframe dumps), compress it
        if self.count_tokens(history_text) > 1500:
            history_text = self.compress_text(history_text, target_rate=0.6)

        return history_text
    

class DataFrameMemory:
    """
    In-memory registry to hold DataFrames generated during a conversation turn.
    Allows downstream tools to reference upstream data via string IDs.
    """
    def __init__(self):
        self.registry: Dict[str, pd.DataFrame] = {}

    def save_df(self, df: pd.DataFrame) -> str:
        """Saves a DataFrame and returns a unique reference ID."""
        df_id = f"df_{uuid.uuid4().hex[:8]}"
        self.registry[df_id] = df
        return df_id

    def get_df(self, df_id: str) -> Optional[pd.DataFrame]:
        """Retrieves a DataFrame by its ID."""
        return self.registry.get(df_id)

    def clear(self):
        """Clears the registry to free up memory between isolated runs."""
        self.registry.clear()

# ─── Session-Scoped Getters ──────────────────────────────────────────────
# These replace the old module-level globals.  Each Streamlit browser session
# gets its own isolated DataFrameMemory and ContextOptimizer instance stored
# in st.session_state, so two users never share the same registry or tokenizer.

def get_df_memory() -> DataFrameMemory:
    """
    Returns the DataFrameMemory instance for the current Streamlit session.
    Creates a new one on first call within a session.
    """
    if "df_memory" not in st.session_state:
        st.session_state["df_memory"] = DataFrameMemory()
    return st.session_state["df_memory"]


def get_context_optimizer() -> ContextOptimizer:
    """
    Returns the ContextOptimizer instance for the current Streamlit session.
    Creates a new one on first call within a session.  ContextOptimizer holds
    only a tiktoken encoding (stateless beyond that), so sharing it is safe,
    but keeping it in session_state is consistent and avoids cross-session
    confusion if the tokenizer model ever becomes configurable per user.
    """
    if "context_optimizer" not in st.session_state:
        st.session_state["context_optimizer"] = ContextOptimizer()
    return st.session_state["context_optimizer"]