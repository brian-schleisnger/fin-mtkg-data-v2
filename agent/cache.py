import json
import os
import pickle
import sqlite3
import time
from typing import Optional, Dict, Any, List

import numpy as np
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import existing authentication and configuration from your base module
from toolkit.base import get_auth_token, databricks_host

# ─── Configuration ───────────────────────────────────────────────
# Standard Databricks hosted embedding model
EMBEDDING_MODEL = "system.ai.gemini-3-1-flash-lite" 
# Cosine similarity threshold (0.90 to 0.95 is ideal for semantic matching)
SIMILARITY_THRESHOLD = 0.92 
CACHE_DB_PATH = "semantic_cache.db"

class SemanticCache:
    """
    A lightweight, persistent semantic cache tailored for complex agent outputs
    (Text + Pandas DataFrames + Plotly Figures).
    """
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database for persistent cache storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT UNIQUE,
                    embedding BLOB,
                    final_text TEXT,
                    dfs_pickle BLOB,
                    figures_pickle BLOB,
                    timestamp REAL
                )
            """)
            conn.commit()

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Fetches a dense vector embedding for the given text from the Databricks
        embedding endpoint. Always uses a freshly fetched auth token.
        Input is lowercased and stripped before embedding for consistent similarity scoring.
        """
        # Dynamically grab the token right before calling the embedding model
        client = OpenAI(
            api_key=get_auth_token(),
            base_url=f"{databricks_host}/serving-endpoints"
        )
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text.strip().lower()]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def check_cache(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Checks if a semantically equivalent prompt exists in the cache.
        Returns the cached dictionary if similarity >= SIMILARITY_THRESHOLD, else None.
        """
        try:
            query_vector = self._get_embedding(user_prompt)
        except Exception as e:
            st.warning(f"Cache embedding generation failed: {e}")
            return None

        best_match = None
        highest_sim = 0.0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT prompt, embedding, final_text, dfs_pickle, figures_pickle FROM semantic_cache")
            rows = cursor.fetchall()

        for row in rows:
            cached_prompt, emb_blob, final_text, dfs_blob, figs_blob = row
            cached_vector = np.frombuffer(emb_blob, dtype=np.float32)
            
            sim = self._cosine_similarity(query_vector, cached_vector)
            if sim > highest_sim:
                highest_sim = sim
                best_match = (final_text, dfs_blob, figs_blob, cached_prompt)

        # If a strong semantic match is found, deserialize and return the payload
        if highest_sim >= SIMILARITY_THRESHOLD and best_match:
            final_text, dfs_blob, figs_blob, matched_prompt = best_match
            
            # Deserialize DataFrames and Figures safely
            dfs = pickle.loads(dfs_blob) if dfs_blob else []
            figures = pickle.loads(figs_blob) if figs_blob else []
            
            return {
                "content": final_text,
                "dfs": dfs,
                "figures": figures,
                "matched_prompt": matched_prompt,
                "similarity": round(highest_sim, 3)
            }

        return None

    def save_to_cache(self, user_prompt: str, final_text: str, dfs: List[pd.DataFrame], figures: List[go.Figure]):
        """
        Serializes and saves a successful agent execution to the database.
        """
        try:
            embedding = self._get_embedding(user_prompt)
            emb_blob = embedding.tobytes()
            
            # Serialize complex Python objects
            dfs_blob = pickle.dumps(dfs)
            figs_blob = pickle.dumps(figures)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO semantic_cache 
                    (prompt, embedding, final_text, dfs_pickle, figures_pickle, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_prompt.strip().lower(), emb_blob, final_text, dfs_blob, figs_blob, time.time()))
                conn.commit()
        except Exception as e:
            # We silently log cache save errors so we never crash the user's primary UI loop
            print(f"Failed to save execution to semantic cache: {e}")

    def delete_from_cache(self, user_prompt: str):
        """
        Permanently removes any cache entry whose prompt is a semantic match for
        user_prompt (above SIMILARITY_THRESHOLD), ensuring the next run is always
        a fresh execution rather than a cache hit.
        """
        try:
            query_vector = self._get_embedding(user_prompt)
        except Exception as e:
            print(f"Cache eviction embedding failed: {e}")
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM semantic_cache")
            rows = cursor.fetchall()

            ids_to_delete = []
            for row_id, emb_blob in rows:
                cached_vector = np.frombuffer(emb_blob, dtype=np.float32)
                sim = self._cosine_similarity(query_vector, cached_vector)
                if sim >= SIMILARITY_THRESHOLD:
                    ids_to_delete.append(row_id)

            if ids_to_delete:
                cursor.executemany(
                    "DELETE FROM semantic_cache WHERE id = ?",
                    [(row_id,) for row_id in ids_to_delete]
                )
                conn.commit()

# Instantiate a global cache object to be imported into app.py
agent_cache = SemanticCache()