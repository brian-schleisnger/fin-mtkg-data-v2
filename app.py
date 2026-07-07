import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from databricks.sdk import WorkspaceClient
import mlflow
import openpyxl
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

# ─── 1. ENVIRONMENT BOOTSTRAPPING (CACHED) ───────────────────────────────
@st.cache_resource
def bootstrap_environment():
    """
    Runs offline caching, PyTorch CPU workarounds, and wheel installations exactly ONCE 
    per server lifecycle, preventing Streamlit from re-running them on every UI interaction.
    """
    print("Initializing environment bootstrap...")
    
    # --- A. TIKTOKEN OFFLINE CACHE SETUP ---
    cache_dir = "/tmp/tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    tiktoken_url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    url_hash = hashlib.sha1(tiktoken_url.encode()).hexdigest()
    tiktoken_cache_path = os.path.join(cache_dir, url_hash)

    if not os.path.exists(tiktoken_cache_path):
        print("Downloading offline tiktoken vocabulary from Workspace via SDK...")
        w = WorkspaceClient()
        with w.workspace.download("/Shared/whl-loading/o200k_base.tiktoken") as response:
            with open(tiktoken_cache_path, "wb") as outfile:
                shutil.copyfileobj(response, outfile)
    else:
        print("Tiktoken offline cache already present!")

    # --- B. PYTORCH CPU WORKAROUND & MODULE FLUSHING ---
    try:
        import torch
        print("PyTorch is already installed and working cleanly!")
    except (ImportError, OSError, ValueError) as e:
        print(f"PyTorch missing or broken C++ CUDA dependencies ({type(e).__name__}). Starting clean CPU setup...")

        wheel_name = "torch-2.4.0+cpu-cp311-cp311-linux_x86_64.whl"
        wheel_path = f"/tmp/{wheel_name}"

        if not os.path.exists(wheel_path):
            print("Connecting to Databricks Workspace via SDK...")
            w = WorkspaceClient()
            workspace_path = f"/Shared/whl-loading/{wheel_name}"
            print(f" -> Downloading CPU-only PyTorch from {workspace_path}...")

            with w.workspace.download(workspace_path) as response:
                with open(wheel_path, "wb") as outfile:
                    shutil.copyfileobj(response, outfile)
        else:
            print(f"Found existing {wheel_path} on disk. Skipping download...")

        print("Force-installing PyTorch CPU into active virtual environment...")
        subprocess.check_call(["pip", "install", wheel_path, "--no-deps", "--force-reinstall"])

        print("Flushing module cache so Python sees the fresh CPU installation...")
        for mod in list(sys.modules.keys()):
            if mod.startswith("torch"):
                del sys.modules[mod]
        importlib.invalidate_caches()

    # --- C. GIT REPO DEPENDENT PACKAGES ---
    print("Installing dependent packages from Git repo...")
    for pkg in [
        "whls/accelerate-1.14.0-py3-none-any.whl",
        "whls/llmlingua-0.2.2-py3-none-any.whl",
    ]:
        subprocess.check_call(["pip", "install", pkg, "--no-deps"])
        
    print("Environment bootstrap complete!")

# Execute bootstrap immediately before importing heavy ML/agent modules
bootstrap_environment()


# ─── 2. AGENT & TOOLKIT IMPORTS ──────────────────────────────────────────
# Now importing our cleanly extracted backend loop from the agent module
from agent.loop import run_agent_loop
from toolkit.base import MODEL

# ─── 3. GLOBAL CONFIGURATION & UI HELPERS ────────────────────────────────


# Set MLflow experiment once globally so it doesn't fire API calls on every chat turn
mlflow.set_experiment("/Workspace/Users/brian.schlesinger@dish.com")
def create_excel_buffer(data_list: list) -> bytes:
    """Extracts DataFrames from the agent's output and writes them to an Excel buffer."""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sheet_counter = 1
        has_data = False
        
        for item in data_list:
            if isinstance(item, pd.DataFrame):
                # Write each DataFrame to its own tab
                item.to_excel(writer, index=False, sheet_name=f"Result_{sheet_counter}")
                sheet_counter += 1
                has_data = True
                
        # Fallback if the agent only returned models/text but no tabular data
        if not has_data:
            pd.DataFrame({"Message": ["No tabular data available for this query."]}).to_excel(writer, index=False, sheet_name="No Data")
            
    return buffer.getvalue()

# ─── 4. SESSION STATE INITIALIZATION ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    st.session_state.prompt_tokens = 0
    st.session_state.completion_tokens = 0

# ─── 5. CHAT HISTORY RENDERING ───────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render historical figures safely using isinstance
            if msg.get("figures"):
                for j, fig in enumerate(msg["figures"]):
                    if isinstance(fig, go.Figure):
                        fig.update_layout(height=500, colorway=["#C4262E", "#A2A4A3", "#000000"])
                        st.plotly_chart(fig, use_container_width=True, key=f"fig_{i}_{j}")
                    
            # Render historical Excel Download Buttons
            if msg.get("dfs"):
                excel_data = create_excel_buffer(msg["dfs"])
                st.download_button(
                    label="📥 Download Raw Data to Excel",
                    data=excel_data,
                    file_name=f"agent_data_export_{i}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_hist_{i}" 
                )
            
            # Render historical Reasoning Logs
            if msg.get("run_log"):
                with st.expander("Agent Reasoning Log"):
                    for log in msg["run_log"]:
                        st.text(log)

if prompt := st.chat_input("Ask a question about the marketing data..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Run the backend loop, passing current chat history
            with st.spinner("Analyzing..."):
                result = run_agent_loop(prompt, st.session_state.messages)
            
            # 2. Render response
            if result.get("is_cached"):
                st.toast("⚡ Served instantly from Semantic Cache!", icon="⚡")
            st.markdown(result["final_text"])
            
            # Render visual figures returned by the current turn
            if result.get("figures"):
                for fig in result["figures"]:
                    if isinstance(fig, go.Figure):
                        fig.update_layout(height=500, colorway=["#C4262E", "#A2A4A3", "#000000"])
                        st.plotly_chart(fig, use_container_width=True)
                
            # 4. Handle Excel export if DataFrames exist
            if result["dfs"]:
                excel_data = create_excel_buffer(result["dfs"])
                st.download_button(
                    label="📥 Download Raw Data to Excel",
                    data=excel_data,
                    file_name="agent_data_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_current"
                )
            
            # Render Execution Context (Reasoning Log)
            if result.get("run_log"):
                with st.expander("Agent Reasoning Log"):
                    for log in result["run_log"]:
                        st.text(log)

            # Update Streamlit session state cleanly in the UI layer
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["final_text"],
                "figures": result["figures"],
                "dfs": result["dfs"],
                "run_log": result["run_log"]
            })
                    
        except Exception as e:
            st.error(f"Agent Orchestration Error: {e}")

# ─── 7. SIDEBAR & METRICS ────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Token Usage Tracker")
    st.metric(label="Total Tokens", value=f"{st.session_state.total_tokens:,}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prompt", value=f"{st.session_state.prompt_tokens:,}")
    with col2:
        st.metric(label="Completion", value=f"{st.session_state.completion_tokens:,}")
    
    st.caption(f"Connected to: {MODEL}")
    st.divider()