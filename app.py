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
import streamlit as st

# --- 1. TIKTOKEN OFFLINE CACHE SETUP ---
# Tell tiktoken where to look for offline vocabulary files
cache_dir = "/tmp/tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# tiktoken names cached files after the SHA1 hash of their public download URL
tiktoken_url = (
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
)
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

# 1. Try importing torch. Catch BOTH missing package errors AND broken C++/CUDA library errors!
try:
    import torch
    print("PyTorch is already installed and working cleanly!")
except (ImportError, OSError, ValueError) as e:
    print(
        f"PyTorch missing or broken C++ CUDA dependencies ({type(e).__name__}). Starting clean CPU setup..."
    )

    wheel_name = "torch-2.4.0+cpu-cp311-cp311-linux_x86_64.whl"
    wheel_path = f"/tmp/{wheel_name}"

    # Only download from Workspace if it's not already sitting in /tmp/
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

    # Force-install the CPU wheel to obliterate the old GPU PyTorch cached in .venv!
    print("Force-installing PyTorch CPU into active virtual environment...")
    subprocess.check_call(
        ["pip", "install", wheel_path, "--no-deps", "--force-reinstall"]
    )

    # CRITICAL FIX: Flush broken/half-loaded torch modules from Python's RAM!
    print("Flushing module cache so Python sees the fresh CPU installation...")
    for mod in list(sys.modules.keys()):
        if mod.startswith("torch"):
            del sys.modules[mod]
    importlib.invalidate_caches()

# 2. Install dependent packages from your Git repo
print("Installing dependent packages from Git repo...")
for pkg in [
    "whls/accelerate-1.14.0-py3-none-any.whl",
    "whls/llmlingua-0.2.2-py3-none-any.whl",
]:
    subprocess.check_call(["pip", "install", pkg, "--no-deps"])

from agent.loop import run_agent_loop

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

st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    st.session_state.prompt_tokens = 0
    st.session_state.completion_tokens = 0

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
            
            # 3. Render any Plotly figures returned
            for fig in result["figures"]:
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
            
            # 5. Display Reasoning Log
            with st.expander("Agent Reasoning Log"):
                for log_item in result["run_log"]:
                    st.text(log_item)

            # 6. Update Streamlit session state cleanly in the UI layer
            st.session_state.total_tokens += 0 # Track your token metrics cleanly
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