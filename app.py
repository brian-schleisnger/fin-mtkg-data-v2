import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

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
from agent.cache import agent_cache
from toolkit.base import AVAILABLE_MODELS, ModelConfig, set_active_model

# ─── 3. GLOBAL CONFIGURATION & UI HELPERS ────────────────────────────────
# Set MLflow experiment once globally so it doesn't fire API calls on every chat turn
mlflow.set_experiment("/Workspace/Users/brian.schlesinger@dish.com")

def load_css():
    """Reads custom CSS from style.css co-located with app.py and injects it."""
    css_path = Path(__file__).parent / "style.css"
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css not found. Proceeding with default styling.")

def create_excel_buffer(data_list: list) -> bytes:
    """Extracts DataFrames from the agent's output, strips timezones, and writes them to an Excel buffer."""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sheet_counter = 1
        has_data = False
        
        for item in data_list:
            if isinstance(item, pd.DataFrame):
                df = item.copy()
                
                # 1. Strip timezone from standard pandas 'datetimetz' columns
                for col in df.select_dtypes(include=['datetimetz']).columns:
                    df[col] = df[col].dt.tz_localize(None)
                
                # 2. Fallback check for any generic datetime dtypes that still hold tz metadata
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]) and getattr(df[col].dt, 'tz', None) is not None:
                        df[col] = df[col].dt.tz_localize(None)
                        
                # Write each cleaned DataFrame to its own tab
                df.to_excel(writer, index=False, sheet_name=f"Result_{sheet_counter}")
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
if "input_tokens" not in st.session_state:
    st.session_state.input_tokens = 0
if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0
if "estimated_cost" not in st.session_state:
    st.session_state.estimated_cost = 0.0
if "last_step_latencies" not in st.session_state:
    st.session_state.last_step_latencies = {}
if "rerun_prompt" not in st.session_state:
    st.session_state.rerun_prompt = None
if "rerun_msg_index" not in st.session_state:
    st.session_state.rerun_msg_index = None

# Apply CSS after session state so any st.warning() from load_css renders correctly
load_css()


# ─── 5. SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Dataset Agent")
    st.divider()

    # ── Model Selection ──
    model_options = list(AVAILABLE_MODELS.keys())
    selected_model = st.selectbox(
        "Active Model",
        options=model_options if model_options else ["No models configured"],
        help="Select the model used for all reasoning steps.",
        disabled=not model_options,
    )
    if model_options:
        set_active_model(selected_model)

    st.divider()

    # ── Estimated Cost ──
    # Displayed directly from session state — cost is accumulated in track_tokens()
    # at the moment each API call completes, so it reflects the actual model rates
    # used, not the currently selected model.
    st.metric(label="💰 Est. Session Cost", value=f"${st.session_state.estimated_cost:.4f}")

    st.divider()

    # ── Step Latencies ──
    st.markdown("#### ⏱️ Last Turn Latency")
    latencies = st.session_state.get("last_step_latencies", {})
    if latencies:
        total_time = latencies.get("Total Execution", 0.0)
        st.metric(label="Total", value=f"{total_time:.2f}s")
        for step_name, duration in latencies.items():
            if step_name != "Total Execution":
                st.markdown(
                    f"<div style='font-size:0.85em; margin-bottom:2px;'>"
                    f"<b>{step_name}</b>: {duration:.2f}s</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No query executed yet.")

    st.divider()

    # ── Clear Chat ──
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.input_tokens = 0
        st.session_state.output_tokens = 0
        st.session_state.estimated_cost = 0.0
        st.session_state.last_step_latencies = {}
        st.rerun()


# ─── 6. MAIN AREA: WELCOME SCREEN (ALWAYS VISIBLE) ──────────────────────
st.markdown("## Marketing Intelligence Agent")
st.markdown("Connected data sources:")

# Wrapper div so card CSS only applies here, not to every stContainer in the app
st.markdown('<div class="welcome-cards">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 🚀 Marketing <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Monthly Marketing Spend**")
        st.caption("Spend by tactic & sub-tactic (01/2021 - 05/2026)")

    with st.container(border=True):
        st.markdown("#### 💲 Sales <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Daily Sales Data**")
        st.caption("Calls / sales / activations / BRMs (01/2019 - 06/2026)")

with col2:
    with st.container(border=True):
        st.markdown("#### 👥 Subscriber <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Monthly Subscriber Counts**")
        st.caption("Activations by channel, deactivations, churn (01/2018 - 06/2026)")

    with st.container(border=True):
        st.markdown("#### 📊 Financials <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Dish P&L**")
        st.caption("Monthly BU P&L statement data (01/2018 - 06/2026)")

with col3:
    with st.container(border=True):
        st.markdown("#### 🔑 Acquisition <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Customer Activation Data**")
        st.caption("ARPU, COGS, SAC, churn, NPV, MCF (10/2018 - 03/2026)")

st.markdown("</div>", unsafe_allow_html=True)


# ─── 7. CHAT HISTORY RENDERING ───────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🌐"):
            st.markdown(msg["content"])

            if msg.get("figures"):
                for j, fig in enumerate(msg["figures"]):
                    if isinstance(fig, go.Figure):
                        fig.update_layout(
                            height=400, 
                            colorway=["#105e62", "#b2d8d8", "#000000"], 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"fig_{i}_{j}")
                    
            if msg["role"] == "assistant" and (msg.get("dfs") or msg.get("run_log")):
                st.markdown("---")
                act_col1, act_col2, act_col3 = st.columns([1, 2, 1])
                
                with act_col1:
                    if msg.get("dfs"):
                        try:
                            excel_data = create_excel_buffer(msg["dfs"])
                            st.download_button(
                                label="📥 Export Data",
                                data=excel_data,
                                file_name=f"agent_data_export_{i}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"download_hist_{i}",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Excel export unavailable: {e}")
                
                with act_col2:
                    if msg.get("run_log"):
                        with st.expander("🧠 View Agent Execution Trace", expanded=False):
                            for step_num, log in enumerate(msg["run_log"], 1):
                                st.markdown(f"**Step {step_num}:**")
                                st.code(log, language="text", wrap_lines=True)

                with act_col3:
                    if i > 0 and st.session_state.messages[i - 1]["role"] == "user":
                        if st.button("🔄 Re-run", key=f"rerun_{i}", use_container_width=True):
                            st.session_state.rerun_prompt = st.session_state.messages[i - 1]["content"]
                            st.session_state.rerun_msg_index = i
                            st.rerun()

# ─── 8. RE-RUN HANDLER ───────────────────────────────────────────────────
if st.session_state.rerun_prompt is not None:
    rerun_prompt = st.session_state.rerun_prompt
    rerun_index = st.session_state.rerun_msg_index

    st.session_state.rerun_prompt = None
    st.session_state.rerun_msg_index = None

    from agent.cache import agent_cache as _cache
    _cache.delete_from_cache(rerun_prompt)

    history_before = st.session_state.messages[: rerun_index - 1]

    with st.chat_message("assistant", avatar="🌐"):
        try:
            with st.spinner(f"Re-running with `{ModelConfig.ACTIVE_MODEL}`..."):
                result = run_agent_loop(rerun_prompt, history_before)

            st.session_state.last_step_latencies = result.get("step_latencies", {})

            st.session_state.messages[rerun_index] = {
                "role": "assistant",
                "content": result["final_text"],
                "figures": result["figures"],
                "dfs": result["dfs"],
                "run_log": result["run_log"]
            }
            st.rerun()

        except Exception as e:
            st.error(f"Re-run Error: {e}")
            with st.expander("Show Traceback"):
                st.code(traceback.format_exc(), language="python")

# ─── 9. CHAT INPUT & EXECUTION ───────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the marketing data..."):
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌐"):
        try:
            with st.spinner("Analyzing..."):
                result = run_agent_loop(prompt, st.session_state.messages)
            
            st.session_state.last_step_latencies = result.get("step_latencies", {})
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["final_text"],
                "figures": result["figures"],
                "dfs": result["dfs"],
                "run_log": result["run_log"]
            })
            
            st.rerun()
                    
        except Exception as e:
            st.error(f"Agent Orchestration Error: {e}")
            with st.expander("Show Traceback"):
                st.code(traceback.format_exc(), language="python")
