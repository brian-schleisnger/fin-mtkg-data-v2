import io
import json
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

from tools import MODEL, raw_llm_call, TOOLS, TOOL_DISPATCHER


st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    st.session_state.prompt_tokens = 0
    st.session_state.completion_tokens = 0

# ─── NEW: Sidebar Token UI Tracker ───
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

# ─── Schema Context ──────────────────────────────────────────────
# 1. Define your tables and their corresponding dictionary files
SCHEMA_CONFIG = {
    '"sandbox"."acquisition_data_v3"': "acquisition_data_dictionary.json",
    '"sandbox"."dbs_marketing_spend_sync"': "marketing_spend_dictionary.json"
}

DATA_DICTIONARY = {}
BASE_DIR = Path(__file__).parent.resolve()

# 2. Load all JSON schemas into a unified dictionary
for table_name, file_name in SCHEMA_CONFIG.items():
    dict_path = BASE_DIR / file_name
    try:
        with dict_path.open("r", encoding="utf-8") as f:
            schema_data = json.load(f)
            schema_data["table_name"] = table_name # Explicitly inject the SQL-safe table name
            DATA_DICTIONARY[table_name] = schema_data
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")

# ─── RAG & Orchestration Helpers ─────────────────────────────────
def filter_schema(user_prompt: str) -> dict:
    """Filters the dictionary so the LLM isn't overwhelmed by irrelevant tables."""
    # In a larger app with dozens of tables, use semantic search or keyword matching.
    # For two tables, passing the whole combined dictionary is fine.
    return DATA_DICTIONARY

def decompose_question(user_prompt: str, schema: dict, history: list) -> list:
    """Step 1: Breaks the user's prompt into specific data questions using chat history."""
    
    # Format the last few turns of history to give the LLM context without blowing up tokens
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:]]) if history else "No previous history."
    
    prompt = f"""You are a data strategist. Break the user's broad request down into specific, actionable data queries.
    
    Available Data Schema: {json.dumps(schema)}
    
    Recent Conversation History:
    {history_text}
    
    User Request: {user_prompt}

    RULES:
    1. ONLY generate data queries if the user is explicitly asking for data analysis, metrics, or insights.
    2. Do not generate more than five queries.
    3. If the user asks for a specific tool in their prompt (e.g., "generate a bar chart" or "run a regression"), ensure that the sub-questions explicitly mention that tool and its required inputs.
    4. Do NOT break down statistical models (like Regression, Random Forest, or ARIMA) into separate questions for their sub-metrics (e.g., coefficients, R-squared, p-values, residuals). Group all requirements for a single model into ONE unified question.
    5. If the user is asking a general question, greeting you, or asking about your capabilities, return the user's exact prompt as a single item and do NOT generate data queries.
    6. CRITICAL MEMORY RULE: Use the 'Recent Conversation History' to resolve pronouns (e.g., "it", "that metric") or missing context (e.g., "what about next month?"). Ensure EVERY generated sub-question is entirely self-contained and explicitly mentions the required columns or context.
    7. CRITICAL VISUALIZATION RULE: If the user asks for a chart, graph, plot, scatterplot, bar chart, histogram, or line chart, you MUST explicitly include the exact visualization type (e.g., "generate a bar chart", "generate a histogram") in the generated sub-question so the downstream routing agent knows to trigger the specific visualization tool.
    
    Respond STRICTLY with a JSON object containing a 'questions' key mapped to a list of strings.
    Example: {{"questions": ["What is the sum of NC_COGS in 2025?", "What is the average NPV?"]}}"""
    
    msgs = [{"role": "user", "content": prompt}]
    response = raw_llm_call(msgs, require_json=True)
    
    try:
        parsed = json.loads(response.get("content", "{}"))
        return parsed.get("questions", [user_prompt]) 
    except json.JSONDecodeError:
        return [user_prompt]
    
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
    
# ─── Agent Orchestration Loop ────────────────────────────────────
def run_agent_loop(user_prompt: str):
    """The main orchestrator chaining the workflow together across multiple tools."""
    st.session_state.run_log = []
    st.session_state.current_turn_dfs = []

    mlflow.set_experiment("/Workspace/Users/brian.schlesinger@dish.com") # Adjust path to your workspace

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="Agent_Interaction") as run:
        mlflow.log_param("user_prompt", user_prompt)
        
        # Capture the starting tokens to calculate how many this specific run costs
        start_tokens = st.session_state.total_tokens
    
    # 1. Filter Context
    relevant_schema = filter_schema(user_prompt)
    
    # 2. Decompose Intent (NOW PASSING MEMORY)
    with st.spinner("Decomposing question..."):
        sub_questions = decompose_question(user_prompt, relevant_schema, st.session_state.messages)
        st.session_state.run_log.append(f"Sub-questions identified: {sub_questions}")
    
    # 3. Execute Tools per Question dynamically
    raw_outputs = []
    for sq in sub_questions:
        with st.spinner(f"Analyzing: '{sq}'..."):
            
            prompt = f"""You are a routing assistant. Select the appropriate tool, or answer directly if no tool is needed. 
                        Use this EXACT schema for column names: {json.dumps(relevant_schema)}"""
            
            msgs = [{"role": "system", "content": prompt}]
            
            # INJECT HISTORICAL MEMORY: Give the tool router the last few turns
            if st.session_state.messages:
                clean_history = [{"role": m["role"], "content": m.get("content", "")} for m in st.session_state.messages[-4:]]
                msgs.extend(clean_history)
                
            # INJECT INTRA-TURN MEMORY: If previous sub-questions in this loop found data, let the router see it
            if raw_outputs:
                intra_turn_context = f"Context from previous sub-questions analyzed just now: {raw_outputs}"
                msgs.append({"role": "system", "content": intra_turn_context})
                
            # Append the actual sub-question to trigger the tool
            msgs.append({"role": "user", "content": sq})

            max_retries = 3
            for attempt in range(max_retries):
                assistant_msg = raw_llm_call(msgs, tools=TOOLS)
            
                # ─── THE DYNAMIC DISPATCHER ───
                if not assistant_msg.get("tool_calls"):
                    raw_outputs.append(f"Sub-question: {sq}\nAnswer: {assistant_msg.get('content')}")
                    break
                
                # Append the LLM's tool call request to the conversation history
                msgs.append(assistant_msg)
                has_error = False
                
                for tool_call in assistant_msg["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    call_id = tool_call.get("id", "call_id") # Move this up so we can use it in the except block
                    
                    # Safely attempt to parse the arguments
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError as e:
                        error_msg = f"Error: Invalid JSON format for arguments. {str(e)}"
                        st.session_state.run_log.append(f"Attempt {attempt+1}: {error_msg}")
                        
                        # Feed the error back to the LLM so it can learn and retry
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": error_msg
                        })
                        has_error = True
                        continue # Skip the rest of this iteration and let the agent retry
                    
                    st.session_state.run_log.append(f"Attempt {attempt+1}: Agent selected {tool_name} with args: {args}")
                    
                    # Execute the tool
                    if tool_name in TOOL_DISPATCHER:
                        func = TOOL_DISPATCHER[tool_name]
                        try:
                            # Standard execution for ALL tools (no more special SQL handling needed)
                            result = func(**args)
                            
                            if isinstance(result, dict):
                                output_text = result.get("text", "")
                                
                                # If it's not an error, append returned objects to the UI payload list
                                if "Error" not in output_text:
                                    if result.get("data") is not None:
                                        st.session_state.current_turn_dfs.append(result["data"])
                                    if result.get("model") is not None:
                                        st.session_state.current_turn_dfs.append(result["model"])
                                    if result.get("figure") is not None:
                                        st.session_state.current_turn_dfs.append(result["figure"])
                            else:
                                output_text = str(result)
                        except Exception as e:
                            output_text = f"Error executing tool: {e}"
                    else:
                        output_text = f"Error: Tool '{tool_name}' does not exist."
                        
                    # Check if the tool failed to trigger the retry
                    if "Error" in output_text or "Exception" in output_text:
                        has_error = True
                        st.session_state.run_log.append(f"Tool failed: {output_text}")
                        
                    # Feed the result (or the error string) back to the LLM so it can learn and retry
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": output_text
                    })
                    
                # If everything succeeded, break the retry loop
                if not has_error:
                    raw_outputs.append(f"Sub-question: {sq}\nTool Used: {tool_name}\nData: {output_text}")
                    break
                elif attempt == max_retries - 1:
                    raw_outputs.append(f"Sub-question: {sq}\nFailed after {max_retries} attempts.")

    # 4. Final Synthesis
    with st.spinner("Synthesizing final answer..."):
        synthesis_prompt = f"""You are a data insights assistant. 
        User's Original Prompt: {user_prompt}
        Raw Data Extracted across all tools: {raw_outputs}
        
        Synthesize the raw data into a clear, business-friendly summary answering the original prompt.
        If any tools failed or returned errors in the raw data, briefly mention what analysis could not be completed and why, alongside the successful insights."""
        
        # Safely strip figures from history before sending to the LLM API
        clean_messages = [{"role": m["role"], "content": m.get("content", "")} for m in st.session_state.messages]
        final_msgs = clean_messages + [{"role": "user", "content": synthesis_prompt}]
        
        final_response = raw_llm_call(final_msgs)
        final_text = final_response.get("content", "")
        
        # Save user prompt to memory
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # NEW: Extract figures and save them in the assistant's message history
        turn_figures = [item for item in st.session_state.current_turn_dfs if type(item).__name__ == "Figure"]
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_text,
            "figures": turn_figures # Attach the figures to the state
        })

        tokens_used_this_turn = st.session_state.total_tokens - start_tokens
        mlflow.log_metric("tokens_this_turn", tokens_used_this_turn)
        mlflow.log_metric("session_total_tokens", st.session_state.total_tokens)
        
        return final_text

# ─── UI ───────────────────────────────────────────────────────────
st.title("Acquisition Finance Agent - Phase 1")
st.caption("Ask questions, and the agent will autonomously decide when to query the data.")

# Initialize System Prompt & Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history (filtering out system/tool messages for a clean UI)
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            if msg.get("content"):
                st.markdown(msg["content"])

            # Render historical figures with fixed height and theme colors
            if msg.get("figures"):
                for fig in msg["figures"]:
                    fig.update_layout(
                        height=500, # Forces a reasonable ~16:9 aspect ratio on desktop
                        colorway=["#C4262E", "#A2A4A3", "#000000"] # Uses your config.toml colors
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Handle new user input
if prompt := st.chat_input("Ask a question about the marketing data..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Trigger the agentic loop
            final_response = run_agent_loop(prompt)
            st.markdown(final_response)
            
            # ─── NEW UI LAYOUT ───
            # Show Visualizations and Download Button directly below the LLM output
            if st.session_state.current_turn_dfs:
                
                for item in st.session_state.current_turn_dfs:
                    if type(item).__name__ == "Figure":
                        item.update_layout(
                            height=500,
                            colorway=["#C4262E", "#A2A4A3", "#000000"] 
                        )
                        st.plotly_chart(item, use_container_width=True)
                
                excel_data = create_excel_buffer(st.session_state.current_turn_dfs)
                st.download_button(
                    label="📥 Download Raw Data to Excel",
                    data=excel_data,
                    file_name="agent_data_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Display execution context (Agent Reasoning Log) at the very bottom
            with st.expander("Agent Reasoning Log"):
                for log in st.session_state.run_log:
                    st.text(log)
                    
        except Exception as e:
            st.error(f"Agent Orchestration Error: {e}")

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