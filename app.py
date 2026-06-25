import json
from pathlib import Path

import pandas as pd
import streamlit as st

from tools import raw_llm_call, TABLE_NAME, TOOLS, TOOL_DISPATCHER


st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

# ─── Schema Context ──────────────────────────────────────────────
# 1. Define the path to your new JSON file
DICT_FILE_PATH = Path(__file__).parent.resolve() / "acquisition_data_dictionary.json"

# 2. Load the JSON and override the table name
try:
    with DICT_FILE_PATH.open("r", encoding="utf-8") as f:
        DATA_DICTIONARY = json.load(f)
        
        # Override the table_name in the JSON with the SQL-safe TABLE_NAME from tools.py
        DATA_DICTIONARY["table_name"] = TABLE_NAME
except Exception as e:
    st.error(f"Error loading data dictionary: {e}")
    DATA_DICTIONARY = {"error": "Could not load schema."}

# ─── RAG & Orchestration Helpers ─────────────────────────────────
def filter_schema(user_prompt: str) -> dict:
    """Filters the dictionary so the LLM isn't overwhelmed by irrelevant tables."""
    # In a larger app, use keyword matching here. For now, we return the whole dict.
    return DATA_DICTIONARY

def decompose_question(user_prompt: str, schema: dict) -> list:
    """Step 1: Breaks the user's prompt into specific data questions."""
    prompt = f"""You are a data strategist. Break the user's broad request down into specific, actionable data queries.
    Available Data Schema: {json.dumps(schema)}
    User Request: {user_prompt}

    RULES:
    1. ONLY generate data queries if the user is explicitly asking for data analysis, metrics, or insights.
    2. Do not generate more than five queries. 
    3. CRITICAL: Do NOT break down statistical models (like Regression, Random Forest, or ARIMA) into separate questions for their sub-metrics (e.g., coefficients, R-squared, p-values, residuals). Group all requirements for a single model into ONE unified question
    4. If the user is asking a general question, greeting you, or asking about your capabilities, return the user's exact prompt as a single item and do NOT generate data queries.
    
    Respond STRICTLY with a JSON object containing a 'questions' key mapped to a list of strings.
    Example: {{"questions": ["What is the sum of NC_COGS in 2025?", "What is the average NPV?"]}}"""
    
    msgs = [{"role": "user", "content": prompt}]
    response = raw_llm_call(msgs, require_json=True)
    
    try:
        parsed = json.loads(response.get("content", "{}"))
        return parsed.get("questions", [user_prompt]) # Fallback to original prompt if parsing fails
    except json.JSONDecodeError:
        return [user_prompt]
    
# ─── Agent Orchestration Loop ────────────────────────────────────
def run_agent_loop(user_prompt: str):
    """The main orchestrator chaining the workflow together across multiple tools."""
    st.session_state.run_log = []
    st.session_state.current_turn_dfs = []
    
    # 1. Filter Context
    relevant_schema = filter_schema(user_prompt)
    
    # 2. Decompose Intent
    with st.spinner("Decomposing question..."):
        sub_questions = decompose_question(user_prompt, relevant_schema)
        st.session_state.run_log.append(f"Sub-questions identified: {sub_questions}")
    
    # 3. Execute Tools per Question dynamically
    raw_outputs = []
    for sq in sub_questions:
        with st.spinner(f"Analyzing: '{sq}'..."):
            
            # Pass the full TOOLS array so the LLM can pick the right one
            prompt = f"""You are a routing assistant. Select the appropriate tool, or answer directly if no tool is needed. 
                        Use this EXACT schema for column names: {json.dumps(relevant_schema)}"""
            msgs = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sq}
            ]

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
                    args = json.loads(tool_call["function"]["arguments"])
                    call_id = tool_call.get("id", "call_id") # Needed for tool tracking
                    
                    st.session_state.run_log.append(f"Attempt {attempt+1}: Agent selected {tool_name} with args: {args}")
                    
                    # Execute the tool
                    if tool_name in TOOL_DISPATCHER:
                        func = TOOL_DISPATCHER[tool_name]
                        try:
                            # Standard execution for ALL tools (no more special SQL handling needed)
                            result = func(**args)
                            
                            if isinstance(result, dict):
                                output_text = result.get("text", "")
                                payload = result.get("data") if result.get("data") is not None else result.get("model")
                                
                                # Only save payload to UI if it wasn't an error
                                if payload is not None and "Error" not in output_text:
                                    st.session_state.current_turn_dfs.append(payload)
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
        
        final_msgs = st.session_state.messages + [{"role": "user", "content": synthesis_prompt}]
        final_response = raw_llm_call(final_msgs)
        
        # Save to memory
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append(final_response)
        
        return final_response.get("content", "")

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
            st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the marketing data..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Trigger the agentic loop
            final_response = run_agent_loop(prompt)
            st.markdown(final_response)
            
            # Display execution context if the database was queried during this loop
            with st.expander("Agent Reasoning Log"):
                for log in st.session_state.run_log:
                    st.text(log)
                    
            if st.session_state.current_turn_dfs:
                with st.expander("View Raw Data Returned"):
                    for i, item in enumerate(st.session_state.current_turn_dfs):
                        st.write(f"**Result {i+1}**")
                        # Check if it's a pandas dataframe before trying to render it as one
                        if isinstance(item, pd.DataFrame):
                            st.dataframe(item)
                        # Check if it's a statsmodels object (they have a summary method)
                        elif hasattr(item, "summary"):
                            st.text(item.summary().as_text())
                        # Fallback for Scikit-Learn models or unknown objects
                        else:
                            st.write(str(item))
                    
        except Exception as e:
            st.error(f"Agent Orchestration Error: {e}")