import json
import pandas as pd
import streamlit as st
from tools import raw_llm_call, TABLE_NAME, TOOLS, TOOL_DISPATCHER

st.set_page_config(page_title="Dataset Agent", page_icon="🤖", layout="wide")

# ─── Schema Context ──────────────────────────────────────────────
DATA_DICTIONARY = f"""
Table Name: {TABLE_NAME}
This table contains marketing data. Key columns include:
- temp_Id (string - unique customer identifier)
- Activation_Date (timestamp - date the ccustomer activated service)
- Beacon_Score_10pt (string - credit score range)
- Core_Package (string - package for the user (e.g., americas top 120))
- Tactic (string - marketing tactic the customer came from)
- WA Churn (float - estimated months on by customer)
- sac (float - subscriber aquisition cost)
- NC_ARPU (float - new customer average total revenue)
- NC_COGS (float - new customer average cogs)
- mcf (float - estimated monthly cash flow)
- npv (float - customer net present value)
there are more columns in the table as well.
"""

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
            msgs = [
                {"role": "system", "content": "You are a routing assistant. Select the appropriate tool to answer the user's question, or answer directly if no tool is needed."},
                {"role": "user", "content": sq}
            ]
            assistant_msg = raw_llm_call(msgs, tools=TOOLS)
            
            # ─── THE DYNAMIC DISPATCHER ───
            if assistant_msg.get("tool_calls"):
                for tool_call in assistant_msg["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    
                    st.session_state.run_log.append(f"Agent selected tool: {tool_name} with args: {args}")
                    
                    # Check if the tool exists in our dictionary
                    if tool_name in TOOL_DISPATCHER:
                        func = TOOL_DISPATCHER[tool_name]
                        
                        try:
                            # Special handling: SQL tool needs the dynamic schema for its auto-correct loop
                            if tool_name == "execute_sql_query_tool":
                                # Assuming the execute_sql_query_tool from the merged_app takes (question, schema)
                                result = func(user_intent=args.get("user_intent", sq), schema=relevant_schema)
                            else:
                                # Standard execution for OLS, ARIMA, and future tools using kwargs unpacking
                                result = func(**args)
                                
                            if isinstance(result, dict):
                                output_text = result.get("text", "")
                                
                                # Handle inner tool logs (like the SQL retry loop)
                                if result.get("logs"):
                                    st.session_state.run_log.extend(result["logs"])
                                    
                                # Safely inject models or dataframes into the UI state
                                if isinstance(result, dict):
                                    output_text = result.get("text", "")
                                
                                # Handle inner tool logs (like the SQL retry loop)
                                if result.get("logs"):
                                    st.session_state.run_log.extend(result["logs"])
                                    
                                # Safely inject models or dataframes into the UI state without evaluating truthiness
                                payload = result.get("data") if result.get("data") is not None else result.get("model")
                                
                                if payload is not None:
                                    st.session_state.current_turn_dfs.append(payload)
                            else:
                                # Fallback just in case a tool returns a raw string
                                output_text = str(result)

                            raw_outputs.append(f"Sub-question: {sq}\nTool Used: {tool_name}\nData: {output_text}")
                            
                        except Exception as e:
                            error_msg = f"Tool {tool_name} failed with error: {e}"
                            st.session_state.run_log.append(error_msg)
                            raw_outputs.append(error_msg)
                    else:
                        st.session_state.run_log.append(f"Warning: LLM hallucinated a non-existent tool '{tool_name}'")
            else:
                # The LLM decided it didn't need a tool for this specific sub-question
                raw_outputs.append(f"Sub-question: {sq}\nAnswer: {assistant_msg.get('content')}")

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