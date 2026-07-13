import json
import traceback
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.cache import agent_cache
from agent.memory import get_df_memory, get_context_optimizer
from agent.schemas import DecomposedQuestions
from toolkit import TOOLS, TOOL_DISPATCHER
from toolkit.base import DATA_DICTIONARY, _extract_text_content, llm_call, ModelConfig, raw_client, track_tokens

# ─── 1. Context & Schema Helpers ─────────────────────────────────────────

def filter_schema(user_prompt: str, run_log: List[str] = None) -> dict:
    """Uses the ROUTING_MODEL to intelligently filter the schema down to only relevant tables."""
    schema_summaries = {}
    for table_name, table_data in DATA_DICTIONARY.items():
        desc = table_data.get("description") or table_data.get("database_context", {}).get("table_metadata", {}).get("table_description", "")
        concepts = table_data.get("related_concepts", [])
        schema_summaries[table_name] = {"description": desc, "related_concepts": concepts}
        
    prompt = f"""You are a data architect. The user asked: '{user_prompt}'
                Here are the available tables and their related concepts:
                {json.dumps(schema_summaries, indent=2)}

                Return a strict JSON list of table names that are required to answer the user's question. 
                If the question is completely irrelevant, return an empty list.
                Example: {{"required_tables": ["\\"sandbox\\".\\"acquisition_data_v3\\""]}}"""

    msgs = [{"role": "user", "content": prompt}]
    
    class SchemaSelection(BaseModel):
        required_tables: List[str]
        
    try:
        parsed_result = llm_call(msgs, response_model=SchemaSelection, model_name=ModelConfig.ACTIVE_MODEL)
        filtered_dict = {t: DATA_DICTIONARY[t] for t in parsed_result.required_tables if t in DATA_DICTIONARY}
        
        if not filtered_dict:
            return DATA_DICTIONARY
            
        if run_log is not None:
            run_log.append(f"Schema filtering selected: {list(filtered_dict.keys())}")
            
        return filtered_dict
    except Exception as e:
        if run_log is not None:
            run_log.append(f"Schema filtering failed ({type(e).__name__}: {str(e)}). Defaulting to full schema.")
        return DATA_DICTIONARY


def decompose_question(user_prompt: str, schema: dict, history: List[dict], run_log: List[str], context_optimizer) -> List[str]:
    """Step 1: Breaks the user's prompt into specific data questions using chat history."""
    
    # Prune by exact token budget and compress dense historical context
    history_text = context_optimizer.format_history_for_prompt(history, max_tokens=50000)
    
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
    6. Use the 'Recent Conversation History' to resolve pronouns (e.g., "it", "that metric") or missing context (e.g., "what about next month?"). Ensure EVERY generated sub-question is entirely self-contained and explicitly mentions the required columns or context.
    7. If the user asks for a chart, graph, plot, scatterplot, bar chart, histogram, or line chart, you MUST explicitly include the exact visualization type (e.g., "generate a bar chart", "generate a histogram") in the generated sub-question so the downstream routing agent knows to trigger the specific visualization tool.
    8. If the user asks for a what-if analysis, scenario planning, or hypothetical simulation, you MUST explicitly include the phrase "run a what-if scenario" in the generated sub-question so the downstream routing agent knows to trigger the scenario planning tool.
    
    Respond STRICTLY with a JSON object containing a 'questions' key mapped to a list of strings.
    Example: {{"questions": ["What is the sum of NC_COGS in 2025?", "What is the average NPV?"]}}"""
    
    msgs = [{"role": "user", "content": prompt}]
    
    try:
        parsed_result = llm_call(msgs, response_model=DecomposedQuestions)
        return parsed_result.questions
    except Exception as e:
        # Improved error logging instead of silently swallowing the failure
        error_msg = f"Decomposition failed ({type(e).__name__}: {str(e)}). Falling back to raw prompt."
        run_log.append(error_msg)
        return [user_prompt]
    

# ─── 2. Extracted Tool Execution Engine ──────────────────────────────────

def execute_tool_call(tool_call: dict, attempt: int, run_log: List[str], df_memory) -> Tuple[str, bool, List[Any]]:
    """
    Handles parsing, Pydantic validation, and execution of a single tool call.
    Returns: (output_text, has_error, extracted_data_objects)
    """
    tool_name = tool_call["function"]["name"]
    call_id = tool_call.get("id", "call_id")
    extracted_objects = []
    
    # 1. Parse & Validate Arguments
    try:
        raw_args = json.loads(tool_call["function"]["arguments"])
        if tool_name in TOOL_DISPATCHER:
            _, validator = TOOL_DISPATCHER[tool_name]
            validated_args_model = validator(**raw_args)
            clean_args = validated_args_model.model_dump()
        else:
            clean_args = raw_args
    except Exception as e:
        error_msg = f"Validation Error on '{tool_name}': {str(e)}"
        run_log.append(error_msg)
        return error_msg, True, []

    run_log.append(f"Attempt {attempt+1}: Agent selected {tool_name} with args: {clean_args}")
    
    # 2. Execute Tool
    if tool_name not in TOOL_DISPATCHER:
        error_msg = f"Error: Tool '{tool_name}' does not exist in TOOL_DISPATCHER."
        run_log.append(error_msg)
        return error_msg, True, []

    func, _ = TOOL_DISPATCHER[tool_name]
    try:
        result = func(**clean_args)
        
        if isinstance(result, dict):
            if result.get("status") == "error":
                output_text = result.get("message", "Tool failed internally.")
                run_log.append(f"Tool '{tool_name}' returned error status: {output_text}")
                return output_text, True, []
            
            output_text = result.get("text", "Tool executed successfully.")
            
            # --- NEW MEMORY INJECTION ---
            if result.get("data") is not None and isinstance(result["data"], pd.DataFrame):
                # Save to registry and append the ID to the LLM's view
                df_id = df_memory.save_df(result["data"])
                output_text += f"\n[System Note: Data saved to Python memory with ID: {df_id}]"
                extracted_objects.append(result["data"])
                
            for key in ["model", "figure"]:
                if result.get(key) is not None:
                    extracted_objects.append(result[key])
        else:
            output_text = str(result)
            
        has_error = "Error" in output_text or "Exception" in output_text
        if has_error:
            run_log.append(f"Tool '{tool_name}' execution flagged issue: {output_text}")
            
        return output_text, has_error, extracted_objects

    except Exception as e:
        error_msg = f"Exception executing tool '{tool_name}': {str(e)}"
        run_log.append(error_msg)
        run_log.append(traceback.format_exc())
        return error_msg, True, []
    

# ─── 3. Main Agent Orchestrator ──────────────────────────────────────────

import time

@mlflow.trace(name="run_agent_loop")
def run_agent_loop(user_prompt: str, chat_history: List[dict]) -> Dict[str, Any]:
    """
    The main orchestrator chaining the workflow together across multiple tools.
    Decoupled from UI: Takes history in, returns structured dictionary out.
    """
    run_log: List[str] = []
    current_turn_dfs: List[Any] = []
    step_latencies: Dict[str, float] = {}

    # ─── Resolve session-scoped singletons ───────────────────────────────
    # Both objects are stored in st.session_state so every browser session
    # (i.e. every user) gets its own isolated instance.
    df_memory = get_df_memory()
    context_optimizer = get_context_optimizer()
    # Clear the DataFrame registry at the start of each new turn so IDs from
    # a previous conversation turn don't leak into this one.
    df_memory.clear()
    
    t_start_total = time.perf_counter()

    # Capture token counts at the start of the turn to compute per-turn MLflow metrics
    start_input_tokens = st.session_state.get("input_tokens", 0)
    start_output_tokens = st.session_state.get("output_tokens", 0)
    start_total_tokens = st.session_state.get("total_tokens", 0)

    with mlflow.start_run(run_name="Agent_Interaction"):
        mlflow.log_param("user_prompt", user_prompt)
        
        # ─── 0. SEMANTIC CACHE INTERCEPT ───
        t0 = time.perf_counter()
        cached_result = agent_cache.check_cache(user_prompt)
        step_latencies["Cache Check"] = round(time.perf_counter() - t0, 2)
        
        if cached_result:
            step_latencies["Total Execution"] = round(time.perf_counter() - t_start_total, 2)
            run_log.append(f"⚡ Served from Semantic Cache. Matched Prompt: '{cached_result['matched_prompt']}' ({cached_result['similarity']*100:.1f}% similarity)")
            
            mlflow.log_metrics({
                "latency_cache_check_sec": step_latencies["Cache Check"],
                "latency_total_sec": step_latencies["Total Execution"],
                "cache_hit": 1
            })
            
            return {
                "final_text": cached_result["content"],
                "dfs": cached_result["dfs"],
                "figures": cached_result["figures"],
                "run_log": run_log,
                "step_latencies": step_latencies,
                "is_cached": True,
                "cache_info": cached_result
            }
    
        mlflow.log_metric("cache_hit", 0)

        t0 = time.perf_counter()
        relevant_schema = filter_schema(user_prompt, run_log=run_log)
        sub_questions = decompose_question(user_prompt, relevant_schema, chat_history, run_log, context_optimizer)
        step_latencies["1. Decomposition"] = round(time.perf_counter() - t0, 2)
        run_log.append(f"Sub-questions identified: {sub_questions}")
        
        # ─── 2. TOOL ROUTING & EXECUTION ───
        t0_tools = time.perf_counter()
        raw_outputs = []
        for idx, sq_obj in enumerate(sub_questions):
            t0_sq = time.perf_counter()
            if isinstance(sq_obj, str):
                sq_text = sq_obj
                category_hint = "SPECIALIZED_ANALYTICS_AND_MODELING"
            elif isinstance(sq_obj, dict):
                sq_text = sq_obj.get("question", str(sq_obj))
                category_hint = sq_obj.get("target_category", "SPECIALIZED_ANALYTICS_AND_MODELING")
            else:
                sq_text = getattr(sq_obj, "question", str(sq_obj))
                category_hint = getattr(sq_obj, "target_category", "SPECIALIZED_ANALYTICS_AND_MODELING")

            prompt = f"""You are a routing assistant. Select the most appropriate tool to answer the sub-question.

            STRICT TOOL SELECTION HIERARCHY:
            Tier 1 - Specialized Analytics & Scenarios (HIGHEST PRIORITY):
            • If the question involves regression, correlations, forecasting (ARIMA), clustering (K-Means), PCA, Random Forest, or unit economics, you MUST use the specialized tool.
            • If the question involves "what-if" simulations, elasticity, or scenario planning, you MUST use `run_scenario_planning_tool`.

            Tier 2 - Visualizations:
            • If the user explicitly asks to plot, chart, or visualize data, use the appropriate `generate_*_tool`.

            Tier 3 - General SQL Execution (LOWEST PRIORITY / LAST RESORT):
            • ONLY use `execute_sql_query_tool` for simple data retrieval, basic filtering (WHERE), or standard mathematical aggregations.

            DATA MEMORY RULE:
            • If previous tool calls saved data to memory and returned an ID (e.g., `df_a1b2c3`), you MUST pass that exact ID into the `dataframe_id` argument of downstream charting or modeling tools instead of querying a base table.

            The upstream planning agent flagged this question as needing a tool from the '{category_hint}' category.
            Use this EXACT schema for column names: {json.dumps(relevant_schema)}"""
            
            msgs = [{"role": "system", "content": prompt}]
            
            if chat_history:
                clean_history = [{"role": m["role"], "content": m.get("content", "")} for m in chat_history[-4:]]
                msgs.extend(clean_history)
                
            if raw_outputs:
                intra_turn_context = f"Context from previous sub-questions analyzed just now: {raw_outputs}"
                msgs.append({"role": "system", "content": intra_turn_context})
                
            msgs.append({"role": "user", "content": sq_text})

            max_retries = 3
            for attempt in range(max_retries):
                response = raw_client.chat.completions.create(
                    model=ModelConfig.ACTIVE_MODEL,
                    messages=msgs,
                    tools=TOOLS
                )
                track_tokens(response)
                assistant_msg = response.choices[0].message.model_dump(exclude_none=True)
                
                if not assistant_msg.get("tool_calls"):
                    raw_outputs.append(f"Sub-question: {sq_text}\nAnswer: {_extract_text_content(response.choices[0].message)}")
                    break
                
                msgs.append(assistant_msg)
                has_turn_error = False
                
                for tool_call in assistant_msg["tool_calls"]:
                    call_id = tool_call.get("id", "call_id")
                    tool_name = tool_call["function"]["name"]
                    
                    output_text, has_error, extracted_objects = execute_tool_call(tool_call, attempt, run_log, df_memory)
                    
                    if has_error:
                        has_turn_error = True
                    else:
                        current_turn_dfs.extend(extracted_objects)
                        
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": output_text
                    })
                    
                if not has_turn_error:
                    raw_outputs.append(f"Sub-question: {sq_text}\nTool Used: {tool_name}\nData: {output_text}")
                    break
                elif attempt == max_retries - 1:
                    raw_outputs.append(f"Sub-question: {sq_text}\nFailed after {max_retries} attempts.")
            
            # <--- RECORD SUB-QUESTION LATENCY HERE (at the bottom of the loop)
            step_latencies[f"  ↳ Tool Exec {idx + 1}"] = round(time.perf_counter() - t0_sq, 2)

        # Record the total routing execution time
        step_latencies["2. Tool Routing & Execution"] = round(time.perf_counter() - t0_tools, 2)

        # ─── 3. FINAL SYNTHESIS ───
        t0 = time.perf_counter()
        raw_outputs_str = str(raw_outputs)
        if context_optimizer.count_tokens(raw_outputs_str) > 20000:
            raw_outputs_str = context_optimizer.compress_text(
                raw_outputs_str, 
                target_rate=0.5,
                context_instruction="Preserve all numerical values, metric names, and tool error messages."
            )

        synthesis_prompt = f"""You are a data insights assistant. 
        User's Original Prompt: {user_prompt}
        Raw Data Extracted across all tools: {raw_outputs_str}
        Relevant Schema: {json.dumps(relevant_schema)}
        
        Synthesize the raw data into a clear, business-friendly summary answering the original prompt.
        If any tools failed or returned errors in the raw data, briefly mention what analysis could not be completed and why, alongside the successful insights."""
        
        clean_messages = [{"role": m["role"], "content": m.get("content", "")} for m in chat_history]
        final_msgs = clean_messages + [{"role": "user", "content": synthesis_prompt}]
        
        try:
            response = raw_client.chat.completions.create(
                model=ModelConfig.ACTIVE_MODEL,
                messages=final_msgs
            )
            track_tokens(response)
            final_text = _extract_text_content(response.choices[0].message)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            run_log.append(f"Synthesis Model Failed: {e}")
            final_text = f"**⚠️ Synthesis Failed:** The synthesis model encountered an error.\n\n**Error:** {e}\n\n**Raw Extracted Data:**\n```text\n{raw_outputs_str[:2000]}...\n```"
        step_latencies["3. Final Synthesis"] = round(time.perf_counter() - t0, 2)
        
        step_latencies["Total Execution"] = round(time.perf_counter() - t_start_total, 2)
        
        turn_figures = [item for item in current_turn_dfs if isinstance(item, go.Figure)]
        turn_dfs = [item for item in current_turn_dfs if isinstance(item, pd.DataFrame)]

        # ─── 4. MLFLOW TELEMETRY LOGGING ───
        turn_input_tokens = st.session_state.get("input_tokens", 0) - start_input_tokens
        turn_output_tokens = st.session_state.get("output_tokens", 0) - start_output_tokens
        turn_total_tokens = st.session_state.get("total_tokens", 0) - start_total_tokens

        mlflow.log_metrics({
            "turn_input_tokens": turn_input_tokens,
            "turn_output_tokens": turn_output_tokens,
            "turn_total_tokens": turn_total_tokens,
            "latency_1_decomposition_sec": step_latencies.get("1. Decomposition", 0.0),
            "latency_2_tools_sec": step_latencies.get("2. Tool Routing & Execution", 0.0),
            "latency_3_synthesis_sec": step_latencies.get("3. Final Synthesis", 0.0),
            "latency_total_sec": step_latencies["Total Execution"]
        })

        agent_cache.save_to_cache(
            user_prompt=user_prompt,
            final_text=final_text,
            dfs=turn_dfs,
            figures=turn_figures
        )

        return {
            "final_text": final_text,
            "dfs": turn_dfs,
            "figures": turn_figures,
            "run_log": run_log,
            "step_latencies": step_latencies,
            "is_cached": False
        }