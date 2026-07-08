import json
import traceback
from typing import Any, Dict, List, Tuple

import mlflow
import pandas as pd
import plotly.graph_objects as go

from agent.cache import agent_cache
from agent.memory import context_optimizer
from agent.schemas import DecomposedQuestions
from toolkit import TOOLS, TOOL_DISPATCHER
from toolkit.base import DATA_DICTIONARY, llm_call, MODEL, raw_client, track_tokens

# ─── 1. Context & Schema Helpers ─────────────────────────────────────────

def filter_schema(user_prompt: str) -> dict:
    """Filters the dictionary so the LLM isn't overwhelmed by irrelevant tables."""
    # For now, passing the whole combined dictionary. 
    # Can be upgraded to semantic search over table schemas later.
    return DATA_DICTIONARY


def decompose_question(user_prompt: str, schema: dict, history: List[dict], run_log: List[str]) -> List[str]:
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

def execute_tool_call(tool_call: dict, attempt: int, run_log: List[str]) -> Tuple[str, bool, List[Any]]:
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
            # Collect data structures (DataFrames, Models, Figures) for UI rendering
            for key in ["data", "model", "figure"]:
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

@mlflow.trace(name="run_agent_loop")
def run_agent_loop(user_prompt: str, chat_history: List[dict]) -> Dict[str, Any]:
    """
    The main orchestrator chaining the workflow together across multiple tools.
    Decoupled from UI: Takes history in, returns structured dictionary out.
    """
    run_log: List[str] = []
    current_turn_dfs: List[Any] = []

    with mlflow.start_run(run_name="Agent_Interaction"):
        mlflow.log_param("user_prompt", user_prompt)
        
        # ─── 0. SEMANTIC CACHE INTERCEPT ───
        cached_result = agent_cache.check_cache(user_prompt)
        if cached_result:
            run_log.append(f"⚡ Served from Semantic Cache. Matched Prompt: '{cached_result['matched_prompt']}' ({cached_result['similarity']*100:.1f}% similarity)")
            
            return {
                "final_text": cached_result["content"],
                "dfs": cached_result["dfs"],
                "figures": cached_result["figures"],
                "run_log": run_log,
                "is_cached": True,
                "cache_info": cached_result
            }
    
        # 1. Filter Context
        relevant_schema = filter_schema(user_prompt)
        
        # 2. Decompose Intent
        sub_questions = decompose_question(user_prompt, relevant_schema, chat_history, run_log)
        run_log.append(f"Sub-questions identified: {sub_questions}")
        
        # 3. Execute Tools per Question dynamically
        raw_outputs = []
        for sq_obj in sub_questions:
            # ─── SAFE TYPE EXTRACTION ───
            # Handles Pydantic models, dictionaries, and string fallbacks cleanly
            if isinstance(sq_obj, str):
                sq_text = sq_obj
                category_hint = "SPECIALIZED_ANALYTICS_AND_MODELING"
            elif isinstance(sq_obj, dict):
                sq_text = sq_obj.get("question", str(sq_obj))
                category_hint = sq_obj.get("target_category", "SPECIALIZED_ANALYTICS_AND_MODELING")
            else:
                # Standard Pydantic SubQuestion model
                sq_text = getattr(sq_obj, "question", str(sq_obj))
                category_hint = getattr(sq_obj, "target_category", "SPECIALIZED_ANALYTICS_AND_MODELING")

            prompt = f"""You are a routing assistant. Select the most appropriate tool to answer the sub-question.

            STRICT TOOL SELECTION HIERARCHY:
            Tier 1 - Specialized Analytics & Scenarios (HIGHEST PRIORITY):
            • If the question involves regression, correlations, forecasting (ARIMA), clustering (K-Means), PCA, Random Forest, or unit economics, you MUST use the specialized tool (e.g., `run_ols_regression_tool`, `run_arima_forecasting_tool`).
            • If the question involves "what-if" simulations, elasticity, or scenario planning, you MUST use `run_scenario_planning_tool`.

            Tier 2 - Visualizations:
            • If the user explicitly asks to plot, chart, or visualize data, use the appropriate `generate_*_tool`.

            Tier 3 - General SQL Execution (LOWEST PRIORITY / LAST RESORT):
            • ONLY use `execute_sql_query_tool` for simple data retrieval, basic filtering (WHERE), or standard mathematical aggregations (SUM, AVG, COUNT, GROUP BY).
            • NEGATIVE CONSTRAINT: DO NOT write complex SQL queries to attempt regressions, forecasting, or statistical modeling. If a Tier 1 tool can do it, writing SQL is strictly forbidden.

            The upstream planning agent flagged this question as needing a tool from the '{category_hint}' category.
            Use this EXACT schema for column names: {json.dumps(relevant_schema)}"""
            
            msgs = [{"role": "system", "content": prompt}]
            
            # Inject historical memory (last 4 turns)
            if chat_history:
                clean_history = [{"role": m["role"], "content": m.get("content", "")} for m in chat_history[-4:]]
                msgs.extend(clean_history)
                
            # Inject intra-turn memory from previous sub-questions in this same loop
            if raw_outputs:
                intra_turn_context = f"Context from previous sub-questions analyzed just now: {raw_outputs}"
                msgs.append({"role": "system", "content": intra_turn_context})
                
            msgs.append({"role": "user", "content": sq_text})

            max_retries = 3
            for attempt in range(max_retries):
                response = raw_client.chat.completions.create(
                    model=MODEL,
                    messages=msgs,
                    tools=TOOLS
                )
                track_tokens(response)
                assistant_msg = response.choices[0].message.model_dump(exclude_none=True)
                
                # If LLM decides no tool is needed, break immediately
                if not assistant_msg.get("tool_calls"):
                    raw_outputs.append(f"Sub-question: {sq_text}\nAnswer: {assistant_msg.get('content')}")
                    break
                
                msgs.append(assistant_msg)
                has_turn_error = False
                
                for tool_call in assistant_msg["tool_calls"]:
                    call_id = tool_call.get("id", "call_id")
                    tool_name = tool_call["function"]["name"]
                    
                    # Call our cleanly extracted execution engine
                    output_text, has_error, extracted_objects = execute_tool_call(tool_call, attempt, run_log)
                    
                    if has_error:
                        has_turn_error = True
                    else:
                        current_turn_dfs.extend(extracted_objects)
                        
                    # Feed the result (or formatted validation/execution error) back to the LLM
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": output_text
                    })
                    
                # If all tool calls in this attempt succeeded, we are done with this sub-question
                if not has_turn_error:
                    raw_outputs.append(f"Sub-question: {sq}\nTool Used: {tool_name}\nData: {output_text}")
                    break
                elif attempt == max_retries - 1:
                    raw_outputs.append(f"Sub-question: {sq}\nFailed after {max_retries} attempts.")

        # 4. Final Synthesis
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
        
        response = raw_client.chat.completions.create(
            model=MODEL,
            messages=final_msgs
        )
        track_tokens(response)
        final_text = response.choices[0].message.content
        
        # ─── Standardize Type Checking & Deduplicate Extractors ───
        # Safely split Plotly figures from tabular DataFrames/Models using isinstance
        turn_figures = [item for item in current_turn_dfs if isinstance(item, go.Figure)]
        turn_dfs = [item for item in current_turn_dfs if isinstance(item, pd.DataFrame)]

        # 5. Save to Semantic Cache
        agent_cache.save_to_cache(
            user_prompt=user_prompt,
            final_text=final_text,
            dfs=turn_dfs,
            figures=turn_figures
        )

        # 6. Return structured payload to UI
        return {
            "final_text": final_text,
            "dfs": turn_dfs,
            "figures": turn_figures,
            "run_log": run_log,
            "is_cached": False
        }