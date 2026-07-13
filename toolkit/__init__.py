import copy

from openai import pydantic_function_tool


# ─── Schema Compatibility Helpers ────────────────────────────────────────────
# Gemini (and some other non-OpenAI endpoints) reject JSON Schemas that contain
# $defs / $ref, which Pydantic v2 emits for any nested or Union type.
# _resolve_refs() recursively inlines all $ref pointers so the final schema is
# a plain, self-contained object that every model API accepts.

def _resolve_refs(schema: dict) -> dict:
    """
    Recursively resolves all $ref pointers in a JSON Schema dict by inlining
    the referenced $defs entries, then strips the $defs key entirely.
    Returns a deep-copied, fully-flattened schema.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def _inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]  # e.g. "#/$defs/ScenarioChange" → "ScenarioChange"
                resolved = copy.deepcopy(defs.get(ref_key, {}))
                node.clear()
                node.update(_inline(resolved))
            else:
                for k, v in node.items():
                    node[k] = _inline(v)
        elif isinstance(node, list):
            return [_inline(item) for item in node]
        return node

    return _inline(schema)


def _flatten_tool(tool: dict) -> dict:
    """
    Takes a pydantic_function_tool dict and returns a copy with its parameters
    schema fully resolved (no $defs / $ref).
    """
    tool = copy.deepcopy(tool)
    params = tool.get("function", {}).get("parameters")
    if params:
        tool["function"]["parameters"] = _resolve_refs(params)
    return tool


# 1. Import schemas explicitly with aliases to prevent name collisions
from agent.schemas import (
    execute_sql_query_tool as execute_sql_query_tool_Schema,
    run_ols_regression_tool as run_ols_regression_tool_Schema,
    run_arima_forecasting_tool as run_arima_forecasting_tool_Schema,
    run_random_forest_tool as run_random_forest_tool_Schema,
    run_pca_tool as run_pca_tool_Schema,
    run_kmeans_clustering_tool as run_kmeans_clustering_tool_Schema,
    generate_scatterplot_tool as generate_scatterplot_tool_Schema,
    generate_barchart_tool as generate_barchart_tool_Schema,
    generate_histogram_tool as generate_histogram_tool_Schema,
    generate_linechart_tool as generate_linechart_tool_Schema,
    calculate_unit_economics_tool as calculate_unit_economics_tool_Schema,
    compare_monthly_metrics_tool as compare_monthly_metrics_tool_Schema,
    run_scenario_planning_tool as run_scenario_planning_tool_Schema,
    execute_python_tool as execute_python_tool_Schema,
    run_neural_network_tool as run_neural_network_tool_Schema,
    run_optimization_tool as run_optimization_tool_Schema
)

# 2. Import actual tool execution functions explicitly
from .analytics import (
    execute_sql_query_tool,
    run_ols_regression_tool,
    run_arima_forecasting_tool,
    run_random_forest_tool,
    run_pca_tool,
    run_kmeans_clustering_tool,
    calculate_unit_economics_tool,
    run_scenario_planning_tool,
    execute_python_tool,
    run_neural_network_tool,
    run_optimization_tool
)
from .visuals import (
    generate_scatterplot_tool,
    generate_barchart_tool,
    generate_histogram_tool,
    generate_linechart_tool,
    compare_monthly_metrics_tool,
)

# 3. Auto-generate OpenAI tool definitions directly from Pydantic schemas
TOOL_SCHEMAS = [
    execute_sql_query_tool_Schema,
    run_ols_regression_tool_Schema,
    run_arima_forecasting_tool_Schema,
    run_random_forest_tool_Schema,
    run_pca_tool_Schema,
    run_kmeans_clustering_tool_Schema,
    generate_scatterplot_tool_Schema,
    generate_barchart_tool_Schema,
    generate_histogram_tool_Schema,
    generate_linechart_tool_Schema,
    calculate_unit_economics_tool_Schema,
    compare_monthly_metrics_tool_Schema,
    run_scenario_planning_tool_Schema,
    execute_python_tool_Schema,
    run_neural_network_tool_Schema,
    run_optimization_tool_Schema
]

TOOLS = [_flatten_tool(pydantic_function_tool(schema)) for schema in TOOL_SCHEMAS]

# ─── Category → Tool Name Mapping ────────────────────────────────────────────
# Maps each SubQuestion.target_category value to the exact set of tool names the
# router is allowed to choose from. loop.py uses this to filter TOOLS before
# passing them to the model, preventing the router from ever selecting an
# out-of-category tool regardless of prompt wording.
CATEGORY_TOOLS: dict[str, list[str]] = {
    "SQL_RETRIEVAL": [
        "execute_sql_query_tool",
    ],
    "UNIT_ECONOMICS": [
        "calculate_unit_economics_tool",
    ],
    "STATISTICAL_MODELING": [
        "run_ols_regression_tool",
        "run_pca_tool",
        "run_kmeans_clustering_tool",
    ],
    "ML_MODELING": [
        "run_random_forest_tool",
        "run_neural_network_tool",
        "run_optimization_tool",
    ],
    "FORECASTING": [
        "run_arima_forecasting_tool",
    ],
    "SCENARIO_SIMULATION": [
        "run_scenario_planning_tool",
    ],
    "VISUALIZATION": [
        "generate_barchart_tool",
        "generate_linechart_tool",
        "generate_scatterplot_tool",
        "generate_histogram_tool",
        "compare_monthly_metrics_tool",
    ],
    "CUSTOM_PYTHON": [
        "execute_python_tool",
    ],
}

# 4. Centralized routing map: maps tool names to (execution_function, pydantic_validator) tuples
TOOL_DISPATCHER = {
    "execute_sql_query_tool": (execute_sql_query_tool, execute_sql_query_tool_Schema),
    "run_ols_regression_tool": (run_ols_regression_tool, run_ols_regression_tool_Schema),
    "run_arima_forecasting_tool": (run_arima_forecasting_tool, run_arima_forecasting_tool_Schema),
    "run_random_forest_tool": (run_random_forest_tool, run_random_forest_tool_Schema),
    "run_pca_tool": (run_pca_tool, run_pca_tool_Schema),
    "run_kmeans_clustering_tool": (run_kmeans_clustering_tool, run_kmeans_clustering_tool_Schema),
    "generate_scatterplot_tool": (generate_scatterplot_tool, generate_scatterplot_tool_Schema),
    "generate_barchart_tool": (generate_barchart_tool, generate_barchart_tool_Schema),
    "generate_histogram_tool": (generate_histogram_tool, generate_histogram_tool_Schema),
    "generate_linechart_tool": (generate_linechart_tool, generate_linechart_tool_Schema),
    "calculate_unit_economics_tool": (calculate_unit_economics_tool, calculate_unit_economics_tool_Schema),
    "compare_monthly_metrics_tool": (compare_monthly_metrics_tool, compare_monthly_metrics_tool_Schema),
    "run_scenario_planning_tool": (run_scenario_planning_tool, run_scenario_planning_tool_Schema),
    "execute_python_tool": (execute_python_tool, execute_python_tool_Schema),
    "run_neural_network_tool": (run_neural_network_tool, run_neural_network_tool_Schema),
    "run_optimization_tool": (run_optimization_tool, run_optimization_tool_Schema)
}