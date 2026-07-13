from openai import pydantic_function_tool

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

TOOLS = [pydantic_function_tool(schema) for schema in TOOL_SCHEMAS]

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