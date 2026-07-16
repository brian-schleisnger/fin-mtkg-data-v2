# Repository Docstrings

Generated via static analysis.

---

# File: .\app.py
### Function: `bootstrap_environment`
```text
Runs offline caching, PyTorch CPU workarounds, and wheel installations exactly ONCE 
per server lifecycle, preventing Streamlit from re-running them on every UI interaction.
```
### Function: `create_excel_buffer`
```text
Extracts DataFrames from the agent's output, strips timezones, and writes them to an Excel buffer.
```

---

# File: .\agent\cache.py
### Class: `SemanticCache`
```text
A lightweight, persistent semantic cache tailored for complex agent outputs
(Text + Pandas DataFrames + Plotly Figures).
```
### Function: `_init_db`
```text
Initializes the SQLite database for persistent cache storage.
```
### Function: `_get_embedding`
```text
Fetches a dense vector embedding for the given text from the Databricks
embedding endpoint. Always uses a freshly fetched auth token.
Input is lowercased and stripped before embedding for consistent similarity scoring.
```
### Function: `_cosine_similarity`
```text
Calculates cosine similarity between two vectors.
```
### Function: `check_cache`
```text
Checks if a semantically equivalent prompt exists in the cache.
Returns the cached dictionary if similarity >= SIMILARITY_THRESHOLD, else None.
```
### Function: `save_to_cache`
```text
Serializes and saves a successful agent execution to the database.
```
### Function: `delete_from_cache`
```text
Permanently removes any cache entry whose prompt is a semantic match for
user_prompt (above SIMILARITY_THRESHOLD), ensuring the next run is always
a fresh execution rather than a cache hit.
```

---

# File: .\agent\loop.py
### Function: `filter_schema`
```text
Uses the ROUTING_MODEL to intelligently filter the schema down to only relevant tables.
```
### Function: `decompose_question`
```text
Step 1: Breaks the user's prompt into specific data questions using chat history.
```
### Function: `execute_tool_call`
```text
Handles parsing, Pydantic validation, and execution of a single tool call.
Returns: (output_text, has_error, extracted_data_objects)
```
### Function: `run_agent_loop`
```text
The main orchestrator chaining the workflow together across multiple tools.
Decoupled from UI: Takes history in, returns structured dictionary out.
```
### Class: `SchemaSelection`
```text
Structured output model for the schema-filtering LLM call.
```

---

# File: .\agent\memory.py
### Function: `get_prompt_compressor`
```text
Initializes and caches the LLMLingua-2 PromptCompressor as a Streamlit singleton.
Uses @st.cache_resource so the model is only loaded into Databricks cluster memory once.
```
### Class: `ContextOptimizer`
```text
Manages conversational memory pruning and semantic prompt compression for the agentic loop.
```
### Class: `DataFrameMemory`
```text
In-memory registry to hold DataFrames generated during a conversation turn.
Allows downstream tools to reference upstream data via string IDs.
```
### Function: `get_df_memory`
```text
Returns the DataFrameMemory instance for the current Streamlit session.
Creates a new one on first call within a session.
```
### Function: `get_context_optimizer`
```text
Returns the ContextOptimizer instance for the current Streamlit session.
Creates a new one on first call within a session.  ContextOptimizer holds
only a tiktoken encoding (stateless beyond that), so sharing it is safe,
but keeping it in session_state is consistent and avoids cross-session
confusion if the tokenizer model ever becomes configurable per user.
```
### Function: `count_tokens`
```text
Returns the exact number of tokens in a string.
```
### Function: `prune_history_by_budget`
```text
Replaces arbitrary turn-slicing (`history[-6:]`) with exact token-budget pruning.
Iterates backwards from newest to oldest messages, including them until the budget is hit.
```
### Function: `compress_text`
```text
Uses LLMLingua-2 to semantically compress verbose text (like raw tool outputs or logs)
while strictly preserving numbers, entity names, and analytical context.
```
### Function: `compress_schema_context`
```text
Specialized compressor for JSON data dictionaries. Converts schema to string and compresses
while protecting table names and key types.
```
### Function: `format_history_for_prompt`
```text
All-in-one helper: Prunes history by token budget, formats to a readable string,
and compresses it if it's still dense.
```
### Function: `save_df`
```text
Saves a DataFrame and returns a unique reference ID.
```
### Function: `get_df`
```text
Retrieves a DataFrame by its ID.
```
### Function: `clear`
```text
Clears the registry to free up memory between isolated runs.
```

---

# File: .\agent\schemas.py
### Class: `SubQuestion`
```text
A single decomposed data question paired with its required execution category.
```
### Class: `DecomposedQuestions`
```text
The broken-down data queries based on the user's prompt, enriched with routing categories.
```
### Class: `run_neural_network_tool`
```text
Trains a Multi-Layer Perceptron (MLP) Neural Network for complex non-linear regression or classification.
Use this for advanced predictive modeling when basic regression or Random Forest is insufficient.
```
### Class: `run_optimization_tool`
```text
Runs linear programming optimization (using scipy.optimize.linprog) to maximize or minimize an objective function subject to linear constraints.
Use this to optimize budget allocation, resource distribution, or find the best mix under constraints.
```
### Class: `execute_python_tool`
```text
Executes raw Python code generated by the LLM in a secure, sandboxed environment.
Use this as a fallback for complex analytics, custom data manipulation, or advanced mathematical operations that pre-built tools cannot handle.

DATA LOADING RULES — read carefully:
- Single table: 'df' is a plain pandas DataFrame containing up to 100,000 rows from that table.
- Multiple tables: 'df' is a Python LIST of DataFrames, one per table in the order provided
  (e.g. df[0] = first table, df[1] = second table). Each table is loaded independently —
  NO pre-join is performed. Use pandas (pd.merge) to join them yourself inside the code.

The code also has access to numpy (np).
To return data to the LLM, assign the final text output to 'result_text' and any resulting DataFrame to 'result_df'.
```
### Class: `execute_sql_query_tool`
```text
Queries the Databricks database using PostgreSQL syntax. Because this is PostgreSQL, you MUST wrap all column names in double quotes to preserve exact capitalization. 
Do NOT use this tool if a more specific tool is available for the user's request:
- Use calculate_unit_economics_tool for CPA or CLV questions.
- Use run_scenario_planning_tool for ANY what-if analysis, simulating changes to variables, or holding variables constant.
- Use regression/forecasting/clustering tools for modeling.
Do NOT attempt to write complex SQL window functions, regressions, or simulations manually if a tool exists for it.
IMPORTANT — joining subcount_data_synced: this table has ~18 rows per month (one per Metric). 
You MUST filter by "Metric" and/or "Row_Type" BEFORE or WITHIN any join to avoid row multiplication. 
For example: JOIN ... ON year/month AND "subcount_data_synced"."Metric" = 'Ending Period Subscribers'.
Never join this table without a Metric or Row_Type filter in the ON or WHERE clause.
```
### Class: `run_ols_regression_tool`
```text
Performs an Ordinary Least Squares (OLS) multiple regression. cannot perform non-linear regression. 
Use this when the user asks to analyze the relationship, correlation, or impact of multiple independent numerical variables on a dependent target variable.
```
### Class: `run_forecasting_tool`
```text
Performs Holt-Winters Exponential Smoothing time series forecasting. Automatically resolves
the correct year/month column names for any table registered in TABLE_DIMENSIONS
(e.g., acquisition_data_v3, dbs_marketing_spend_sync, subcount_data_synced).
Use this when the user asks to predict or forecast future values based on historical trends.

NOTE: For subcount_data_synced, filter to a single Metric and Row_Type via a prior
execute_sql_query_tool call and pass the dataframe_id, since the table has multiple
metric rows per month that must be isolated before forecasting.

MODEL PARAMETER GUIDE:
- trend / seasonal: use 'add' (additive) when seasonal swings are roughly constant in size
  over time. Use 'mul' (multiplicative) when swings grow proportionally with the level
  of the series (e.g., a metric that doubles each year and whose seasonal spikes also double).
- seasonal_periods: set to 12 for monthly data (default), 4 for quarterly, 52 for weekly.
```
### Class: `run_pca_tool`
```text
Performs Principal Component Analysis (PCA) to reduce dimensionality and find the underlying variance/patterns in a set of features. 
Use this to identify which combinations of variables explain the most variance in the dataset.
```
### Class: `run_kmeans_clustering_tool`
```text
Performs K-Means clustering to group data into distinct segments based on feature similarities. 
Use this to discover customer segments, group similar behaviors, or identify natural groupings in the data.
```
### Class: `run_random_forest_tool`
```text
Trains a Random Forest machine learning model to predict a target variable based on multiple features. 
Use this to find non-linear relationships, classify outcomes, or determine the importance/impact of various features.
```
### Class: `compare_monthly_metrics_tool`
```text
Compares monthly marketing spend metrics against rolled-up acquisition metrics over time.
```
### Class: `calculate_unit_economics_tool`
```text
marketing cost per acquistion (CPA), Lifetime Value (CLV), and CLV:CPA ratios by safely merging marketing spend and acquisition volumes. Use this whenever the user asks about unit economics, cost per acquisition, or marketing efficiency.
```
### Class: `ScenarioChange`
```text
A single hypothetical change to a feature variable.
```
### Class: `run_scenario_planning_tool`
```text
Performs statistical what-if scenario planning and simulations using OLS regression. 
Use this tool whenever the user asks:
- What would happen to a target variable (Z) if a feature (X) changes by a certain percentage or to a specific value.
- Questions containing phrases like "what if", "assume X is", "increase/decrease by X%", or "hold Y constant".
This tool automatically computes baseline averages, applies the hypothetical changes, holds specified control variables constant at their historical means, and returns expected predictions with 95% confidence intervals.
```
### Class: `generate_scatterplot_tool`
```text
Generates an interactive scatterplot to explore relationships between two numerical variables.
Supports querying a single table or automatically joining multiple tables.
```
### Class: `generate_barchart_tool`
```text
Generates a bar chart to compare aggregated numerical values across categorical groups or time periods.
Automatically handles pre-aggregation (SUM, AVG, COUNT) to ensure clean visualizations.
```
### Class: `generate_histogram_tool`
```text
Generates a histogram with an executive box-plot marginal to visualize data distributions, spread, and outliers.
```
### Class: `generate_linechart_tool`
```text
Generates a continuous line chart to visualize trends over time or sequences.
Automatically groups duplicate timestamps and sorts chronologically to prevent erratic line jumps.
```

---

# File: .\toolkit\analytics.py
### Function: `link_tables`
```text
Centralized data-fetching helper. Dynamically builds SQL queries and joins multiple 
tables based on shared conformed dimensions defined in TABLE_DIMENSIONS in base.py.
```
### Function: `execute_sql_query_tool`
```text
Executes an arbitrary PostgreSQL query and returns up to 100 preview rows as CSV text.
Returns a dict with 'text' (summary + CSV preview) and 'data' (full DataFrame).
```
### Function: `run_ols_regression_tool`
```text
Fits an OLS multiple regression model using statsmodels and returns the full
summary table as text plus the fitted model object.
Accepts data either from a live table query (TABLE_NAME) or a pre-fetched
DataFrame stored in memory (dataframe_id).
```
### Function: `run_forecasting_tool`
```text
Aggregates value_column to one observation per calendar month, then fits a
Holt-Winters Exponential Smoothing model and forecasts `steps` periods ahead.
Year/month column names are resolved automatically from TABLE_DIMENSIONS so
this function works across all registered tables without manual configuration.
Returns forecast values as formatted text and the fitted model object.
```
### Function: `run_pca_tool`
```text
Standardizes the requested feature columns and fits a PCA model to identify
the principal components that explain the most variance. Returns per-component
explained variance ratios and the top feature loadings (|loading| > 0.3) for
the first two components, plus the fitted PCA object.
```
### Function: `run_kmeans_clustering_tool`
```text
Standardizes features and fits a K-Means model to partition data into
n_clusters groups. Returns cluster population sizes, the top 5 defining
standardized centroid values per cluster, and the fitted KMeans object.
```
### Function: `calculate_unit_economics_tool`
```text
Joins monthly marketing spend against monthly activation counts to compute
CPA (Cost Per Acquisition), CLV (Customer Lifetime Value via NPV of MCF), and
the CLV:CPA ratio for each month. Returns a blended summary and the merged
monthly DataFrame with all computed columns.

Uses MONTHLY_WACC and avg_churn to discount future cash flows for CLV.
Both where_clause params are applied independently to their respective tables
before the inner join, allowing independent filtering (e.g. by channel or segment).
```
### Function: `run_scenario_planning_tool`
```text
Fits an OLS regression on historical data, then predicts the target variable
under a hypothetical scenario where specified features are set to new values
and all remaining features are held at their historical means.

Returns a formatted report including: baseline stats, per-feature marginal
impacts (β coefficients), the scenario prediction, net change vs baseline,
and a prediction interval at the requested confidence level.

For multi-table scenarios involving marketing + acquisition data, pass both
table names and the function will merge them automatically on year/month.
```
### Function: `run_neural_network_tool`
```text
Trains a scikit-learn MLPRegressor or MLPClassifier on the provided features.
Features are one-hot encoded for categoricals and StandardScaler-normalized before
training. Uses early stopping to avoid overfitting on small datasets.
Returns the R² score (regression) or accuracy (classification) on the held-out
test set, plus the fitted model and the cleaned DataFrame.
```
### Function: `run_optimization_tool`
```text
Solves a linear programming problem using scipy.optimize.linprog (HiGHS solver).
Minimizes c·x subject to A_ub·x ≤ b_ub, A_eq·x = b_eq, and per-variable bounds.
To maximize instead of minimize, pass negative objective coefficients.
Returns the optimal objective value and decision variable values on success,
or the solver failure message on infeasibility.
```
### Function: `execute_python_tool`
```text
Executes LLM-generated Python code in a restricted sandbox with access to
pre-loaded pandas DataFrames (df), numpy (np), and pandas (pd).

Single-table: df is a plain DataFrame. Multi-table: df is a list of DataFrames
in the same order as TABLE_NAME, each loaded independently (no pre-join).

The code can write results back via two reserved variables:
  - result_text (str): narrative output to surface to the LLM
  - result_df (DataFrame): tabular output to attach to the agent turn

Blocks execution if the code references any forbidden modules or SQL mutations.
Stdout is captured and included in the return text.
```

---

# File: .\toolkit\base.py
### Class: `ModelConfig`
```text
Global configuration holder for the active LLM endpoint.
ACTIVE_MODEL is updated at runtime via set_active_model() when the user
selects a model from the sidebar, and is read by every LLM call site.
```
### Function: `set_active_model`
```text
Updates the single active model endpoint based on the user's sidebar selection.
```
### Function: `get_auth_token`
```text
Dynamically fetches a fresh token from the Databricks SDK on every call.
```
### Function: `_make_fresh_openai_client`
```text
Single factory for all OpenAI client construction.
Always fetches a fresh token so short-lived Databricks tokens never go stale.
Used by both DynamicOpenAIClient and llm_call so token rotation and the
base URL are defined in exactly one place.
```
### Class: `DynamicOpenAIClient`
```text
Proxy wrapper that ensures every API call uses a freshly rotated token.
Intercepts .chat.completions.create to strip 'strict: True' from tool
definitions, which Claude rejects.
```
### Function: `get_db_engine`
```text
Caches the SQLAlchemy engine pool, but delegates physical connection 
creation to a dynamic function so Postgres always gets a fresh token.
```
### Function: `run_sql_query`
```text
Executes a raw SQL string against the Postgres engine and returns a DataFrame.
```
### Function: `_is_gpt_model`
```text
Returns True if the endpoint name indicates a GPT model (OpenAI-native tool-calling).
```
### Function: `_extract_text_content`
```text
Safely extracts the text string from a ChatCompletionMessage regardless of
whether content is a plain string or a Gemini-style list of content blocks
(e.g. [{'type': 'text', 'text': '...', 'thoughtSignature': '...'}]).
```
### Function: `llm_call`
```text
Structured-output LLM call with cross-model compatibility.

- GPT endpoints: uses instructor TOOLS mode (native tool-calling + JSON schema).
  Requires a real OpenAI instance, so uses _make_fresh_openai_client() directly.
- All other endpoints (Gemini, Claude, etc.): uses raw_client for the completion
  call so token rotation and Claude 'strict' sanitization are handled automatically,
  then parses the plain-text JSON response with Pydantic.
```
### Function: `track_tokens`
```text
Directly extracts token usage from a live OpenAI/Databricks SDK response object.
```
### Function: `get_join_clause`
```text
Dynamically generates a SQL ON clause by intersecting the shared 
conformed dimensions between two tables.
```
### Function: `__getattr__`
```text
Intercepts all attribute access. Builds a fresh OpenAI client on every
call (token rotation), then returns either a ChatProxy (for .chat) or
the raw attribute from the underlying client for everything else.
```
### Function: `get_fresh_connection`
```text
Creates a new pg8000 connection with a freshly fetched auth token.
```
### Class: `ChatProxy`
```text
Wraps the OpenAI chat namespace to allow pre-call sanitization.
```
### Function: `completions`
```text
Returns a CompletionsProxy that strips model-incompatible fields.
```
### Class: `CompletionsProxy`
```text
Sanitizes tool definitions before forwarding to the real client.
```

---

# File: .\toolkit\visuals.py
### Function: `_resolve_column`
```text
Safely resolves column names in Pandas DataFrames after SQL execution.
```
### Function: `_fetch_chart_data`
```text
SHARED LOGIC HELPER: Handles routing between SQL and Python Memory.
Automatically applies Pandas .groupby() or SQL GROUP BY if an aggregation is requested.
Returns: (df, resolved_x, resolved_y, resolved_cat, y_axis_label)
```
### Function: `generate_scatterplot_tool`
```text
Generates an interactive Plotly scatterplot of y_column vs x_column.
Optionally color-codes points by category_column and overlays an OLS trendline.
Both axes are coerced to numeric and non-finite rows are dropped before plotting.
Returns the figure object, the plotted DataFrame, and a status text string.
```
### Function: `generate_barchart_tool`
```text
Generates an interactive Plotly bar chart aggregating y_column per x_column group.
Aggregation (SUM/AVG/COUNT/MAX/MIN) is pushed down to SQL or applied via pandas
groupby depending on the data source. When category_column is provided, bars are
rendered side-by-side (grouped) and color-coded by category.
Sorts by x_column ascending; falls back to y_column descending if x is non-sortable.
```
### Function: `generate_histogram_tool`
```text
Generates a Plotly histogram with a box-plot marginal for x_column.
When category_column is provided, multiple overlapping distributions are rendered,
one per category. n_bins controls bin granularity (auto if omitted).
Non-numeric and null values are dropped before plotting.
```
### Function: `generate_linechart_tool`
```text
Generates a Plotly line chart showing the trend of y_column over x_column.
Data is sorted ascending by x_column to prevent line jumps. Aggregation is
applied if multiple rows share the same x value (e.g., duplicate months).
When category_column is provided, a separate colored line is drawn per category.
Uses unified hover mode so all series values are visible at each x position.
```
### Function: `compare_monthly_metrics_tool`
```text
Kept as SQL-only because it relies on a very specific cross-table join macro.
```

---

# File: .\toolkit\__init__.py
### Function: `_resolve_refs`
```text
Recursively resolves all $ref pointers in a JSON Schema dict by inlining
the referenced $defs entries, then strips the $defs key entirely.
Returns a deep-copied, fully-flattened schema.
```
### Function: `_flatten_tool`
```text
Takes a pydantic_function_tool dict and returns a copy with its parameters
schema fully resolved (no $defs / $ref).
```

---

