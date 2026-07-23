import contextlib
import io
import traceback
from typing import Any, Dict, List, Optional, Union


import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.optimize import linprog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Project imports
from .base import run_sql_query, get_join_clause, TABLE_DIMENSIONS
from agent.memory import DataFrameMemory

YEARLY_WACC = 0.1
MONTHLY_WACC = (1 + YEARLY_WACC) ** (1 / 12) - 1

__all__ = [
    "link_tables",
    "execute_sql_query_tool",
    "run_ols_regression_tool",
    "run_forecasting_tool",
    "run_random_forest_tool",
    "run_pca_tool",
    "run_kmeans_clustering_tool",
    "calculate_unit_economics_tool",
    "calculate_ratio_tool",
    "run_scenario_planning_tool",
    "execute_python_tool",
    "run_neural_network_tool",
    "run_optimization_tool"
]


@mlflow.trace(name="link_tables")
def link_tables(
    tables: Union[str, List[str]], 
    columns: Optional[List[str]] = None, 
    where_clause: Optional[str] = None, 
    group_by: Optional[List[str]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = 100000, 
    random_order: bool = False
) -> pd.DataFrame:
    """
    Centralized data-fetching helper. Dynamically builds SQL queries and joins multiple 
    tables based on shared conformed dimensions defined in TABLE_DIMENSIONS in base.py.
    """
    if isinstance(tables, str):
        if "," in tables:
            table_list = [t.strip() for t in tables.split(",")]
        else:
            table_list = [tables.strip()]
    else:
        table_list = [str(t).strip() for t in tables]
        
    table_list = list(dict.fromkeys(table_list))
    
    if columns:
        safe_cols = []
        for col in columns:
            clean_col = col.replace('"', '').replace("'", "").strip()
            if "." in clean_col or any(func in clean_col.upper() for func in ["SUM(", "AVG(", "COUNT(", "MIN(", "MAX("]):
                safe_cols.append(col)
            else:
                safe_cols.append(f'"{clean_col}"')
        columns_str = ", ".join(safe_cols)
    else:
        columns_str = "*"
        
    base_table = table_list[0]
    from_clause = f"FROM {base_table}"
    
    if len(table_list) > 1:
        joined_tables = [base_table]
        for next_table in table_list[1:]:
            join_condition = None
            for joined_t in joined_tables:
                try:
                    cond = get_join_clause(joined_t, next_table)
                    if cond:
                        join_condition = cond
                        break
                except ValueError:
                    continue
            
            if not join_condition:
                raise ValueError(
                    f"No shared dimensions found in TABLE_DIMENSIONS between '{next_table}' "
                    f"and currently joined tables ({joined_tables}). Please update base.py."
                )
            
            from_clause += f" INNER JOIN {next_table} ON {join_condition}"
            joined_tables.append(next_table)
            
    sql_query = f"SELECT {columns_str} {from_clause}"
    
    if where_clause:
        sql_query += f" WHERE {where_clause}"
        
    if group_by:
        safe_group = []
        for col in group_by:
            if "." not in col:
                clean_col = col.replace('"', '').replace("'", "").strip()
                safe_group.append(f'"{clean_col}"')
            else:
                safe_group.append(col)
        sql_query += f" GROUP BY {', '.join(safe_group)}"
        
    if random_order:
        sql_query += " ORDER BY RANDOM()"
    elif order_by:
        sql_query += f" ORDER BY {order_by}"
        
    if limit:
        sql_query += f" LIMIT {limit}"
        
    df = run_sql_query(sql_query)
    df.columns = [str(col).replace('"', '').replace("'", "").strip() for col in df.columns]
    return df


@mlflow.trace(name="execute_sql_query")
def execute_sql_query_tool(sql_query: str) -> dict:
    """
    Executes an arbitrary PostgreSQL query and returns up to 100 preview rows as CSV text.
    Returns a dict with 'text' (summary + CSV preview) and 'data' (full DataFrame).
    """
    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: Query executed successfully, but returned 0 rows.", "data": None}
        
        csv_text = df.head(100).to_csv(index=False)
        return {"text": f"Success. Showing top 100 rows:\n{csv_text}", "data": df}
        
    except Exception as e:
        return {"text": f"Error executing SQL: {str(e)}", "data": None}
    

@mlflow.trace(name="run_ols_regression_tool")
def run_ols_regression_tool(
    dependent_variable: str, 
    independent_variables: list,
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    df_memory: DataFrameMemory = None
) -> dict:
    """
    Fits an OLS multiple regression model using statsmodels and returns the full
    summary table as text plus the fitted model object.
    Accepts data either from a live table query (TABLE_NAME) or a pre-fetched
    DataFrame stored in memory (dataframe_id).
    """
    columns_to_fetch = [dependent_variable] + independent_variables
    
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}
        elif TABLE_NAME:
            df = link_tables(TABLE_NAME, columns=columns_to_fetch, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}
            
        df = df.dropna(subset=[col for col in columns_to_fetch if col in df.columns])
        
        if df.empty or len(df) <= len(independent_variables):
            return {"text": "Error: Not enough valid data points to perform regression.", "data": None}
            
        Y = pd.to_numeric(df[dependent_variable])
        X = df[independent_variables].apply(pd.to_numeric, errors='coerce')
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        return {"text": model.summary().as_text(), "data": model}
        
    except Exception as e:
        return {"text": f"Regression Error: {e}", "data": None}
    

@mlflow.trace(name="run_forecasting_tool")
def run_forecasting_tool(
    value_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    aggregation: str = "SUM", 
    steps: int = 6,
    trend: str = "add",      # 'add' or 'mul'
    seasonal: str = "add",   # 'add' or 'mul'
    seasonal_periods: int = 12,
    df_memory: DataFrameMemory = None
) -> dict:
    """
    Aggregates value_column to one observation per calendar month, then fits a
    Holt-Winters Exponential Smoothing model and forecasts `steps` periods ahead.
    Year/month column names are resolved automatically from TABLE_DIMENSIONS so
    this function works across all registered tables without manual configuration.
    Returns forecast values as formatted text and the fitted model object.
    """
    safe_value = '"{}"'.format(value_column.replace('"', ''))
    agg_func = aggregation.upper() if aggregation.upper() in ["SUM", "AVG", "COUNT"] else "SUM"
    val_col_clean = value_column.replace('"', '').strip()

    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}

            # Detect whichever year/month columns are present in the dataframe.
            # Check TABLE_DIMENSIONS values first, then fall back to common synonyms.
            known_year_cols = {dims["year"] for dims in TABLE_DIMENSIONS.values()}
            known_month_cols = {dims["month"] for dims in TABLE_DIMENSIONS.values()}

            year_col = next((c for c in known_year_cols if c in df.columns), None)
            month_col = next((c for c in known_month_cols if c in df.columns), None)

            if year_col and month_col and val_col_clean in df.columns:
                # Aggregate down to one row per period
                agg_map = {"SUM": "sum", "AVG": "mean", "COUNT": "count"}
                df = (
                    df.groupby([year_col, month_col], as_index=False)[val_col_clean]
                    .agg(agg_map[agg_func])
                    .sort_values(by=[year_col, month_col])
                    .rename(columns={val_col_clean: "target_value"})
                )
            elif val_col_clean in df.columns:
                # Dataframe is already a clean time series — use it as-is
                df = df.copy()
                df["target_value"] = df[val_col_clean]
            else:
                return {"text": f"Error: Column '{val_col_clean}' not found in the provided dataframe. Available columns: {df.columns.tolist()}", "data": None}

        elif TABLE_NAME:
            # Resolve the canonical year/month column names for this table from TABLE_DIMENSIONS.
            # Normalize TABLE_NAME to a single string key for the lookup.
            table_key = TABLE_NAME if isinstance(TABLE_NAME, str) else TABLE_NAME[0]
            dims = TABLE_DIMENSIONS.get(table_key)

            if dims is None:
                return {"text": f"Error: Table '{table_key}' not found in TABLE_DIMENSIONS. Please add it to base.py.", "data": None}

            year_col = dims["year"]
            month_col = dims["month"]

            columns_to_fetch = [
                f'"{year_col}"',
                f'"{month_col}"',
                f'{agg_func}({safe_value}) AS target_value'
            ]
            df = link_tables(
                tables=TABLE_NAME,
                columns=columns_to_fetch,
                where_clause=f'"{year_col}" IS NOT NULL AND "{month_col}" IS NOT NULL',
                group_by=[year_col, month_col],
                order_by=f'"{year_col}" ASC, "{month_col}" ASC',
                limit=None
            )
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}

        df["target_value"] = pd.to_numeric(df.get("target_value", pd.Series(dtype=float)), errors="coerce")
        df = df.dropna(subset=["target_value"])

        if df.empty or len(df) < 10:
            return {"text": "Error: Not enough historical data points (minimum 10 required) to perform ARIMA.", "data": None}

        series = df["target_value"].values

        model = ExponentialSmoothing(
            series, 
            trend=trend, 
            seasonal=seasonal, 
            seasonal_periods=seasonal_periods
        )
        # Using optimized=True allows statsmodels to find the best smoothing weights
        model_fit = model.fit(optimized=True) 
        forecast = model_fit.forecast(steps=steps)

        result_text = f"Holt-Winters Forecasting Results for {agg_func} of {value_column}:\n"
        result_text += f"Based on {len(series)} periods (Trend: {trend}, Seasonal: {seasonal}, Periods: {seasonal_periods})\n"
        result_text += f"Predictions for the next {steps} periods:\n"
        for i, val in enumerate(forecast, start=1):
            result_text += f"  • Period +{i}: {val:.4f}\n"

        return {"text": result_text, "data": model_fit}

    except Exception as e:
        return {"text": f"Holt-Winters Forecasting Error: {e}", "data": None}


@mlflow.trace(name="run_random_forest_tool")
def run_random_forest_tool(
    target_variable: str, 
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    task_type: str = "regression", 
    n_estimators: int = 100,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    columns_to_fetch = [target_variable] + feature_variables

    try:
        # ── 1. Data Loading ──────────────────────────────────────────────
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = link_tables(TABLE_NAME, columns=columns_to_fetch, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}

        # ── 2. Column Validation ─────────────────────────────────────────
        # Check requested columns actually exist before doing any work.
        missing = [c for c in columns_to_fetch if c not in df.columns]
        if missing:
            return {
                "text": f"Error: The following columns were not found in the data: {missing}. "
                        f"Available columns: {df.columns.tolist()}",
                "model": None
            }

        # Narrow the dataframe to only the columns we care about so stray
        # underscore-named columns from the broader table can never leak in.
        df = df[columns_to_fetch].copy()

        if df.empty or len(df) <= len(feature_variables):
            return {"text": "Error: Not enough data points.", "model": None}

        # ── 3. Target Preparation ────────────────────────────────────────
        task = task_type.lower()
        if task == "regression":
            df[target_variable] = pd.to_numeric(df[target_variable], errors="coerce")
        else:
            # For classification keep target as string so class labels are readable
            df[target_variable] = df[target_variable].astype(str)

        # Drop rows where the target is missing before encoding features
        df = df.dropna(subset=[target_variable])

        if len(df) < 10:
            return {"text": "Error: Not enough valid target rows to train a model.", "model": None}

        # ── 4. Feature Encoding ──────────────────────────────────────────
        # Identify which of the *requested* feature columns are categorical
        categorical_features = [
            col for col in feature_variables
            if col in df.columns and df[col].dtype == "object"
        ]
        numeric_features = [
            col for col in feature_variables
            if col in df.columns and col not in categorical_features
        ]

        # Convert numeric features, coercing unparseable values to NaN
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # One-hot encode categoricals; drop_first avoids perfect multicollinearity
        if categorical_features:
            df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # Rebuild the feature list from the current df columns — this correctly
        # picks up the new one-hot columns (e.g. 'channel_TV', 'channel_Digital')
        # while excluding the target and any other columns that might have slipped in.
        current_features = [col for col in df.columns if col != target_variable]

        # Drop any rows with NaN in features or target
        df = df.dropna(subset=[target_variable] + current_features)

        if len(df) < 10:
            return {"text": "Error: Data size too small after cleaning to train a valid model.", "model": None}

        # ── 5. Train / Test Split ────────────────────────────────────────
        X = df[current_features]
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ── 6. Model Fitting & Evaluation ────────────────────────────────
        if task == "regression":
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=7,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            result_text = f"Random Forest Regression Results (n_estimators={n_estimators}):\n"
            result_text += f"  • Test R²:   {r2_score(y_test, preds):.4f}\n"
            result_text += f"  • Test RMSE: {mean_squared_error(y_test, preds) ** 0.5:.4f}\n\n"
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=7,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            result_text = f"Random Forest Classification Results (n_estimators={n_estimators}):\n"
            result_text += f"  • Test Accuracy: {accuracy_score(y_test, preds):.4f}\n"
            result_text += f"Classification Report:\n{classification_report(y_test, preds)}\n\n"

        # ── 7. Feature Importances ───────────────────────────────────────
        feat_imp = sorted(
            zip(current_features, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        result_text += f"Feature Importances — top {min(10, len(feat_imp))} of {len(feat_imp)} features "
        result_text += f"(trained on {len(X_train):,} rows, tested on {len(X_test):,} rows):\n"
        for feat, imp in feat_imp[:10]:
            result_text += f"  • {feat}: {imp:.4f}\n"

        return {"text": result_text, "model": model}

    except Exception as e:
        return {"text": f"Random Forest Error: {e}", "model": None}
    

@mlflow.trace(name="run_pca_tool")
def run_pca_tool(
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    n_components: int = None,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Standardizes the requested feature columns and fits a PCA model to identify
    the principal components that explain the most variance. Returns per-component
    explained variance ratios and the top feature loadings (|loading| > 0.3) for
    the first two components, plus the fitted PCA object.
    """
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = link_tables(TABLE_NAME, columns=feature_variables, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}
            
        if df.empty or len(df) < 2:
            return {"text": "Error: Not enough data points fetched to perform PCA.", "model": None}
            
        df = df[[col for col in feature_variables if col in df.columns]]
        df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'], drop_first=True)
        df = df.dropna()
        
        current_features = df.columns.tolist()
        
        if len(df) < 2 or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning to perform PCA.", "model": None}
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        max_components = min(len(df), len(current_features))
        actual_components = max_components if n_components is None or n_components > max_components else n_components
            
        pca = PCA(n_components=actual_components)
        pca.fit(scaled_data)
        
        result_text = f"PCA Results (n_components={actual_components}):\n"
        explained_variance = pca.explained_variance_ratio_
        
        result_text += "Explained Variance Ratio per Component:\n"
        for i, var in enumerate(explained_variance):
            result_text += f"  • PC{i+1}: {var:.4f} ({(var*100):.1f}%)\n"
        result_text += f"Total Explained Variance: {sum(explained_variance):.4f} ({(sum(explained_variance)*100):.1f}%)\n\n"
        
        components_to_show = min(2, actual_components)
        result_text += "Top Feature Loadings (absolute magnitude > 0.3):\n"
        
        for i in range(components_to_show):
            result_text += f"  PC{i+1} Signficant Loadings:\n"
            loadings = pca.components_[i]
            feat_loadings = sorted(zip(current_features, loadings), key=lambda x: abs(x[1]), reverse=True)
            for feat, load in feat_loadings:
                if abs(load) > 0.3:
                    result_text += f"    - {feat}: {load:.4f}\n"
        
        return {"text": result_text, "model": pca}
        
    except Exception as e:
        return {"text": f"PCA Error: {e}", "model": None}
    

@mlflow.trace(name="run_kmeans_clustering_tool")
def run_kmeans_clustering_tool(
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    n_clusters: int = 3,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Standardizes features and fits a K-Means model to partition data into
    n_clusters groups. Returns cluster population sizes, the top 5 defining
    standardized centroid values per cluster, and the fitted KMeans object.
    """
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = link_tables(TABLE_NAME, columns=feature_variables, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}
            
        if df.empty or len(df) < n_clusters:
            return {"text": f"Error: Not enough data points fetched to perform {n_clusters}-means clustering.", "model": None}
            
        df = df[[col for col in feature_variables if col in df.columns]]
        df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'], drop_first=True)
        df = df.dropna()
        current_features = df.columns.tolist()
        
        if len(df) < n_clusters or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning to perform clustering.", "model": None}
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(scaled_data)
        
        df['Cluster'] = kmeans.labels_
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        result_text = f"K-Means Clustering Results (n_clusters={n_clusters}):\n"
        result_text += "Cluster Population Sizes:\n"
        for cluster_id, count in cluster_counts.items():
            result_text += f"  • Cluster {cluster_id}: {count} data points\n"
            
        result_text += "\nCluster Profiles (Standardized Centroids):\n"
        
        centroids = kmeans.cluster_centers_
        for i in range(n_clusters):
            result_text += f"  Cluster {i} Defining Features (Top 5):\n"
            feat_centroids = sorted(zip(current_features, centroids[i]), key=lambda x: abs(x[1]), reverse=True)
            for feat, val in feat_centroids[:5]:
                if abs(val) > 0.15: 
                    result_text += f"    - {feat}: {val:.4f}\n"
                    
        return {"text": result_text, "model": kmeans}
        
    except Exception as e:
        return {"text": f"K-Means Error: {e}", "model": None}
    

@mlflow.trace(name="calculate_unit_economics_tool")
def calculate_unit_economics_tool(marketing_where_clause: str = None, subscriber_where_clause: str = None) -> dict:
    """
    Joins monthly marketing spend against monthly activation counts to compute
    CPA (Cost Per Acquisition), CLV (Customer Lifetime Value via NPV of MCF), and
    the CLV:CPA ratio for each month. Returns a blended summary and the merged
    monthly DataFrame with all computed columns.

    Uses MONTHLY_WACC and avg_churn to discount future cash flows for CLV.
    Both where_clause params are applied independently to their respective tables
    before the inner join, allowing independent filtering (e.g. by channel or segment).
    """
    # Kept as-is since the schema explicitly handles the two-table where clauses without a TABLE_NAME arg.
    try:
        # 1. Marketing Data
        df_mkt = link_tables(
            tables='"sandbox"."dbs_marketing_sync"',
            # Keep the quotes for Postgres, but drop the 'AS' alias
            columns=['"Year"', '"Month"', 'SUM("Amount") AS total_spend'], 
            where_clause=marketing_where_clause,
            group_by=['"Year"', '"Month"'],
            limit=None
        )
        
        # Strip the quotes out of the resulting pandas column names if the driver leaves them in
        if not df_mkt.empty:
            df_mkt.columns = [col.replace('"', '') for col in df_mkt.columns]

        # 2. Acquisition Data
        df_acq = link_tables(
            tables='"sandbox"."subcount_data_synced"',
            # Keep the quotes to protect the capital letters for Postgres
            columns=[
                '"Year"', 
                '"Month"', 
                'SUM("Amount") AS total_activations'
            ],
            where_clause=subscriber_where_clause,
            group_by=['"Year"', '"Month"'],
            limit=None
        )

        for df_tmp in [df_mkt, df_acq]:
            df_tmp['Year'] = pd.to_numeric(df_tmp['Year'], errors='coerce')
            df_tmp['Month'] = pd.to_numeric(df_tmp['Month'], errors='coerce')

        df_merged = pd.merge(df_mkt, df_acq, on=['Year', 'Month'], how='inner')

        if df_merged.empty:
            return {"text": "Error: Could not calculate UNit Economics. No overlapping months found.", "data": None}

        df_merged['cpa'] = df_merged['total_spend'] / df_merged['total_activations']

        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_merged = df_merged.sort_values(by=['Year', 'Month'])

        df_merged['Date'] = pd.to_datetime(
            df_merged['Year'].astype(int).astype(str) + '-' + 
            df_merged['Month'].astype(int).astype(str) + '-01', 
            errors='coerce'
        )

        overall_spend = df_merged['total_spend'].sum()
        overall_acq = df_merged['total_activations'].sum()
        blended_cpa = overall_spend / overall_acq if overall_acq > 0 else 0
        
        text_output = (
            f"Unit Economics Summary:\n"
            f"  • Total Marketing Spend Analyzed: ${overall_spend:,.2f}\n"
            f"  • Total Activations: {overall_acq:,.0f}\n"
            f"  • Blended CPA: ${blended_cpa:,.2f}\n"
        )

        return {"text": text_output, "data": df_merged}

    except Exception as e:
        return {"text": f"Unit Economics Calculation Error: {e}", "data": None}


@mlflow.trace(name="calculate_ratio_tool")
def calculate_ratio_tool(
    numerator_column: str,
    numerator_table: str,
    denominator_column: str,
    denominator_table: str,
    where_clause: str = None,
    numerator_aggregation: str = "SUM",
    denominator_aggregation: str = "SUM",
) -> dict:
    """
    Calculates a monthly ratio (numerator / denominator) between any two numeric columns,
    which may live in the same table or in two different tables.

    Both sides are independently aggregated to one value per (year, month) period using
    their respective aggregation functions, then joined on the shared time dimensions.
    Returns a DataFrame with columns: year, month, <numerator_column>, <denominator_column>,
    and ratio_<numerator_column>_per_<denominator_column>.
    """
    VALID_AGGS = {"SUM", "AVG", "COUNT"}
    num_agg = numerator_aggregation.upper() if numerator_aggregation.upper() in VALID_AGGS else "SUM"
    den_agg = denominator_aggregation.upper() if denominator_aggregation.upper() in VALID_AGGS else "SUM"

    try:
        same_table = numerator_table.strip() == denominator_table.strip()

        if same_table:
            # Both columns live in the same table — fetch in a single query.
            dims = TABLE_DIMENSIONS.get(numerator_table.strip())
            if dims is None:
                return {"text": f"Error: Table '{numerator_table}' not found in TABLE_DIMENSIONS.", "data": None}

            year_col = dims["year"]
            month_col = dims["month"]

            df = link_tables(
                tables=numerator_table,
                columns=[
                    f'"{year_col}"',
                    f'"{month_col}"',
                    f'{num_agg}("{numerator_column}") AS numerator_val',
                    f'{den_agg}("{denominator_column}") AS denominator_val',
                ],
                where_clause=where_clause,
                group_by=[year_col, month_col],
                order_by=f'"{year_col}" ASC, "{month_col}" ASC',
                limit=None,
            )
            df.columns = [c.replace('"', '') for c in df.columns]
            df.rename(columns={year_col: "year", month_col: "month"}, inplace=True)

        else:
            # Two different tables — query each independently then join on year/month.
            num_dims = TABLE_DIMENSIONS.get(numerator_table.strip())
            den_dims = TABLE_DIMENSIONS.get(denominator_table.strip())

            if num_dims is None:
                return {"text": f"Error: Table '{numerator_table}' not found in TABLE_DIMENSIONS.", "data": None}
            if den_dims is None:
                return {"text": f"Error: Table '{denominator_table}' not found in TABLE_DIMENSIONS.", "data": None}

            num_year, num_month = num_dims["year"], num_dims["month"]
            den_year, den_month = den_dims["year"], den_dims["month"]

            df_num = link_tables(
                tables=numerator_table,
                columns=[
                    f'"{num_year}"',
                    f'"{num_month}"',
                    f'{num_agg}("{numerator_column}") AS numerator_val',
                ],
                where_clause=where_clause,
                group_by=[num_year, num_month],
                order_by=f'"{num_year}" ASC, "{num_month}" ASC',
                limit=None,
            )
            df_num.columns = [c.replace('"', '') for c in df_num.columns]
            df_num.rename(columns={num_year: "year", num_month: "month"}, inplace=True)

            df_den = link_tables(
                tables=denominator_table,
                columns=[
                    f'"{den_year}"',
                    f'"{den_month}"',
                    f'{den_agg}("{denominator_column}") AS denominator_val',
                ],
                where_clause=where_clause,
                group_by=[den_year, den_month],
                order_by=f'"{den_year}" ASC, "{den_month}" ASC',
                limit=None,
            )
            df_den.columns = [c.replace('"', '') for c in df_den.columns]
            df_den.rename(columns={den_year: "year", den_month: "month"}, inplace=True)

            for df_tmp in [df_num, df_den]:
                df_tmp["year"] = pd.to_numeric(df_tmp["year"], errors="coerce")
                df_tmp["month"] = pd.to_numeric(df_tmp["month"], errors="coerce")

            df = pd.merge(df_num, df_den, on=["year", "month"], how="inner")

        if df.empty:
            return {"text": "Error: No overlapping year/month periods found for the two columns.", "data": None}

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df["numerator_val"] = pd.to_numeric(df["numerator_val"], errors="coerce")
        df["denominator_val"] = pd.to_numeric(df["denominator_val"], errors="coerce")

        ratio_col = f"ratio_{numerator_column}_per_{denominator_column}"
        df[ratio_col] = df["numerator_val"] / df["denominator_val"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.rename(columns={
            "numerator_val": numerator_column,
            "denominator_val": denominator_column,
        }, inplace=True)

        df = df.sort_values(by=["year", "month"]).reset_index(drop=True)

        avg_ratio = df[ratio_col].mean()
        min_ratio = df[ratio_col].min()
        max_ratio = df[ratio_col].max()

        text_output = (
            f"Monthly Ratio — {numerator_column} / {denominator_column}:\n"
            f"  • Periods computed: {len(df)}\n"
            f"  • Average ratio: {avg_ratio:,.4f}\n"
            f"  • Min: {min_ratio:,.4f}  |  Max: {max_ratio:,.4f}\n"
        )

        return {"text": text_output, "data": df}

    except Exception as e:
        return {"text": f"Ratio Calculation Error: {e}", "data": None}


@mlflow.trace(name="run_scenario_planning_tool")
def run_scenario_planning_tool(
    target_variable: str, 
    scenario_changes: list,
    hold_constant_variables: list,
    TABLE_NAME: Optional[Union[str, List[str]]] = None, 
    dataframe_id: Optional[str] = None,
    where_clause: Optional[str] = None,
    confidence_level: float = 0.95,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Table-agnostic scenario planning tool. Fits an OLS regression on historical data, 
    then predicts the target variable under a hypothetical scenario where specified 
    features are set to new values and specified control features are held constant at 
    their historical means.
    
    Dynamically joins any combination of tables using link_tables and TABLE_DIMENSIONS.
    """
    # Clean input strings
    target_variable = str(target_variable).replace('"', '').replace("'", "").strip()
    hold_constant_variables = [str(col).replace('"', '').replace("'", "").strip() for col in hold_constant_variables]
    
    # Map out the changes
    changes_map = {}
    for item in scenario_changes:
        col_name = item.get("column_name", "") if isinstance(item, dict) else getattr(item, "column_name", "")
        val = item.get("new_value", 0.0) if isinstance(item, dict) else getattr(item, "new_value", 0.0)
        clean_col = str(col_name).replace('"', '').replace("'", "").strip()
        changes_map[clean_col] = float(val)
        
    all_features = list(set(hold_constant_variables + list(changes_map.keys())))
    all_columns = [target_variable] + all_features
    
    try:
        # 1. Fetch Data
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None, "model": None}
        elif TABLE_NAME:
            # Delegate all complex cross-table joining to the centralized helper
            df = link_tables(
                tables=TABLE_NAME, 
                columns=all_columns, 
                where_clause=where_clause,
                random_order=True, 
                limit=100000
            )
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None, "model": None}
            
        # 2. Validate and Clean Data
        missing_cols = [col for col in all_columns if col not in df.columns]
        if missing_cols:
            return {
                "text": f"Error: Missing required columns: {missing_cols}. Available: {df.columns.tolist()}", 
                "data": None, 
                "model": None
            }

        for col in all_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=all_columns)
        
        if df.empty or len(df) <= len(all_features) + 3:
            return {"text": "Error: Not enough data points to build a reliable scenario model.", "data": None, "model": None}
            
        # 3. Fit OLS Model
        historical_target_mean = df[target_variable].mean()
        
        Y = df[target_variable]
        X = df[all_features]
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(Y, X_with_const).fit()
        
        # 4. Construct Scenario Point
        scenario_point = pd.Series(index=X_with_const.columns, dtype=float)
        scenario_point['const'] = 1.0
        
        held_constant_log = []
        
        for col in all_features:
            col_mean = X[col].mean()
            
            if col in changes_map:
                scenario_point[col] = float(changes_map[col])
            else:
                scenario_point[col] = col_mean
                held_constant_log.append(f"{col} (held at avg: {col_mean:,.2f})")
                
        # 5. Predict and Evaluate
        prediction_results = model.get_prediction(scenario_point)
        pred_df = prediction_results.summary_frame(alpha=1.0 - confidence_level)
        
        predicted_val = pred_df['mean'].values[0]
        ci_lower = pred_df['obs_ci_lower'].values[0]
        ci_upper = pred_df['obs_ci_upper'].values[0]
        diff_from_baseline = predicted_val - historical_target_mean
        
        # 6. Format Output
        result_text = f"--- Scenario Analysis for Target: '{target_variable}' ---\n\n"
            
        result_text += f"1. Baseline Context ({len(df)} observations analyzed):\n"
        result_text += f"  * Historical Average of {target_variable}: {historical_target_mean:,.2f}\n"
        result_text += f"  * Model R-Squared: {model.rsquared:.4f}\n\n"
        
        result_text += f"2. Scenario Conditions & Sensitivity:\n"
        for col, new_val in changes_map.items():
            hist_mean = X[col].mean()
            pct_change = ((new_val - hist_mean) / hist_mean) * 100 if hist_mean != 0 else 0
            coef_val = model.params.get(col, 0.0)
            
            result_text += f"  * CHANGED: '{col}' set to {new_val:,.2f}\n"
            result_text += f"    - Historical Avg: {hist_mean:,.2f} ({pct_change:+.1f}% change)\n"
            result_text += f"    - Marginal Impact (β): {coef_val:+.4f} {target_variable} per +1.0 unit of {col}\n"
            
        if held_constant_log:
            result_text += "\n  * HELD CONSTANT:\n    - " + "\n    - ".join(held_constant_log) + "\n\n"
            
        result_text += f"3. Scenario Prediction ({int(confidence_level*100)}% Confidence):\n"
        result_text += f"  * Expected {target_variable}: {predicted_val:,.2f}\n"
        result_text += f"  * Net Impact vs Baseline: {diff_from_baseline:+,.2f}\n"
        result_text += f"  * Interval: [{ci_lower:,.2f} to {ci_upper:,.2f}]\n"
        
        return {"text": result_text, "data": df, "model": model}
        
    except Exception as e:
        return {"text": f"Scenario Planning Error: {str(e)}", "data": None, "model": None}

@mlflow.trace(name="run_neural_network_tool")
def run_neural_network_tool(
    target_variable: str,
    feature_variables: List[str],
    task_type: str,
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    hidden_layer_sizes: List[int] = [100, 50],
    max_iter: int = 500,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Trains a scikit-learn MLPRegressor or MLPClassifier on the provided features.
    Features are one-hot encoded for categoricals and StandardScaler-normalized before
    training. Uses early stopping to avoid overfitting on small datasets.
    Returns the R² score (regression) or accuracy (classification) on the held-out
    test set, plus the fitted model and the cleaned DataFrame.
    """
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None, "model": None}
        elif TABLE_NAME:
            df = link_tables(TABLE_NAME, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None, "model": None}
            
        df_clean = df.dropna(subset=[target_variable] + feature_variables)
        X = df_clean[feature_variables]
        y = df_clean[target_variable]
        
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if task_type == 'regression':
            model = MLPRegressor(hidden_layer_sizes=tuple(hidden_layer_sizes), max_iter=max_iter, early_stopping=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            result_text = f"MLP Regression completed.\nTarget: {target_variable}\nR^2 Score on test set: {score:.4f}"
        else:
            model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layer_sizes), max_iter=max_iter, early_stopping=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            result_text = f"MLP Classification completed.\nTarget: {target_variable}\nAccuracy on test set: {score:.4f}"
            
        return {"text": result_text, "data": df_clean, "model": model}
        
    except Exception as e:
        return {"text": f"Neural Network Error: {e}\n{traceback.format_exc()}", "data": None, "model": None}

@mlflow.trace(name="run_optimization_tool")
def run_optimization_tool(
    objective_coefficients: List[float],
    inequality_constraints_matrix: Optional[List[List[float]]] = None,
    inequality_constraints_bounds: Optional[List[float]] = None,
    equality_constraints_matrix: Optional[List[List[float]]] = None,
    equality_constraints_bounds: Optional[List[float]] = None,
    bounds: Optional[List[List[Optional[float]]]] = None
) -> Dict[str, Any]:
    """
    Solves a linear programming problem using scipy.optimize.linprog (HiGHS solver).
    Minimizes c·x subject to A_ub·x ≤ b_ub, A_eq·x = b_eq, and per-variable bounds.
    To maximize instead of minimize, pass negative objective coefficients.
    Returns the optimal objective value and decision variable values on success,
    or the solver failure message on infeasibility.
    """
    try:
        formatted_bounds = None
        if bounds is not None:
            formatted_bounds = [(b[0], b[1]) if len(b) >= 2 else (None, None) for b in bounds]
            
        res = linprog(
            c=objective_coefficients,
            A_ub=inequality_constraints_matrix,
            b_ub=inequality_constraints_bounds,
            A_eq=equality_constraints_matrix,
            b_eq=equality_constraints_bounds,
            bounds=formatted_bounds,
            method='highs'
        )
        
        if res.success:
            result_text = f"Optimization Successful!\nOptimal Objective Value: {res.fun:.4f}\nOptimal Variables: {res.x}"
        else:
            result_text = f"Optimization Failed: {res.message}"
            
        return {"text": result_text, "data": None, "model": res}
        
    except Exception as e:
        return {"text": f"Optimization Error: {str(e)}", "data": None, "model": None}

@mlflow.trace(name="execute_python_tool")
def execute_python_tool(
    code: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Executes LLM-generated Python code in a restricted sandbox with access to
    pre-loaded pandas DataFrames (df), numpy (np), and pandas (pd).

    Single-table: df is a plain DataFrame. Multi-table: df is a list of DataFrames
    in the same order as TABLE_NAME, each loaded independently (no pre-join).

    The code can write results back via two reserved variables:
      - result_text (str): narrative output to surface to the LLM
      - result_df (DataFrame): tabular output to attach to the agent turn

    Blocks execution if the code references any forbidden modules or SQL mutations.
    Stdout is captured and included in the return text.
    """
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}
        elif TABLE_NAME:
            # Normalise to a list
            if isinstance(TABLE_NAME, str):
                table_list = [t.strip() for t in TABLE_NAME.split(",")] if "," in TABLE_NAME else [TABLE_NAME]
            else:
                table_list = list(TABLE_NAME)

            if len(table_list) == 1:
                # Single table: load normally, expose as plain `df`
                df = link_tables(table_list[0], limit=100000)
            else:
                # Multiple tables: load each independently to avoid a costly cross-join.
                # `df` is a list where df[i] corresponds to table_list[i].
                # The generated code can merge/join however it needs with pandas.
                df = [link_tables(t, limit=100000) for t in table_list]
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}
            
        forbidden_terms = ['os', 'sys', 'subprocess', 'shutil', 'databricks', 'pg8000', 'sqlalchemy', 'requests', 'urllib', 'eval(', 'exec(', 'open(', 'drop table', 'delete from', 'update ', 'insert into']
        code_lower = code.lower()
        for term in forbidden_terms:
            if term in code_lower:
                return {"text": f"Error: Python code contains forbidden term: '{term}'. Execution blocked for security.", "data": None}
                
        local_env = {
            'df': df.copy() if isinstance(df, pd.DataFrame) else df,
            'pd': pd,
            'np': np,
            'result_df': None,
            'result_text': None
        }
        
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, {"__builtins__": __builtins__}, local_env)
            
        output = stdout_buffer.getvalue()
        
        final_text = "Python Execution Successful.\n"
        if output:
            final_text += f"Console Output:\n{output}\n"
            
        if local_env.get('result_text'):
            final_text += f"Model Result Text:\n{local_env['result_text']}\n"
            
        return {
            "text": final_text, 
            "data": local_env.get('result_df') if isinstance(local_env.get('result_df'), pd.DataFrame) else (local_env.get('df') if isinstance(local_env.get('df'), pd.DataFrame) else None)
        }
        
    except Exception as e:
        return {"text": f"Python Execution Error: {e}", "data": None}