from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Project imports
from .base import run_sql_query, get_join_clause
from agent.memory import get_df_memory

YEARLY_WACC = 0.1
MONTHLY_WACC = (1 + YEARLY_WACC) ** (1 / 12) - 1

__all__ = [
    "link_tables",
    "execute_sql_query_tool",
    "run_ols_regression_tool",
    "run_arima_forecasting_tool",
    "run_random_forest_tool",
    "run_pca_tool",
    "run_kmeans_clustering_tool",
    "calculate_unit_economics_tool",
    "run_scenario_planning_tool"
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
    tables based on relationships defined in TABLE_RELATIONSHIPS in base.py.
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
                cond = get_join_clause(joined_t, next_table)
                if cond:
                    join_condition = cond
                    break
            
            if not join_condition:
                raise ValueError(
                    f"No join relationship defined in TABLE_RELATIONSHIPS between '{next_table}' "
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
    dataframe_id: Optional[str] = None
) -> dict:
    columns_to_fetch = [dependent_variable] + independent_variables
    
    try:
        if dataframe_id:
            df = get_df_memory().get_df(dataframe_id)
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
    

@mlflow.trace(name="run_arima_forecasting_tool")
def run_arima_forecasting_tool(
    value_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    aggregation: str = "SUM", 
    steps: int = 5, 
    p: int = 1, 
    d: int = 1, 
    q: int = 1
) -> dict:
    safe_value = '"{}"'.format(value_column.replace('"', ''))
    agg_func = aggregation.upper() if aggregation.upper() in ["SUM", "AVG", "COUNT"] else "SUM"
    
    try:
        if dataframe_id:
            df = get_df_memory().get_df(dataframe_id)
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}
            
            # Pandas fallback aggregation if it wasn't pre-aggregated
            val_col_clean = value_column.replace('"', '').strip()
            if 'Activation_Year' in df.columns and 'Activation_Month' in df.columns:
                if agg_func == "SUM":
                    df = df.groupby(['Activation_Year', 'Activation_Month'], as_index=False)[val_col_clean].sum()
                elif agg_func == "AVG":
                    df = df.groupby(['Activation_Year', 'Activation_Month'], as_index=False)[val_col_clean].mean()
                elif agg_func == "COUNT":
                    df = df.groupby(['Activation_Year', 'Activation_Month'], as_index=False)[val_col_clean].count()
                df = df.sort_values(by=['Activation_Year', 'Activation_Month'])
                df.rename(columns={val_col_clean: 'target_value'}, inplace=True)
            else:
                # Assume it's already a clean time series
                df['target_value'] = df[val_col_clean]
        
        elif TABLE_NAME:
            columns_to_fetch = [
                '"Activation_Year"', 
                '"Activation_Month"', 
                f'{agg_func}({safe_value}) AS target_value'
            ]
            df = link_tables(
                tables=TABLE_NAME,
                columns=columns_to_fetch,
                where_clause='"Activation_Year" IS NOT NULL AND "Activation_Month" IS NOT NULL',
                group_by=['Activation_Year', 'Activation_Month'],
                order_by='"Activation_Year" ASC, "Activation_Month" ASC',
                limit=None
            )
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}
            
        df['target_value'] = pd.to_numeric(df.get('target_value', pd.Series(dtype=float)), errors='coerce')
        df = df.dropna(subset=['target_value'])
        
        if df.empty or len(df) < 10:
            return {"text": "Error: Not enough historical data points (minimum 10 required) to perform ARIMA.", "data": None}
            
        series = df['target_value'].values
        
        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        
        result_text = f"ARIMA({p},{d},{q}) Forecasting Results for {agg_func} of {value_column}:\n"
        result_text += f"Based on {len(series)} periods of historical data, predictions for the next {steps} periods:\n"
        for i, val in enumerate(forecast, start=1):
            result_text += f"  • Period +{i}: {val:.4f}\n"
            
        return {"text": result_text, "data": model_fit}
        
    except Exception as e:
        return {"text": f"ARIMA Forecasting Error: {e}", "data": None}
    

@mlflow.trace(name="run_random_forest_tool")
def run_random_forest_tool(
    target_variable: str, 
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    task_type: str = "regression", 
    n_estimators: int = 100
) -> Dict[str, Any]:
    columns_to_fetch = [target_variable] + feature_variables
    
    try:
        if dataframe_id:
            df = get_df_memory().get_df(dataframe_id)
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = link_tables(TABLE_NAME, columns=columns_to_fetch, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}
            
        if df.empty or len(df) <= len(feature_variables):
            return {"text": "Error: Not enough data points.", "model": None}
            
        if task_type.lower() == "regression":
            df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')
            
        df = pd.get_dummies(df, columns=[col for col in feature_variables if col in df.columns and df[col].dtype == 'object'], drop_first=True)
        
        current_features = [col for col in df.columns if col != target_variable and col in feature_variables or '_' in col]
        df = df.dropna(subset=[target_variable] + current_features)
        
        if len(df) < 10:
            return {"text": "Error: Data size too small after cleaning to train a valid model.", "model": None}
            
        X = df[current_features]
        y = df[target_variable]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if task_type.lower() == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=7, min_samples_leaf=3, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            result_text = f"Random Forest Regression Results (n_estimators={n_estimators}):\n"
            result_text += f"Model Test R-squared: {r2_score(y_test, preds):.4f}\n"
            result_text += f"Model Test MSE: {mean_squared_error(y_test, preds):.4f}\n\n"
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=7, min_samples_leaf=3, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            result_text = f"Random Forest Classification Results (n_estimators={n_estimators}):\n"
            result_text += f"Model Test Accuracy: {accuracy_score(y_test, preds):.4f}\n"
            result_text += f"Classification Report:\n{classification_report(y_test, preds)}\n\n"
            
        importances = model.feature_importances_
        feat_imp = sorted(zip(current_features, importances), key=lambda x: x[1], reverse=True)
        
        result_text += "Feature Importances (higher is more impactful):\n"
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
    n_components: int = None
) -> Dict[str, Any]:
    try:
        if dataframe_id:
            df = get_df_memory().get_df(dataframe_id)
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
    n_clusters: int = 3
) -> Dict[str, Any]:
    try:
        if dataframe_id:
            df = get_df_memory().get_df(dataframe_id)
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
def calculate_unit_economics_tool(marketing_where_clause: str = None, acquisition_where_clause: str = None) -> dict:
    # Kept as-is since the schema explicitly handles the two-table where clauses without a TABLE_NAME arg.
    try:
        # 1. Marketing Data
        df_mkt = link_tables(
            tables='"sandbox"."dbs_marketing_spend_sync"',
            # Keep the quotes for Postgres, but drop the 'AS' alias
            columns=['"year"', '"month"', 'SUM("amount") AS total_spend'], 
            where_clause=marketing_where_clause,
            group_by=['"year"', '"month"'],
            limit=None
        )
        
        # Strip the quotes out of the resulting pandas column names if the driver leaves them in
        if not df_mkt.empty:
            df_mkt.columns = [col.replace('"', '') for col in df_mkt.columns]

        # 2. Acquisition Data
        df_acq = link_tables(
            tables='"sandbox"."acquisition_data_v3"',
            # Keep the quotes to protect the capital letters for Postgres
            columns=[
                '"Activation_Year"', 
                '"Activation_Month"', 
                'COUNT(*) AS total_activations', 
                'AVG("mcf") AS avg_mcf', 
                'AVG("Ve_Churn") AS avg_churn'
            ],
            where_clause=acquisition_where_clause,
            group_by=['"Activation_Year"', '"Activation_Month"'],
            limit=None
        )

        # Standardize the column names for the merge
        if not df_acq.empty:
            # First, clean any lingering double quotes from the column names
            df_acq.columns = [col.replace('"', '') for col in df_acq.columns]
            # Then rename to match df_mkt
            df_acq.rename(columns={
                'Activation_Year': 'year', 
                'Activation_Month': 'month'
            }, inplace=True)

        for df_tmp in [df_mkt, df_acq]:
            df_tmp['year'] = pd.to_numeric(df_tmp['year'], errors='coerce')
            df_tmp['month'] = pd.to_numeric(df_tmp['month'], errors='coerce')

        df_merged = pd.merge(df_mkt, df_acq, on=['year', 'month'], how='inner')

        if df_merged.empty:
            return {"text": "Error: Could not calculate CAC. No overlapping months found.", "data": None}

        df_merged['cpa'] = df_merged['total_spend'] / df_merged['total_activations']
        df_merged['clv'] = df_merged['avg_mcf'] / (MONTHLY_WACC + (df_merged['avg_churn'] / 100))
        df_merged['clv_cpa_ratio'] = df_merged['clv'] / df_merged['cpa']

        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_merged = df_merged.sort_values(by=['year', 'month'])

        df_merged['Date'] = pd.to_datetime(
            df_merged['year'].astype(int).astype(str) + '-' + 
            df_merged['month'].astype(int).astype(str) + '-01', 
            errors='coerce'
        )

        overall_spend = df_merged['total_spend'].sum()
        overall_acq = df_merged['total_activations'].sum()
        blended_cpa = overall_spend / overall_acq if overall_acq > 0 else 0
        avg_clv = df_merged['clv'].mean()
        clv_cpa = avg_clv / blended_cpa if blended_cpa > 0 else 0
        
        text_output = (
            f"Unit Economics Summary:\n"
            f"  • Total Marketing Spend Analyzed: ${overall_spend:,.2f}\n"
            f"  • Total Activations: {overall_acq:,.0f}\n"
            f"  • Blended CPA: ${blended_cpa:,.2f}\n"
            f"  • Average CLV (NPV): ${avg_clv:,.2f}\n"
            f"  • Blended CLV:CPA Ratio: {clv_cpa:.2f}x\n\n"
        )

        return {"text": text_output, "data": df_merged}

    except Exception as e:
        return {"text": f"Unit Economics Calculation Error: {e}", "data": None}
    

@mlflow.trace(name="run_scenario_planning_tool")
def run_scenario_planning_tool(
    target_variable: str, 
    feature_variables: list, 
    scenario_changes: list,
    TABLE_NAME: Optional[Union[str, List[str]]] = None, 
    dataframe_id: Optional[str] = None,
    confidence_level: float = 0.95,
    marketing_where_clause: Optional[str] = None,
    acquisition_where_clause: Optional[str] = None
) -> Dict[str, Any]:
    target_variable = str(target_variable).replace('"', '').replace("'", "").strip()
    feature_variables = [str(col).replace('"', '').replace("'", "").strip() for col in feature_variables]
    
    changes_map = {}
    for item in scenario_changes:
        col_name = item.get("column_name", "") if isinstance(item, dict) else getattr(item, "column_name", "")
        val = item.get("new_value", 0.0) if isinstance(item, dict) else getattr(item, "new_value", 0.0)
        clean_col = str(col_name).replace('"', '').replace("'", "").strip()
        changes_map[clean_col] = float(val)
        
    all_features = list(set(feature_variables + list(changes_map.keys())))
    
    try:
        if dataframe_id:
            df = get_df_memory().get_df(dataframe_id)
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None, "model": None}
        elif TABLE_NAME:
            is_multi_table = isinstance(TABLE_NAME, list) or (isinstance(TABLE_NAME, str) and "," in TABLE_NAME)
            
            if is_multi_table and any(t in str(TABLE_NAME).lower() for t in ["marketing", "acquisition"]):
                df_mkt = link_tables(
                    tables='"sandbox"."dbs_marketing_spend_sync"',
                    columns=['"year" AS year', '"month" AS month', 'SUM("amount") AS total_marketing_spend', 'AVG("amount") AS avg_transaction_spend', 'COUNT(*) AS marketing_transactions'],
                    where_clause=marketing_where_clause,
                    group_by=['year', 'month'],
                    limit=None
                )
                df_acq = link_tables(
                    tables='"sandbox"."acquisition_data_v3"',
                    columns=['"Activation_Year" AS year', '"Activation_Month" AS month', 'COUNT(*) AS total_activations', 'AVG("mcf") AS avg_mcf', 'AVG("Ve_Churn") AS avg_churn', 'SUM("mcf") AS total_mcf'],
                    where_clause=acquisition_where_clause,
                    group_by=['Activation_Year', 'Activation_Month'],
                    limit=None
                )
                
                if df_mkt.empty or df_acq.empty:
                    return {"text": "Error: One or both tables returned no data.", "data": None, "model": None}
                
                for df_tmp in [df_mkt, df_acq]:
                    df_tmp['year'] = pd.to_numeric(df_tmp['year'], errors='coerce')
                    df_tmp['month'] = pd.to_numeric(df_tmp['month'], errors='coerce')
                    
                df = pd.merge(df_mkt, df_acq, on=['year', 'month'], how='inner')
                df = df.sort_values(by=['year', 'month']).reset_index(drop=True)
                
            else:
                combined_where = " AND ".join(filter(None, [marketing_where_clause, acquisition_where_clause]))
                df = link_tables(
                    tables=TABLE_NAME, 
                    columns=[target_variable] + all_features, 
                    where_clause=combined_where if combined_where else None,
                    random_order=True, 
                    limit=100000
                )
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None, "model": None}
            
        missing_cols = [col for col in [target_variable] + all_features if col not in df.columns]
        if missing_cols:
            return {
                "text": f"Error: Missing required columns: {missing_cols}. Available: {df.columns.tolist()}", 
                "data": None, 
                "model": None
            }

        for col in [target_variable] + all_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[target_variable] + all_features)
        
        if df.empty or len(df) <= len(all_features) + 3:
            return {"text": "Error: Not enough data points to build a reliable scenario model.", "data": None, "model": None}
            
        historical_target_mean = df[target_variable].mean()
        
        Y = df[target_variable]
        X = df[all_features]
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(Y, X_with_const).fit()
        
        scenario_point = pd.Series(index=X_with_const.columns, dtype=float)
        scenario_point['const'] = 1.0
        
        held_constant_log = []
        elasticity_log = []
        
        for col in all_features:
            col_mean = X[col].mean()
            coef = model.params.get(col, 0.0)
            
            if col in changes_map:
                scenario_point[col] = float(changes_map[col])
                if col_mean != 0 and historical_target_mean != 0:
                    elasticity = (coef * col_mean) / historical_target_mean
                    elasticity_log.append((col, coef, elasticity))
            else:
                scenario_point[col] = col_mean
                held_constant_log.append(f"{col} (held at avg: {col_mean:,.2f})")
                
        prediction_results = model.get_prediction(scenario_point)
        pred_df = prediction_results.summary_frame(alpha=1.0 - confidence_level)
        
        predicted_val = pred_df['mean'].values[0]
        ci_lower = pred_df['obs_ci_lower'].values[0]
        ci_upper = pred_df['obs_ci_upper'].values[0]
        diff_from_baseline = predicted_val - historical_target_mean
        
        result_text = f"--- Scenario Analysis for Target: '{target_variable}' ---\n\n"
            
        result_text += f"1. Baseline Context ({len(df)} observations analyzed):\n"
        result_text += f"  • Historical Average of {target_variable}: {historical_target_mean:,.2f}\n"
        result_text += f"  • Model R-Squared: {model.rsquared:.4f}\n\n"
        
        result_text += f"2. Scenario Conditions & Sensitivity:\n"
        for col, new_val in changes_map.items():
            hist_mean = X[col].mean()
            pct_change = ((new_val - hist_mean) / hist_mean) * 100 if hist_mean != 0 else 0
            corr = df[col].corr(df[target_variable])
            
            coef_val = model.params.get(col, 0.0)
            elast_val = next((e[2] for e in elasticity_log if e[0] == col), 0.0)
            
            result_text += f"  • CHANGED: '{col}' set to {new_val:,.2f}\n"
            result_text += f"    - Historical Avg: {hist_mean:,.2f} ({pct_change:+.1f}% change)\n"
            result_text += f"    - Marginal Impact (β): {coef_val:+.4f} {target_variable} per +1.0 unit of {col}\n"
            
        if held_constant_log:
            result_text += "\n  • HELD CONSTANT:\n    - " + "\n    - ".join(held_constant_log) + "\n\n"
            
        result_text += f"3. Scenario Prediction ({int(confidence_level*100)}% Confidence):\n"
        result_text += f"  • Expected {target_variable}: {predicted_val:,.2f}\n"
        result_text += f"  • Net Impact vs Baseline: {diff_from_baseline:+,.2f}\n"
        result_text += f"  • Interval: [{ci_lower:,.2f} to {ci_upper:,.2f}]\n"
        
        return {"text": result_text, "data": df, "model": model}
        
    except Exception as e:
        return {"text": f"Scenario Planning Error: {str(e)}", "data": None, "model": None}