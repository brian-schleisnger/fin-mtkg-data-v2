from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
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

from .base import run_sql_query

YEARLY_WACC = 0.1
MONTHLY_WACC = (1 + YEARLY_WACC) ** (1 / 12) - 1

__all__ = [
    "execute_sql_query_tool",
    "run_ols_regression_tool",
    "run_arima_forecasting_tool",
    "run_random_forest_tool",
    "run_pca_tool",
    "run_kmeans_clustering_tool",
    "calculate_unit_economics_tool"
]

@mlflow.trace(name="execute_sql_query")
def execute_sql_query_tool(sql_query: str) -> dict:
    """Executes SQL directly. Orchestrator now handles the retry logic."""
    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: Query executed successfully, but returned 0 rows.", "data": None}
        
        # Limit rows converted to text to prevent blowing up the LLM context window
        csv_text = df.head(100).to_csv(index=False)
        return {"text": f"Success. Showing top 100 rows:\n{csv_text}", "data": df}
        
    except Exception as e:
        return {"text": f"Error executing SQL: {str(e)}", "data": None}
    
@mlflow.trace(name="run_ols_regression_tool")
def run_ols_regression_tool(TABLE_NAME, dependent_variable: str, independent_variables: list) -> dict:
    """
    Sub-agent tool: Fetches specific numerical columns and runs an OLS multiple regression.
    """
    columns_to_fetch = [dependent_variable] + independent_variables
    # NEW: Strip any existing quotes the LLM might have added, then forcefully wrap in double quotes
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in columns_to_fetch]
    columns_str = ", ".join(safe_columns)
    
    # We query more rows here than the SQL tool to ensure a valid sample size for regression
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    
    try:
        # Fetch the data using your existing helper
        df = run_sql_query(sql_query)
        
        # Drop rows with missing values to prevent the regression from crashing
        df = df.dropna(subset=columns_to_fetch)
        
        # Ensure we have enough data points to run a valid regression
        if df.empty or len(df) <= len(independent_variables):
            return "Error: Not enough valid data points to perform regression."
            
        # 2. Define target (Y) and features (X)
        Y = pd.to_numeric(df[dependent_variable])
        
        # Convert all independent variables to numeric, coercing errors to NaN, then drop again if needed
        X = df[independent_variables].apply(pd.to_numeric, errors='coerce')
        
        # Add a constant (intercept) to the model, which is required for standard OLS
        X = sm.add_constant(X)
        
        # 3. Fit the OLS model
        model = sm.OLS(Y, X).fit()
        
        # 4. Return the statistical summary as a string for the LLM to interpret
        return {"text": model.summary().as_text(), "data": model}
        
    except Exception as e:
        return f"Regression Error: {e}"
    
@mlflow.trace(name="run_arima_forecasting_tool")
def run_arima_forecasting_tool(TABLE_NAME, value_column: str, aggregation: str = "SUM", steps: int = 5, p: int = 1, d: int = 1, q: int = 1) -> dict:
    """
    Sub-agent tool: Fetches historical data grouped by Activation_Year and Activation_Month, 
    and forecasts future periods using an ARIMA model.
    """
    safe_value = '"{}"'.format(value_column.replace('"', ''))
    
    # Protect against SQL injection and ensure a valid aggregation function
    agg_func = aggregation.upper() if aggregation.upper() in ["SUM", "AVG", "COUNT"] else "SUM"
    
    # Group by Year and Month, ensuring chronological order for the time series
    sql_query = f"""
        SELECT 
            "Activation_Year", 
            "Activation_Month", 
            {agg_func}({safe_value}) AS target_value 
        FROM {TABLE_NAME} 
        WHERE "Activation_Year" IS NOT NULL AND "Activation_Month" IS NOT NULL
        GROUP BY "Activation_Year", "Activation_Month" 
        ORDER BY "Activation_Year" ASC, "Activation_Month" ASC
    """
    
    try:
        df = run_sql_query(sql_query)
        
        # Clean the target values
        df['target_value'] = pd.to_numeric(df['target_value'], errors='coerce')
        df = df.dropna(subset=['target_value'])
        
        if df.empty or len(df) < 10:
            return "Error: Not enough historical monthly data points (minimum 10 required) to perform ARIMA forecasting."
            
        # Parse series data (already ordered chronologically by SQL)
        series = df['target_value'].values
        
        # Fit ARIMA model
        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
        
        # Generate future forecasts
        forecast = model_fit.forecast(steps=steps)
        
        # Build text summary for the LLM
        result_text = f"ARIMA({p},{d},{q}) Forecasting Results for {agg_func} of {value_column}:\n"
        result_text += f"Based on {len(series)} months of historical data, here are the predictions for the next {steps} months:\n"
        
        for i, val in enumerate(forecast, start=1):
            result_text += f"  • Month +{i}: {val:.4f}\n"
            
        return {"text": result_text, "data": model_fit}
        
    except Exception as e:
        return f"ARIMA Forecasting Error: {e}"
    
@mlflow.trace(name="run_random_forest_tool")
def run_random_forest_tool(TABLE_NAME, target_variable: str, feature_variables: list, task_type: str = "regression", n_estimators: int = 100) -> Dict[str, Any]:
    """
    Sub-agent tool: Fetches columns, preprocesses data, and runs a Random Forest model.
    Returns a dictionary containing the LLM-readable text result and the trained model object.
    """
    columns_to_fetch = [target_variable] + feature_variables
    
    # TODO: Add explicit validation of `columns_to_fetch` against your database schema here
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in columns_to_fetch]
    columns_str = ", ".join(safe_columns)
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 100000"
    
    try:
        df = run_sql_query(sql_query)
        
        if df.empty or len(df) <= len(feature_variables):
            return {"text": "Error: Not enough data points.", "model": None}
            
        # Coerce target to numeric if regression
        if task_type.lower() == "regression":
            df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')
            
        # Handle categorical features by creating dummy variables (One-Hot Encoding)
        # We don't blindly coerce features to numeric; we let pd.get_dummies handle strings
        df = pd.get_dummies(df, columns=[col for col in feature_variables if df[col].dtype == 'object'], drop_first=True)
        
        # Update feature variables list after dummy creation
        current_features = [col for col in df.columns if col != target_variable]
        
        # Single, clean dropna step
        df = df.dropna(subset=[target_variable] + current_features)
        
        if len(df) < 10:
            return {"text": "Error: Data size too small after cleaning to train a valid model.", "model": None}
            
        X = df[current_features]
        y = df[target_variable]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model with sensible regularization to prevent severe overfitting
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
            
        # Feature importances
        importances = model.feature_importances_
        feat_imp = sorted(zip(current_features, importances), key=lambda x: x[1], reverse=True)
        
        result_text += "Feature Importances (higher is more impactful):\n"
        # Only show top 10 if there are many dummy variables to save LLM context window
        for feat, imp in feat_imp[:10]:
            result_text += f"  • {feat}: {imp:.4f}\n"
            
        return {"text": result_text, "model": model}
        
    except Exception as e:
        # Consider logging the full traceback here for debugging
        return {"text": f"Random Forest Error: {e}", "model": None}
    
@mlflow.trace(name="run_pca_tool")
def run_pca_tool(TABLE_NAME, feature_variables: list, n_components: int = None) -> Dict[str, Any]:
    """
    Sub-agent tool: Fetches columns, standardizes data, and runs Principal Component Analysis (PCA).
    Returns a dictionary containing the LLM-readable text result and the trained PCA object.
    """
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in feature_variables]
    columns_str = ", ".join(safe_columns)
    
    # Using the same row limit strategy as the Random Forest tool
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 100000"
    
    try:
        df = run_sql_query(sql_query)
        
        if df.empty or len(df) < 2:
            return {"text": "Error: Not enough data points fetched to perform PCA.", "model": None}
            
        # 1. Handle categorical features by creating dummy variables
        df = pd.get_dummies(df, columns=[col for col in feature_variables if df[col].dtype == 'object'], drop_first=True)
        
        # 2. Drop any rows with missing values
        df = df.dropna()
        
        current_features = df.columns.tolist()
        
        if len(df) < 2 or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning and encoding to perform PCA.", "model": None}
        
        # 3. Standardize the columns (Crucial for PCA so large magnitude features don't dominate)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Cap n_components to the maximum mathematically possible if the LLM requests too many
        max_components = min(len(df), len(current_features))
        if n_components is None or n_components > max_components:
            actual_components = max_components
        else:
            actual_components = n_components
            
        # 4. Fit PCA
        pca = PCA(n_components=actual_components)
        pca.fit(scaled_data)
        
        # 5. Build text summary for the LLM
        result_text = f"PCA Results (n_components={actual_components}):\n"
        explained_variance = pca.explained_variance_ratio_
        
        result_text += "Explained Variance Ratio per Component:\n"
        for i, var in enumerate(explained_variance):
            result_text += f"  • PC{i+1}: {var:.4f} ({(var*100):.1f}%)\n"
        result_text += f"Total Explained Variance: {sum(explained_variance):.4f} ({(sum(explained_variance)*100):.1f}%)\n\n"
        
        # Extract feature loadings to explain *what* makes up each component
        # We limit the output to the top 2 components and only significant loadings to save LLM context
        components_to_show = min(2, actual_components)
        result_text += "Top Feature Loadings (absolute magnitude > 0.3) for primary components:\n"
        
        for i in range(components_to_show):
            result_text += f"  PC{i+1} Signficant Loadings:\n"
            loadings = pca.components_[i]
            
            # Match loadings to feature names and sort by absolute impact
            feat_loadings = sorted(zip(current_features, loadings), key=lambda x: abs(x[1]), reverse=True)
            for feat, load in feat_loadings:
                if abs(load) > 0.3:
                    result_text += f"    - {feat}: {load:.4f}\n"
        
        return {"text": result_text, "model": pca}
        
    except Exception as e:
        return {"text": f"PCA Error: {e}", "model": None}
    
@mlflow.trace(name="run_kmeans_clustering_tool")
def run_kmeans_clustering_tool(TABLE_NAME, feature_variables: list, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Sub-agent tool: Fetches columns, standardizes data, and runs K-Means clustering.
    Returns a dictionary containing the LLM-readable text result and the trained KMeans object.
    """
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in feature_variables]
    columns_str = ", ".join(safe_columns)
    
    # Using the same row limit strategy as the Random Forest and PCA tools
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 100000"
    
    try:
        df = run_sql_query(sql_query)
        
        if df.empty or len(df) < n_clusters:
            return {"text": f"Error: Not enough data points fetched to perform {n_clusters}-means clustering.", "model": None}
            
        # 1. Handle categorical features by creating dummy variables
        df = pd.get_dummies(df, columns=[col for col in feature_variables if df[col].dtype == 'object'], drop_first=True)
        
        # 2. Drop any rows with missing values
        df = df.dropna()
        current_features = df.columns.tolist()
        
        if len(df) < n_clusters or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning and encoding to perform clustering.", "model": None}
        
        # 3. Standardize the columns (Crucial for distance-based clustering algorithms)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # 4. Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(scaled_data)
        
        # 5. Build text summary for the LLM
        df['Cluster'] = kmeans.labels_
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        result_text = f"K-Means Clustering Results (n_clusters={n_clusters}):\n"
        result_text += "Cluster Population Sizes:\n"
        for cluster_id, count in cluster_counts.items():
            result_text += f"  • Cluster {cluster_id}: {count} data points\n"
            
        result_text += "\nCluster Profiles (Standardized Centroids):\n"
        result_text += "Note: Values > 0 mean the cluster is above the dataset average for that feature, < 0 means below average.\n"
        
        centroids = kmeans.cluster_centers_
        
        # Limit to the most defining features per cluster to save LLM context
        for i in range(n_clusters):
            result_text += f"  Cluster {i} Defining Features (Top 5):\n"
            
            # Match standardized centroid values to feature names and sort by absolute magnitude
            feat_centroids = sorted(zip(current_features, centroids[i]), key=lambda x: abs(x[1]), reverse=True)
            
            for feat, val in feat_centroids[:5]:
                # Only show features that have a somewhat distinct deviation from the mean
                if abs(val) > 0.15: 
                    result_text += f"    - {feat}: {val:.4f}\n"
                    
        return {"text": result_text, "model": kmeans}
        
    except Exception as e:
        return {"text": f"K-Means Error: {e}", "model": None}
    
@mlflow.trace(name="calculate_unit_economics_tool")
def calculate_unit_economics_tool(marketing_where_clause: str = None, acquisition_where_clause: str = None) -> dict:
    """
    Sub-agent tool: Calculates Marketing cost per acquisition (CPA) and est. CLV:cpa ratios 
    by aggregating marketing spend and acquisition volumes at a monthly grain, 
    safely merged via Pandas.
    """
    # 1. Fetch Marketing Spend
    mkt_query = """
        SELECT 
            "year" AS year,
            "month" AS month,
            SUM("amount") AS total_spend
        FROM "sandbox"."dbs_marketing_spend_sync"
    """
    if marketing_where_clause:
        mkt_query += f" WHERE {marketing_where_clause}"
    mkt_query += ' GROUP BY "year", "month"'

    # 2. Fetch Acquisition Data (Activations & Net Present Value)
    acq_query = """
        SELECT 
            "Activation_Year" AS year,
            "Activation_Month" AS month,
            COUNT(*) AS total_activations,
            AVG("mcf") AS avg_mcf,
            AVG("Ve_Churn") AS avg_churn
        FROM "sandbox"."acquisition_data_v3"
    """
    if acquisition_where_clause:
        acq_query += f" WHERE {acquisition_where_clause}"
    acq_query += ' GROUP BY "Activation_Year", "Activation_Month"'

    try:
        df_mkt = run_sql_query(mkt_query)
        df_acq = run_sql_query(acq_query)

        if df_mkt.empty or df_acq.empty:
            return {"text": "Error: One or both tables returned no data for the specified filters.", "data": None}

        # 3. Clean and Merge safely in Pandas
        df_mkt['year'] = pd.to_numeric(df_mkt['year'], errors='coerce')
        df_mkt['month'] = pd.to_numeric(df_mkt['month'], errors='coerce')
        df_acq['year'] = pd.to_numeric(df_acq['year'], errors='coerce')
        df_acq['month'] = pd.to_numeric(df_acq['month'], errors='coerce')

        df_merged = pd.merge(df_mkt, df_acq, on=['year', 'month'], how='inner')

        if df_merged.empty:
            return {"text": "Error: Could not calculate CAC. No overlapping months found between the two datasets.", "data": None}

        # 4. Calculate Unit Economics
        df_merged['cpa'] = df_merged['total_spend'] / df_merged['total_activations']
        df_merged['clv'] = df_merged['avg_mcf'] /(MONTHLY_WACC + (df_merged['avg_churn']/100))
        df_merged['clv_cpa_ratio'] = df_merged['clv'] / df_merged['cpa']

        # Clean up infinities if there were months with zero activations
        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Sort chronologically for the time series
        df_merged = df_merged.sort_values(by=['year', 'month'])

        # 5. Build Interactive Trend Chart
        df_merged['Date'] = pd.to_datetime(
            df_merged['year'].astype(int).astype(str) + '-' + 
            df_merged['month'].astype(int).astype(str) + '-01', 
            errors='coerce'
        )

        # 6. Generate Business Summary for the LLM
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
            f"Note: Data is grouped by month. See the attached dataframe and chart for trend lines."
        )

        return {"text": text_output, "data": df_merged}

    except Exception as e:
        return {"text": f"Unit Economics Calculation Error: {e}", "data": None}
    

@mlflow.trace(name="run_scenario_planning_tool")
def run_scenario_planning_tool(
    TABLE_NAME: str, 
    target_variable: str, 
    feature_variables: list, 
    scenario_changes: list, 
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Sub-agent tool: Simulates what-if scenarios using OLS regression.
    Changes specified variables to hypothetical values while holding all other features at their historical mean.
    Returns point predictions, confidence intervals, and Pearson correlations.
    """
    # 0. Convert List[dict] (from Pydantic model_dump) into a simple {col: val} mapping
    changes_map = {
        item["column_name"]: float(item["new_value"]) if isinstance(item, dict) else float(item.new_value)
        for item in scenario_changes
    }
    
    # Ensure any variables in changes_map are included in our feature list
    all_features = list(set(feature_variables + list(changes_map.keys())))
    columns_to_fetch = [target_variable] + all_features
    
    safe_columns = ['"{}"'.format(col.replace('"', '')) for col in columns_to_fetch]
    columns_str = ", ".join(safe_columns)
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 100000"
    
    try:
        df = run_sql_query(sql_query)
        
        # Clean data: coerce to numeric and drop NAs
        for col in columns_to_fetch:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=columns_to_fetch)
        
        if df.empty or len(df) <= len(all_features) + 5:
            return {"text": "Error: Not enough valid historical data points to build a reliable scenario model.", "data": None, "model": None}
        
        # 1. Calculate historical baseline and correlations
        historical_target_mean = df[target_variable].mean()
        
        correlations = {}
        for col in changes_map.keys():
            if col in df.columns:
                correlations[col] = df[col].corr(df[target_variable])
                
        # 2. Fit the Multiple Linear Regression (OLS) Model
        Y = df[target_variable]
        X = df[all_features]
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(Y, X_with_const).fit()
        
        # 3. Construct the Scenario Data Point (Hold unmentioned variables at historical mean)
        scenario_point = pd.Series(index=X_with_const.columns, dtype=float)
        scenario_point['const'] = 1.0  # Intercept
        
        held_constant_log = []
        for col in all_features:
            if col in changes_map:
                scenario_point[col] = float(changes_map[col])
            else:
                col_mean = X[col].mean()
                scenario_point[col] = col_mean
                held_constant_log.append(f"{col} (held at avg: {col_mean:.4f})")
                
        # 4. Predict and generate Confidence/Prediction Intervals
        prediction_results = model.get_prediction(scenario_point)
        pred_df = prediction_results.summary_frame(alpha=1.0 - confidence_level)
        
        predicted_val = pred_df['mean'].values[0]
        ci_lower = pred_df['obs_ci_lower'].values[0]
        ci_upper = pred_df['obs_ci_upper'].values[0]
        
        # 5. Build LLM-Friendly Summary Text
        result_text = f"--- Scenario Planning Results for Target: '{target_variable}' ---\n\n"
        result_text += f"1. Baseline Context:\n"
        result_text += f"  • Historical Average of {target_variable}: {historical_target_mean:.4f}\n"
        result_text += f"  • Model R-Squared (Overall Fit): {model.rsquared:.4f}\n\n"
        
        result_text += f"2. Scenario Conditions:\n"
        for col, new_val in changes_map.items():
            hist_mean = X[col].mean()
            pct_change = ((new_val - hist_mean) / hist_mean) * 100 if hist_mean != 0 else 0
            corr_str = f"Pearson r = {correlations.get(col, 0):.2f}"
            result_text += f"  • CHANGED: '{col}' set to {new_val:,.4f} (Historical Avg: {hist_mean:,.4f} | Change: {pct_change:+.1f}% | Correlation with target: {corr_str})\n"
            
        if held_constant_log:
            result_text += "  • HELD CONSTANT:\n    - " + "\n    - ".join(held_constant_log) + "\n\n"
        else:
            result_text += "\n"
            
        result_text += f"3. Prediction & Confidence Interval ({int(confidence_level*100)}% Confidence):\n"
        result_text += f"  • Expected {target_variable}: {predicted_val:,.4f}\n"
        result_text += f"  • Prediction Interval: [{ci_lower:,.4f} to {ci_upper:,.4f}]\n"
        
        diff_from_baseline = predicted_val - historical_target_mean
        result_text += f"  • Net Impact vs Historical Average: {diff_from_baseline:+,,.4f}\n\n"
        
        result_text += "Statistical Interpretation for the User:\n"
        result_text += f"Based on the historical correlation and regression coefficients, changing the specified variables while holding others constant is projected to move {target_variable} to approximately {predicted_val:,.2f}. "
        result_text += f"Because relationships are not exact, we can be {int(confidence_level*100)}% confident that the true outcome will fall between {ci_lower:,.2f} and {ci_upper:,.2f} given normal historical variance."

        return {"text": result_text, "data": pred_df, "model": model}
        
    except Exception as e:
        return {"text": f"Scenario Planning Error: {str(e)}", "data": None, "model": None}