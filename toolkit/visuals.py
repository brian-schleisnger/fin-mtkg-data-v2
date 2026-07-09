from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Leverage the centralized join engine from analytics.py
from .analytics import link_tables
from .base import run_sql_query

__all__ = [
    "generate_scatterplot_tool",
    "generate_barchart_tool",
    "generate_histogram_tool",
    "generate_linechart_tool",
    "compare_monthly_metrics_tool"
]


# ─── Helper Functions ────────────────────────────────────────────
def _resolve_column(requested_col: str, df: pd.DataFrame) -> str:
    """
    Safely resolves column names in Pandas DataFrames after SQL execution.
    Handles stripped quotes, table prefixes (table.col vs col), and case sensitivity.
    """
    if not requested_col or df.empty:
        return requested_col
        
    clean_req = str(requested_col).replace('"', '').replace("'", "").strip()
    
    # 1. Exact match
    if clean_req in df.columns:
        return clean_req
        
    # 2. Match without table prefix (e.g., 'table.col' -> 'col')
    simple_req = clean_req.split('.')[-1]
    for col in df.columns:
        if str(col).split('.')[-1] == simple_req:
            return str(col)
            
    # 3. Case-insensitive match
    for col in df.columns:
        if str(col).split('.')[-1].lower() == simple_req.lower():
            return str(col)
            
    return clean_req


def _aggregate_if_needed(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    cat_col: Optional[str] = None, 
    agg_func: str = "SUM"
) -> pd.DataFrame:
    """
    Ensures bar and line charts receive aggregated data (1 row per X/Category coordinate)
    to prevent stacked slivers or zigzagging spaghetti lines.
    """
    if df.empty or not agg_func or agg_func.upper() == "NONE":
        return df
        
    group_cols = [x_col]
    if cat_col and cat_col in df.columns:
        group_cols.append(cat_col)
        
    # Check if duplicates exist across the X/Category dimensions
    if df.duplicated(subset=group_cols).any():
        func = agg_func.lower() if agg_func.lower() in ["sum", "mean", "count", "min", "max"] else "sum"
        if func == "count":
            df = df.groupby(group_cols, as_index=False)[y_col].count()
        else:
            df = df.groupby(group_cols, as_index=False)[y_col].agg(func)
            
    return df


# ─── Plotting Functions ────────────────────────────────────────────
@mlflow.trace(name="generate_scatterplot_tool")
def generate_scatterplot_tool(
    TABLE_NAME: Union[str, List[str]], 
    x_column: str, 
    y_column: str, 
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None, 
    include_trendline: bool = False
) -> Dict[str, Any]:
    """Sub-agent tool: Generates a robust scatterplot across single or joined tables."""
    cols_to_fetch = [x_column, y_column]
    if category_column:
        cols_to_fetch.append(category_column)

    try:
        df = link_tables(TABLE_NAME, columns=cols_to_fetch, where_clause=where_clause, random_order=True, limit=10000)
        if df.empty:
            return {"text": "Error: No data returned. Check filters or table relationships.", "data": None, "figure": None}

        x_col = _resolve_column(x_column, df)
        y_col = _resolve_column(y_column, df)
        cat_col = _resolve_column(category_column, df) if category_column else None

        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])

        if len(df) < 2:
            return {"text": "Error: Not enough valid numeric data points to generate scatterplot.", "data": None, "figure": None}

        t_line = "ols" if include_trendline and len(df) > 3 else None
        
        kwargs = {
            "x": x_col, "y": y_col, "trendline": t_line,
            "title": f"Scatterplot: {y_col} vs {x_col}",
            "template": "plotly_white", "opacity": 0.75
        }
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col

        fig = px.scatter(df, **kwargs)
        fig.update_layout(hovermode="closest", margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated scatterplot for {y_col} vs {x_col} ({len(df)} points).", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Scatterplot Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="generate_barchart_tool")
def generate_barchart_tool(
    TABLE_NAME: Union[str, List[str]], 
    x_column: str, 
    y_column: str, 
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None,
    aggregation: str = "SUM"
) -> Dict[str, Any]:
    """Sub-agent tool: Generates an aggregated bar chart across single or joined tables."""
    cols_to_fetch = [x_column, y_column]
    if category_column:
        cols_to_fetch.append(category_column)

    try:
        df = link_tables(TABLE_NAME, columns=cols_to_fetch, where_clause=where_clause, limit=50000)
        if df.empty:
            return {"text": "Error: No data returned for bar chart.", "data": None, "figure": None}

        x_col = _resolve_column(x_column, df)
        y_col = _resolve_column(y_column, df)
        cat_col = _resolve_column(category_column, df) if category_column else None

        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])

        # Automatically group and aggregate to prevent sliver bars
        df = _aggregate_if_needed(df, x_col, y_col, cat_col, agg_func=aggregation)
        
        # Sort X-axis naturally if possible, otherwise by Y value descending
        try:
            df = df.sort_values(by=x_col, ascending=True)
        except Exception:
            df = df.sort_values(by=y_col, ascending=False)

        kwargs = {
            "x": x_col, "y": y_col,
            "title": f"{aggregation.upper()} of {y_col} by {x_col}",
            "template": "plotly_white"
        }
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col
            kwargs["barmode"] = "group"

        fig = px.bar(df, **kwargs)
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated bar chart ({len(df)} aggregated bars).", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Bar Chart Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="generate_histogram_tool")
def generate_histogram_tool(
    TABLE_NAME: Union[str, List[str]], 
    x_column: str, 
    n_bins: Optional[int] = None, 
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None
) -> Dict[str, Any]:
    """Sub-agent tool: Generates a distribution histogram with statistical box marginals."""
    cols_to_fetch = [x_column]
    if category_column:
        cols_to_fetch.append(category_column)

    try:
        df = link_tables(TABLE_NAME, columns=cols_to_fetch, where_clause=where_clause, random_order=True, limit=50000)
        if df.empty:
            return {"text": "Error: No data returned for histogram.", "data": None, "figure": None}

        x_col = _resolve_column(x_column, df)
        cat_col = _resolve_column(category_column, df) if category_column else None

        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df = df.dropna(subset=[x_col])

        kwargs = {
            "x": x_col, "title": f"Distribution of {x_col}",
            "template": "plotly_white", "marginal": "box", # Adds executive box plot above histogram
            "opacity": 0.8
        }
        if n_bins:
            kwargs["nbins"] = n_bins
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col
            kwargs["barmode"] = "overlay"

        fig = px.histogram(df, **kwargs)
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated histogram for {x_col} across {len(df)} records.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Histogram Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="generate_linechart_tool")
def generate_linechart_tool(
    TABLE_NAME: Union[str, List[str]], 
    x_column: str, 
    y_column: str, 
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None,
    aggregation: str = "SUM"
) -> Dict[str, Any]:
    """Sub-agent tool: Generates a clean, sorted time-series or sequential line chart."""
    cols_to_fetch = [x_column, y_column]
    if category_column:
        cols_to_fetch.append(category_column)

    try:
        df = link_tables(TABLE_NAME, columns=cols_to_fetch, where_clause=where_clause, limit=50000)
        if df.empty:
            return {"text": "Error: No data returned for line chart.", "data": None, "figure": None}

        x_col = _resolve_column(x_column, df)
        y_col = _resolve_column(y_column, df)
        cat_col = _resolve_column(category_column, df) if category_column else None

        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])

        # Group and aggregate to eliminate zigzagging spaghetti lines
        df = _aggregate_if_needed(df, x_col, y_col, cat_col, agg_func=aggregation)
        
        # Mandatory sort by X-axis for chronological sequence
        df = df.sort_values(by=x_col, ascending=True)

        kwargs = {
            "x": x_col, "y": y_col, "markers": True,
            "title": f"Trend of {aggregation.upper()}({y_col}) over {x_col}",
            "template": "plotly_white"
        }
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col

        fig = px.line(df, **kwargs)
        fig.update_layout(hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated line chart over {len(df)} intervals.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Line Chart Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="compare_monthly_metrics_tool")
def compare_monthly_metrics_tool(
    marketing_metric: str = "amount", 
    acquisition_metric_func: str = "COUNT", 
    acquisition_column: str = "*"
) -> Dict[str, Any]:
    """Aggregates acquisition data and joins with marketing spend to generate a dual-metric trend line."""
    marketing_clean = marketing_metric.replace('"', '').split('.')[-1]
    acq_clean = acquisition_column.replace('"', '').split('.')[-1] if acquisition_column != "*" else "*"
    func_clean = acquisition_metric_func.upper() if acquisition_metric_func.upper() in ["COUNT", "SUM", "AVG"] else "COUNT"

    sql_query = f"""
        SELECT 
            m."year" AS year,
            m."month" AS month,
            SUM(m."{marketing_clean}") AS marketing_total,
            {func_clean}(a."{acq_clean}") AS acquisition_total
        FROM "sandbox"."dbs_marketing_spend_sync" m
        LEFT JOIN "sandbox"."acquisition_data_v3" a 
            ON m."year" = a."Activation_Year" AND m."month" = a."Activation_Month"
        GROUP BY m."year", m."month"
        ORDER BY m."year" ASC, m."month" ASC
    """

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for monthly metric comparison.", "data": None, "figure": None}

        df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01', errors='coerce')
        df = df.dropna(subset=['Date', 'marketing_total', 'acquisition_total']).sort_values(by='Date')

        df_melted = df.melt(
            id_vars=['Date'], 
            value_vars=['marketing_total', 'acquisition_total'], 
            var_name='Metric', 
            value_name='Value'
        )
        df_melted['Metric'] = df_melted['Metric'].replace({
            'marketing_total': f'Marketing Spend ({marketing_clean})',
            'acquisition_total': f'Acquisitions ({func_clean} of {acq_clean})'
        })

        fig = px.line(
            df_melted, x='Date', y='Value', color='Metric', markers=True,
            title=f"Monthly Trend Comparison: Marketing vs. Acquisitions",
            template="plotly_white"
        )
        fig.update_layout(hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated monthly comparison chart across {len(df)} months.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Monthly Comparison Error: {str(e)}", "data": None, "figure": None}