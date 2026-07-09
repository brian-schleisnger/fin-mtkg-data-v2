from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analytics import link_tables
from .base import run_sql_query

__all__ = [
    "generate_scatterplot_tool",
    "generate_barchart_tool",
    "generate_histogram_tool",
    "generate_linechart_tool",
    "compare_monthly_metrics_tool"
]


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


def _prepare_query_params(
    x_col: str, 
    y_col: str, 
    cat_col: Optional[str] = None, 
    agg_func: Optional[str] = None,
    default_limit: int = 50000
) -> dict:
    """
    Determines whether to push aggregation down to the SQL engine (to avoid LIMIT truncation)
    or fetch raw rows for plotting.
    """
    x_clean = x_col.replace('"', '').split('.')[-1]
    y_clean = y_col.replace('"', '').split('.')[-1]
    cat_clean = cat_col.replace('"', '').split('.')[-1] if cat_col else None

    func = agg_func.upper() if agg_func and agg_func.upper() in ["SUM", "AVG", "COUNT", "MAX", "MIN"] else None

    if func:
        # SQL Push-Down: Group by X and Category at the database level
        group_by_cols = [x_clean]
        if cat_clean:
            group_by_cols.append(cat_clean)

        if func == "COUNT" and (y_clean == "*" or y_clean.lower() == "count"):
            y_sql = 'COUNT(*) AS "y_value"'
            y_label = "COUNT(*)"
        else:
            y_sql = f'{func}("{y_clean}") AS "y_value"'
            y_label = f"{func}({y_clean})"

        cols_to_fetch = [f'"{x_clean}"']
        if cat_clean:
            cols_to_fetch.append(f'"{cat_clean}"')
        cols_to_fetch.append(y_sql)

        return {
            "columns": cols_to_fetch,
            "group_by": group_by_cols,
            "limit": 5000, # Grouped queries only return a few rows, but 5000 prevents runaway memory
            "y_target": "y_value",
            "y_label": y_label
        }
    else:
        # Fallback to raw row fetching
        cols = [f'"{x_clean}"', f'"{y_clean}"']
        if cat_clean:
            cols.append(f'"{cat_clean}"')
        return {
            "columns": cols,
            "group_by": None,
            "limit": default_limit,
            "y_target": y_clean,
            "y_label": y_clean
        }


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
    """Sub-agent tool: Generates an aggregated bar chart using SQL push-down aggregation."""
    params = _prepare_query_params(x_column, y_column, category_column, aggregation)
    x_clean = x_column.replace('"', '').split('.')[-1]

    try:
        # Pushing aggregation down to the database to prevent LIMIT truncation
        df = link_tables(
            TABLE_NAME, 
            columns=params["columns"], 
            where_clause=where_clause, 
            group_by=params["group_by"],
            order_by=f'"{x_clean}" ASC',
            limit=params["limit"]
        )
        if df.empty:
            return {"text": "Error: No data returned for bar chart.", "data": None, "figure": None}

        x_col = _resolve_column(x_clean, df)
        y_col = _resolve_column(params["y_target"], df)
        cat_col = _resolve_column(category_column, df) if category_column else None

        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])
        
        try:
            df = df.sort_values(by=x_col, ascending=True)
        except Exception:
            df = df.sort_values(by=y_col, ascending=False)

        kwargs = {
            "x": x_col, "y": y_col,
            "title": f"{params['y_label']} by {x_clean}",
            "template": "plotly_white"
        }
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col
            kwargs["barmode"] = "group"

        fig = px.bar(df, **kwargs)
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        fig.update_yaxes(title_text=params['y_label'])

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
            "template": "plotly_white", "marginal": "box",
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
    """Sub-agent tool: Generates a clean, sorted time-series line chart using SQL push-down aggregation."""
    params = _prepare_query_params(x_column, y_column, category_column, aggregation)
    x_clean = x_column.replace('"', '').split('.')[-1]

    try:
        # Pushing GROUP BY and ORDER BY down to SQL prevents row truncation!
        df = link_tables(
            TABLE_NAME, 
            columns=params["columns"], 
            where_clause=where_clause, 
            group_by=params["group_by"],
            order_by=f'"{x_clean}" ASC',
            limit=params["limit"]
        )
        if df.empty:
            return {"text": "Error: No data returned for line chart.", "data": None, "figure": None}

        x_col = _resolve_column(x_clean, df)
        y_col = _resolve_column(params["y_target"], df)
        cat_col = _resolve_column(category_column, df) if category_column else None

        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])

        try:
            df = df.sort_values(by=x_col, ascending=True)
        except Exception:
            pass

        kwargs = {
            "x": x_col, "y": y_col, "markers": True,
            "title": f"Trend of {params['y_label']} over {x_clean}",
            "template": "plotly_white"
        }
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col

        fig = px.line(df, **kwargs)
        fig.update_layout(hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))
        fig.update_yaxes(title_text=params['y_label'])

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