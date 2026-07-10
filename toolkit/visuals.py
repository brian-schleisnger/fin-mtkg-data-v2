from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analytics import link_tables
from .base import run_sql_query
from agent.memory import get_df_memory

__all__ = [
    "generate_scatterplot_tool",
    "generate_barchart_tool",
    "generate_histogram_tool",
    "generate_linechart_tool",
    "compare_monthly_metrics_tool"
]


def _resolve_column(requested_col: str, df: pd.DataFrame) -> str:
    """Safely resolves column names in Pandas DataFrames after SQL execution."""
    if not requested_col or df.empty:
        return requested_col
        
    clean_req = str(requested_col).replace('"', '').replace("'", "").strip()
    if clean_req in df.columns: return clean_req
    
    simple_req = clean_req.split('.')[-1]
    for col in df.columns:
        if str(col).split('.')[-1] == simple_req: return str(col)
    for col in df.columns:
        if str(col).split('.')[-1].lower() == simple_req.lower(): return str(col)
            
    return clean_req


def _fetch_chart_data(
    TABLE_NAME: Optional[Union[str, List[str]]],
    dataframe_id: Optional[str],
    x_column: str,
    y_column: Optional[str] = None,
    category_column: Optional[str] = None,
    where_clause: Optional[str] = None,
    aggregation: Optional[str] = None,
    limit: int = 50000
) -> tuple:
    """
    SHARED LOGIC HELPER: Handles routing between SQL and Python Memory.
    Automatically applies Pandas .groupby() or SQL GROUP BY if an aggregation is requested.
    Returns: (df, resolved_x, resolved_y, resolved_cat, y_axis_label)
    """
    x_clean = x_column.replace('"', '').split('.')[-1] if x_column else None
    y_clean = y_column.replace('"', '').split('.')[-1] if y_column else None
    cat_clean = category_column.replace('"', '').split('.')[-1] if category_column else None
    func = aggregation.upper() if aggregation and aggregation.upper() in ["SUM", "AVG", "COUNT", "MAX", "MIN"] else None

    y_label = y_clean
    if func and y_clean:
        y_label = "COUNT(*)" if (func == "COUNT" and y_clean in ["*", "count"]) else f"{func}({y_clean})"

    # --- BRANCH 1: PYTHON MEMORY ---
    if dataframe_id:
        df = get_df_memory().get_df(dataframe_id)
        if df is None:
            raise ValueError(f"No DataFrame found for ID '{dataframe_id}'.")
            
        x_col = _resolve_column(x_clean, df)
        y_col = _resolve_column(y_clean, df) if y_clean else None
        cat_col = _resolve_column(cat_clean, df) if cat_clean else None

        # Apply Pandas Aggregation if needed
        if func and y_col:
            group_cols = [x_col]
            if cat_col: group_cols.append(cat_col)
            
            if func == "SUM": df = df.groupby(group_cols, as_index=False)[y_col].sum()
            elif func == "AVG": df = df.groupby(group_cols, as_index=False)[y_col].mean()
            elif func == "MAX": df = df.groupby(group_cols, as_index=False)[y_col].max()
            elif func == "MIN": df = df.groupby(group_cols, as_index=False)[y_col].min()
            elif func == "COUNT": 
                df = df.groupby(group_cols, as_index=False)[y_col].count()

        return df, x_col, y_col, cat_col, y_label

    # --- BRANCH 2: SQL DATABASE ---
    elif TABLE_NAME:
        if func and y_clean:
            # SQL Push-Down Aggregation
            y_sql = 'COUNT(*) AS "y_value"' if (func == "COUNT" and y_clean in ["*", "count"]) else f'{func}("{y_clean}") AS "y_value"'
            cols_to_fetch = [f'"{x_clean}"']
            if cat_clean: cols_to_fetch.append(f'"{cat_clean}"')
            cols_to_fetch.append(y_sql)
            
            df = link_tables(
                TABLE_NAME, 
                columns=cols_to_fetch, 
                where_clause=where_clause, 
                group_by=[x_clean] + ([cat_clean] if cat_clean else []),
                order_by=f'"{x_clean}" ASC',
                limit=5000
            )
            y_target = "y_value"
        else:
            # Raw Row Fetching
            cols_to_fetch = [f'"{x_clean}"']
            if y_clean: cols_to_fetch.append(f'"{y_clean}"')
            if cat_clean: cols_to_fetch.append(f'"{cat_clean}"')
            
            df = link_tables(TABLE_NAME, columns=cols_to_fetch, where_clause=where_clause, random_order=True, limit=limit)
            y_target = y_clean

        if df.empty:
            raise ValueError("Query executed successfully but returned 0 rows.")

        x_col = _resolve_column(x_clean, df)
        y_col = _resolve_column(y_target, df) if y_clean else None
        cat_col = _resolve_column(cat_clean, df) if cat_clean else None
        
        return df, x_col, y_col, cat_col, y_label

    else:
        raise ValueError("Must provide either TABLE_NAME or dataframe_id.")


@mlflow.trace(name="generate_scatterplot_tool")
def generate_scatterplot_tool(
    x_column: str, 
    y_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None, 
    include_trendline: bool = False
) -> Dict[str, Any]:
    try:
        df, x_col, y_col, cat_col, _ = _fetch_chart_data(TABLE_NAME, dataframe_id, x_column, y_column, category_column, where_clause)
        
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])

        if len(df) < 2:
            return {"text": "Error: Not enough valid numeric data points.", "data": None, "figure": None}

        kwargs = {
            "x": x_col, "y": y_col, "trendline": "ols" if (include_trendline and len(df) > 3) else None,
            "title": f"Scatterplot: {y_col} vs {x_col}",
            "template": "plotly_white", "opacity": 0.75
        }
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col

        fig = px.scatter(df, **kwargs)
        fig.update_layout(hovermode="closest", margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated scatterplot ({len(df)} points).", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Scatterplot Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="generate_barchart_tool")
def generate_barchart_tool(
    x_column: str, 
    y_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None, 
    dataframe_id: Optional[str] = None,
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None,
    aggregation: str = "SUM"
) -> Dict[str, Any]:
    try:
        df, x_col, y_col, cat_col, y_label = _fetch_chart_data(TABLE_NAME, dataframe_id, x_column, y_column, category_column, where_clause, aggregation)
        
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])
        try: df = df.sort_values(by=x_col, ascending=True)
        except Exception: df = df.sort_values(by=y_col, ascending=False)

        kwargs = {"x": x_col, "y": y_col, "title": f"{y_label} by {x_col}", "template": "plotly_white"}
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col
            kwargs["barmode"] = "group"

        fig = px.bar(df, **kwargs)
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        fig.update_yaxes(title_text=y_label)

        return {"text": f"Successfully generated bar chart ({len(df)} bars).", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Bar Chart Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="generate_histogram_tool")
def generate_histogram_tool(
    x_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    n_bins: Optional[int] = None, 
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None
) -> Dict[str, Any]:
    try:
        df, x_col, _, cat_col, _ = _fetch_chart_data(TABLE_NAME, dataframe_id, x_column, None, category_column, where_clause)
        
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df = df.dropna(subset=[x_col])

        kwargs = {"x": x_col, "title": f"Distribution of {x_col}", "template": "plotly_white", "marginal": "box", "opacity": 0.8}
        if n_bins: kwargs["nbins"] = n_bins
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col
            kwargs["barmode"] = "overlay"

        fig = px.histogram(df, **kwargs)
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated histogram ({len(df)} records).", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Histogram Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="generate_linechart_tool")
def generate_linechart_tool(
    x_column: str, 
    y_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None, 
    dataframe_id: Optional[str] = None,
    category_column: Optional[str] = None, 
    where_clause: Optional[str] = None,
    aggregation: str = "SUM"
) -> Dict[str, Any]:
    try:
        df, x_col, y_col, cat_col, y_label = _fetch_chart_data(TABLE_NAME, dataframe_id, x_column, y_column, category_column, where_clause, aggregation)

        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[x_col, y_col])
        try: df = df.sort_values(by=x_col, ascending=True)
        except Exception: pass

        kwargs = {"x": x_col, "y": y_col, "markers": True, "title": f"Trend of {y_label} over {x_col}", "template": "plotly_white"}
        if cat_col and cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            kwargs["color"] = cat_col

        fig = px.line(df, **kwargs)
        fig.update_layout(hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))
        fig.update_yaxes(title_text=y_label)

        return {"text": f"Successfully generated line chart.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Line Chart Error: {str(e)}", "data": None, "figure": None}


@mlflow.trace(name="compare_monthly_metrics_tool")
def compare_monthly_metrics_tool(
    marketing_metric: str = "amount", 
    acquisition_metric_func: str = "COUNT", 
    acquisition_column: str = "*"
) -> Dict[str, Any]:
    """Kept as SQL-only because it relies on a very specific cross-table join macro."""
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
            return {"text": "Error: No data returned.", "data": None, "figure": None}

        df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01', errors='coerce')
        df = df.dropna(subset=['Date', 'marketing_total', 'acquisition_total']).sort_values(by='Date')

        df_melted = df.melt(id_vars=['Date'], value_vars=['marketing_total', 'acquisition_total'], var_name='Metric', value_name='Value')
        df_melted['Metric'] = df_melted['Metric'].replace({
            'marketing_total': f'Marketing Spend ({marketing_clean})',
            'acquisition_total': f'Acquisitions ({func_clean} of {acq_clean})'
        })

        fig = px.line(df_melted, x='Date', y='Value', color='Metric', markers=True, title="Monthly Trend Comparison", template="plotly_white")
        fig.update_layout(hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))

        return {"text": f"Successfully generated monthly comparison chart.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Monthly Comparison Error: {str(e)}", "data": None, "figure": None}