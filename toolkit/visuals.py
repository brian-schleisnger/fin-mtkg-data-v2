import mlflow
import pandas as pd
import plotly.express as px

from .base import run_sql_query, get_join_clause

__all__ = [
    "generate_scatterplot_tool",
    "generate_barchart_tool",
    "generate_histogram_tool",
    "generate_linechart_tool",
    "compare_monthly_metrics_tool"
]

@mlflow.trace(name="generate_scatterplot_tool")
def generate_scatterplot_tool(x_config: dict, y_config: dict, category_column: str = None, where_clause: str = None, include_trendline: bool = False) -> dict:
    """Sub-agent tool: Generates an interactive Plotly scatterplot. Supports single-table or cross-table queries."""
    
    table_x = x_config["table_name"]
    col_x = x_config["column_name"].replace('"', '').split('.')[-1]
    
    table_y = y_config["table_name"]
    col_y = y_config["column_name"].replace('"', '').split('.')[-1]

    # Handle optional category column
    cat_select = ""
    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        # Defaulting category to table_x for simplicity in this snippet
        cat_select = f', {table_x}."{cat_clean}" AS category_col'

    # Single-Table Scenario
    if table_x == table_y:
        sql_query = f'SELECT "{col_x}" AS x, "{col_y}" AS y {cat_select} FROM {table_x}'
        if where_clause:
            sql_query += f" WHERE {where_clause}"
            
    # Cross-Table Scenario
    else:
        join_condition = get_join_clause(table_x, table_y)
        if not join_condition:
            return {"text": f"Error: No known relationship to join {table_x} and {table_y}.", "data": None}
            
        sql_query = f"""
            SELECT 
                {table_x}."{col_x}" AS x, 
                {table_y}."{col_y}" AS y 
                {cat_select}
            FROM {table_x}
            JOIN {table_y} ON {join_condition}
        """
        if where_clause:
            sql_query += f" WHERE {where_clause}"

    sql_query += " ORDER BY RANDOM() LIMIT 50000"

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned. Check filters or join conditions.", "data": None}

        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['x', 'y'])

        if df.empty:
            return {"text": "Error: Not enough valid numeric data to plot.", "data": None}

        t_line = "ols" if include_trendline else None
        
        if category_column:
            df['category_col'] = df['category_col'].astype(str)
            fig = px.scatter(df, x='x', y='y', color='category_col', trendline=t_line, title=f"{col_y} vs {col_x}")
        else:
            fig = px.scatter(df, x='x', y='y', trendline=t_line, title=f"{col_y} vs {col_x}")

        # Update axis labels to be descriptive
        fig.update_layout(xaxis_title=col_x, yaxis_title=col_y)

        return {
            "text": f"Successfully generated scatterplot for {col_y} vs {col_x}.", 
            "data": df, 
            "figure": fig
        }
    except Exception as e:
        return {"text": f"Scatterplot Error: {e}", "data": None}
    
@mlflow.trace(name="generate_barchart_tool")
def generate_barchart_tool(TABLE_NAME, x_column: str, y_column: str, category_column: str = None, where_clause: str = None) -> dict:
    """Sub-agent tool: Generates an interactive Plotly bar chart."""
    x_clean = x_column.replace('"', '').split('.')[-1]
    y_clean = y_column.replace('"', '').split('.')[-1]
    columns_to_fetch = [x_clean, y_clean]

    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        columns_to_fetch.append(cat_clean)

    safe_columns = ['"{}"'.format(col) for col in list(set(columns_to_fetch))]
    columns_str = ", ".join(safe_columns)
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    if where_clause:
        sql_query += f" WHERE {where_clause}"
    sql_query += " LIMIT 5000"

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for the bar chart.", "data": None}

        df[y_clean] = pd.to_numeric(df[y_clean], errors='coerce')
        df = df.dropna(subset=[x_clean, y_clean])

        if category_column:
            df[cat_clean] = df[cat_clean].astype(str)
            fig = px.bar(df, x=x_clean, y=y_clean, color=cat_clean, title=f"{y_clean} by {x_clean}")
        else:
            fig = px.bar(df, x=x_clean, y=y_clean, title=f"{y_clean} by {x_clean}")

        return {"text": f"Successfully generated bar chart for {y_clean} by {x_clean}.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Bar Chart Error: {e}", "data": None}

@mlflow.trace(name="generate_histogram_tool")
def generate_histogram_tool(TABLE_NAME, x_column: str, n_bins: int = None, category_column: str = None, where_clause: str = None) -> dict:
    """Sub-agent tool: Generates an interactive Plotly histogram to show data distributions."""
    x_clean = x_column.replace('"', '').split('.')[-1]
    columns_to_fetch = [x_clean]

    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        columns_to_fetch.append(cat_clean)

    safe_columns = ['"{}"'.format(col) for col in list(set(columns_to_fetch))]
    columns_str = ", ".join(safe_columns)
    
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    if where_clause:
        sql_query += f" WHERE {where_clause}"
    sql_query += " LIMIT 10000" # Slightly larger limit for distributions

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for the histogram.", "data": None}

        df[x_clean] = pd.to_numeric(df[x_clean], errors='coerce')
        df = df.dropna(subset=[x_clean])

        kwargs = {"x": x_clean, "title": f"Distribution of {x_clean}"}
        if n_bins:
            kwargs["nbins"] = n_bins
        if category_column:
            df[cat_clean] = df[cat_clean].astype(str)
            kwargs["color"] = cat_clean
            kwargs["barmode"] = "overlay" # Better default for overlapping distributions

        fig = px.histogram(df, **kwargs)

        return {"text": f"Successfully generated histogram for {x_clean}.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Histogram Error: {e}", "data": None}

@mlflow.trace(name="generate_linechart_tool")
def generate_linechart_tool(TABLE_NAME, x_column: str, y_column: str, category_column: str = None, where_clause: str = None) -> dict:
    """Sub-agent tool: Generates an interactive Plotly line chart, sorting by X-axis."""
    x_clean = x_column.replace('"', '').split('.')[-1]
    y_clean = y_column.replace('"', '').split('.')[-1]
    columns_to_fetch = [x_clean, y_clean]

    if category_column:
        cat_clean = category_column.replace('"', '').split('.')[-1]
        columns_to_fetch.append(cat_clean)

    safe_columns = ['"{}"'.format(col) for col in list(set(columns_to_fetch))]
    columns_str = ", ".join(safe_columns)
    
    # We order by the X column at the SQL level to ensure the line draws correctly chronologically/sequentially
    sql_query = f"SELECT {columns_str} FROM {TABLE_NAME}"
    if where_clause:
        sql_query += f" WHERE {where_clause}"
    sql_query += f" ORDER BY \"{x_clean}\" ASC LIMIT 5000"

    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: No data returned for the line chart.", "data": None}

        df[y_clean] = pd.to_numeric(df[y_clean], errors='coerce')
        df = df.dropna(subset=[x_clean, y_clean])

        if category_column:
            df[cat_clean] = df[cat_clean].astype(str)
            fig = px.line(df, x=x_clean, y=y_clean, color=cat_clean, title=f"Trend of {y_clean} over {x_clean}")
        else:
            fig = px.line(df, x=x_clean, y=y_clean, title=f"Trend of {y_clean} over {x_clean}")

        return {"text": f"Successfully generated line chart for {y_clean} over {x_clean}.", "data": df, "figure": fig}
    except Exception as e:
        return {"text": f"Line Chart Error: {e}", "data": None}
    
@mlflow.trace(name="compare_monthly_metrics_tool")
def compare_monthly_metrics_tool(marketing_metric: str, acquisition_metric_func: str = "COUNT", acquisition_column: str = "*") -> dict:
    """
    Aggregates acquisition data to a monthly grain and joins it with monthly marketing metrics
    to generate an interactive trend line chart.
    """
    marketing_clean = marketing_metric.replace('"', '').split('.')[-1]
    acq_clean = acquisition_column.replace('"', '').split('.')[-1] if acquisition_column != "*" else "*"
    func_clean = acquisition_metric_func.upper()
    
    # Validate aggregation function to prevent injection
    if func_clean not in ["COUNT", "SUM", "AVG"]:
        func_clean = "COUNT"

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
            return {"text": "Error: No data returned for the time-series comparison.", "data": None}

        # Create a unified date string for the X-axis
        df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01', errors='coerce')
        df = df.dropna(subset=['Date', 'marketing_total', 'acquisition_total'])

        # Melt the dataframe so Plotly can easily plot two lines with a legend
        df_melted = df.melt(id_vars=['Date'], value_vars=['marketing_total', 'acquisition_total'], 
                            var_name='Metric', value_name='Value')

        fig = px.line(df_melted, x='Date', y='Value', color='Metric', 
                      title=f"Monthly Trend: {marketing_clean} vs {func_clean} of {acq_clean}")

        return {
            "text": f"Successfully generated monthly comparison chart for {marketing_clean} and {func_clean} of {acq_clean}.", 
            "data": df, 
            "figure": fig
        }
    except Exception as e:
        return {"text": f"Monthly Comparison Error: {e}", "data": None}