import os
import pandas as pd
import streamlit as st
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────
DATA_PATH = "data/my_dataset.csv"  # Path to your bundled data file

# Use Databricks Foundation Model Serving as the LLM backend
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints"
)
MODEL = "databricks-meta-llama-3-3-70b-instruct"  # Or any available model

# ─── Load Data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load the dataset from the app's local data folder."""
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# ─── Build Schema Context for the LLM ────────────────────────────
def get_schema_description(dataframe):
    """Generate a text description of the dataframe schema."""
    lines = [f"Dataset has {len(dataframe)} rows and {len(dataframe.columns)} columns.\n"]
    lines.append("Columns:")
    for col in dataframe.columns:
        dtype = dataframe[col].dtype
        sample_values = dataframe[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({dtype}): e.g. {sample_values}")
    return "\n".join(lines)

schema_context = get_schema_description(df)

# ─── System Prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a data analyst assistant. The user will ask questions about a dataset.
You have access to a pandas DataFrame called `df` with the following schema:

{schema_context}

When the user asks a question:
1. Write Python code using pandas to answer it (assign the result to a variable called `result`)
2. Wrap your code in ```python ... ``` markers
3. After the code, provide a brief natural-language explanation of what the answer means

Rules:
- Only use pandas operations on `df`
- Do NOT import any additional libraries beyond pandas and numpy
- Do NOT modify the original dataframe
- Always assign your final answer to `result`
- `result` should be a simple value, Series, or small DataFrame (not the full df)
"""

# ─── UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Dataset Q&A", page_icon="📊", layout="wide")
st.title("📊 Ask Questions About Your Data")

# Show data preview in sidebar
with st.sidebar:
    st.header("Dataset Preview")
    st.write(f"**{len(df):,} rows × {len(df.columns)} columns**")
    st.dataframe(df.head(20), use_container_width=True)
    st.divider()
    st.header("Column Info")
    for col in df.columns:
        st.write(f"• `{col}` ({df[col].dtype})")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result_df" in message:
            st.dataframe(message["result_df"])

if prompt := st.chat_input("Ask a question about the data..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call LLM
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            answer = response.choices[0].message.content

            # Extract and execute code
            import re
            code_match = re.search(r"```python\n(.*?)```", answer, re.DOTALL)

            if code_match:
                code = code_match.group(1)
                st.code(code, language="python")

                try:
                    import numpy as np
                    local_vars = {"df": df.copy(), "pd": pd, "np": np}
                    exec(code, {}, local_vars)
                    result = local_vars.get("result", "No result produced.")

                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "result_df": result
                        })
                    elif isinstance(result, pd.Series):
                        st.dataframe(result.to_frame())
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "result_df": result.to_frame()
                        })
                    else:
                        st.success(f"**Result:** {result}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"{answer}\n\n**Result:** {result}"
                        })
                except Exception as e:
                    st.error(f"Error executing code: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {e}"
                    })
            else:
                # No code — just show the text response
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })