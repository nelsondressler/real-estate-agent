"""
app.py
Streamlit user interface for the Real Estate Asset Management Assistant.
Run: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from main import run_query

load_dotenv()

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Asset Manager",
    page_icon="🏢",
    layout="centered",
)

st.title("🏢 Real Estate Asset Management Assistant")
st.caption(
    "Ask about property values, P&L, or asset details using natural language."
)

# ── Example prompts ──────────────────────────────────────────────────────────
with st.expander("💡 Example questions"):
    st.markdown(
        """
- *What is the price of my asset at 123 Main St compared to 456 Oak Ave?*
- *What is the total P&L for all my properties this year?*
- *Show me the details for the property at 789 Pine Ln.*
- *What is a cap rate?*
        """
    )

# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your real estate portfolio…"):
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run multi-agent graph
    with st.chat_message("assistant"):
        with st.spinner("Consulting the asset management agents…"):
            try:
                response = run_query(prompt)
            except Exception as e:
                response = f"⚠️ An error occurred: {str(e)}"
        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
