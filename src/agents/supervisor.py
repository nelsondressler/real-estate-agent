"""
agents/supervisor.py
Supervisor agent: detects user intent and extracts property addresses.
"""

import os
import json
from langchain_core.messages import HumanMessage, SystemMessage

from state.agent_state import AgentState
from utils.llm_client import get_llm

model_type = os.getenv("MODEL_TYPE", "openai")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
llm = get_llm(model_type=model_type, model_name=model_name)


SUPERVISOR_SYSTEM = """
You are a real estate asset management assistant router.
Given a user query, return a JSON object with exactly two keys:
- "intent": one of ["price", "pnl", "detail", "general", "unknown"]
- "addresses": list of property addresses mentioned (empty list if none)

Intent rules:
- "price"   → user wants to compare or know property prices / market values
- "pnl"     → user wants profit, loss, P&L, return, or financial performance
- "detail"  → user wants specific property details (appraisal date, property type, etc.)
- "general" → generic real estate knowledge question not tied to a specific asset
- "unknown" → unclear, ambiguous, or unsupported request

Respond ONLY with valid JSON. No markdown. No explanation.
"""


def supervisor_node(state: AgentState) -> dict:
    """
    Detect intent and extract addresses from the user query.

    Returns a partial state update with `intent` and `extracted_addresses`.
    Falls back to "unknown" if the LLM response cannot be parsed.
    """
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=state["user_query"]),
    ]
    response = llm.invoke(messages)

    try:
        parsed = json.loads(response.content) # type: ignore
        intent = parsed.get("intent", "unknown")
        addresses = parsed.get("addresses", [])
    except (json.JSONDecodeError, AttributeError):
        intent = "unknown"
        addresses = []

    return {
        "intent": intent,
        "extracted_addresses": addresses,
    }
