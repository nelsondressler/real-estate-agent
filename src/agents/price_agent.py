"""
agents/price_agent.py
Specialist agent for property price / value comparison queries.
"""

import os
import json
from langchain_core.messages import HumanMessage

from state.agent_state import AgentState
from tools.data_tools import find_property
from utils.llm_client import get_llm
from utils.data_loader import load_properties

model_type = os.getenv("MODEL_TYPE", "openai")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
llm = get_llm(model_type=model_type, model_name=model_name)

synthetic = os.getenv("SYNTHETIC_DATA", "false") == "true"  # Set to False to load real data from DATA_PATH
df = load_properties(synthetic=synthetic)


def price_agent_node(state: AgentState) -> dict:
    """
    Retrieve current values for one or more properties and compare them.

    Handles:
    - Single property value lookup
    - Multi-property comparison
    - Partial address not found
    - Empty address list
    """
    addresses = state["extracted_addresses"]

    if not addresses:
        return {
            "final_response": (
                "Please specify at least one property address "
                "to look up or compare prices."
            )
        }

    found: dict[str, float] = {}
    not_found: list[str] = []

    for addr in addresses:
        row = find_property(addr, df)
        if row is not None:
            if 'price' in row:
                found[addr] = float(row["price"])
            elif 'profit' in row:
                found[addr] = float(row["profit"])
            else:
                found[addr] = 0.0
        else:
            not_found.append(addr)

    if not found:
        return {
            "final_response": (
                f"No properties were found matching: {', '.join(not_found)}. "
                "Please check the address and try again."
            )
        }

    price_summary = "\n".join([f"- {addr}: ${val:,.0f}" for addr, val in found.items()])
    not_found_note = (
        f"\nNote: The following addresses were not found in the database: "
        f"{', '.join(not_found)}"
        if not_found
        else ""
    )

    prompt = f"""
The user asked: "{state['user_query']}"

Retrieved property values:
{price_summary}
{not_found_note}

Write a clear, concise response comparing the property values.
If only one property was found, report its value directly.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"retrieved_data": found, "final_response": response.content}
