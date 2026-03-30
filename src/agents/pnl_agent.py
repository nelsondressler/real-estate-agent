"""
agents/pnl_agent.py
Specialist agent for Profit & Loss calculations.
"""

import os
from langchain_core.messages import HumanMessage

from state.agent_state import AgentState
from tools.data_tools import find_property, calculate_pnl
from utils.llm_client import get_llm
from utils.data_loader import load_properties

llm = get_llm(model_type="openai", model_name="gpt-4o-mini")

synthetic = os.getenv("SYNTHETIC_DATA", "false") == "true"  # Set to False to load real data from DATA_PATH
df = load_properties(synthetic=synthetic)


def pnl_agent_node(state: AgentState) -> dict:
    """
    Calculate P&L for specific properties or the entire portfolio.

    - If addresses are provided, calculates P&L per property.
    - If no addresses are given, calculates total portfolio P&L.
    """
    addresses = state["extracted_addresses"]
    pnl_data: dict[str, float] = {}
    not_found: list[str] = []

    if addresses:
        for addr in addresses:
            row = find_property(addr, df)
            if row is not None:
                pnl_data[row["address"]] = calculate_pnl(row)
            else:
                not_found.append(addr)
    else:
        # No specific address → aggregate portfolio P&L
        total_pnl = sum(calculate_pnl(row) for _, row in df.iterrows())
        pnl_data["Total Portfolio"] = total_pnl

    if not pnl_data:
        return {
            "final_response": (
                f"No properties were found matching: {', '.join(not_found)}. "
                "Please check the address and try again."
            )
        }

    pnl_summary = "\n".join([f"- {k}: ${v:,.0f}" for k, v in pnl_data.items()])
    not_found_note = (
        f"\nNote: The following addresses were not found: {', '.join(not_found)}"
        if not_found
        else ""
    )

    prompt = f"""
The user asked: "{state['user_query']}"

Calculated Profit & Loss:
{pnl_summary}
{not_found_note}

Write a clear, concise response presenting the P&L results.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"retrieved_data": pnl_data, "final_response": response.content}
