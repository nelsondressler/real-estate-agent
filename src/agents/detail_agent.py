"""
agents/detail_agent.py
Specialist agent for property detail retrieval queries.
"""

import os
import json
from langchain_core.messages import HumanMessage

from state.agent_state import AgentState
from tools.data_tools import find_property
from utils.llm_client import get_llm
from utils.data_loader import load_properties

llm = get_llm(model_type="openai", model_name="gpt-4o-mini")

synthetic = os.getenv("SYNTHETIC_DATA", "false") == "true"  # Set to False to load real data from DATA_PATH
df = load_properties(synthetic=synthetic)


def detail_agent_node(state: AgentState) -> dict:
    """
    Retrieve and present full property details for one or more addresses.
    """
    addresses = state["extracted_addresses"]

    if not addresses:
        return {
            "final_response": (
                "Please specify a property address to retrieve its details."
            )
        }

    details: dict[str, dict] = {}
    not_found: list[str] = []

    for addr in addresses:
        row = find_property(addr, df)
        if row is not None:
            details[row["address"]] = row.to_dict()
        else:
            not_found.append(addr)

    if not details:
        return {
            "final_response": (
                f"No property found matching: {', '.join(not_found)}. "
                "Please check the address and try again."
            )
        }

    details_json = json.dumps(details, indent=2, default=str)
    not_found_note = (
        f"\nNote: Not found: {', '.join(not_found)}" if not_found else ""
    )

    prompt = f"""
The user asked: "{state['user_query']}"

Property details:
{details_json}
{not_found_note}

Present the information in a clear, structured, and readable format.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"retrieved_data": details, "final_response": response.content}
