"""
state/agent_state.py
Shared state schema for the LangGraph multi-agent graph.
"""

from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    """
    Single source of truth passed between every node in the graph.

    Fields
    ------
    user_query : str
        The raw natural-language input from the user.
    intent : Optional[str]
        Detected intent: "price" | "pnl" | "detail" | "general" | "unknown".
    extracted_addresses : List[str]
        Property addresses parsed from the query.
    retrieved_data : Optional[dict]
        Structured data fetched from the dataset (prices, P&L, details).
    final_response : Optional[str]
        Human-readable response to be shown to the user.
    error : Optional[str]
        Error message if processing failed at any stage.
    """

    user_query: str
    intent: Optional[str]
    extracted_addresses: List[str]
    retrieved_data: Optional[dict]
    final_response: Optional[str]
    error: Optional[str]
