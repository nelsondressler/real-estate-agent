"""
graph.py
Assembles and compiles the LangGraph multi-agent graph.

Graph topology
--------------
                      supervisor
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     price_agent      pnl_agent     detail_agent
          │               │               │
          └───────────────┼───────────────┘
                      general_agent
                      unknown_agent
                          │
                         END
"""

from langgraph.graph import StateGraph, END

from state.agent_state import AgentState
from agents.supervisor import supervisor_node
from agents.price_agent import price_agent_node
from agents.pnl_agent import pnl_agent_node
from agents.detail_agent import detail_agent_node
from agents.general_agent import general_agent_node, unknown_agent_node


def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge: map detected intent to the corresponding agent node name.

    Returns the name of the next node to execute.
    """
    intent_map = {
        "price": "price_agent",
        "pnl": "pnl_agent",
        "detail": "detail_agent",
        "general": "general_agent",
    }
    return intent_map.get(state["intent"], "unknown_agent") # type: ignore


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Returns
    -------
    CompiledGraph
        Ready-to-invoke compiled graph.
    """
    builder = StateGraph(AgentState)

    # --- Register nodes ---
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("price_agent", price_agent_node)
    builder.add_node("pnl_agent", pnl_agent_node)
    builder.add_node("detail_agent", detail_agent_node)
    builder.add_node("general_agent", general_agent_node)
    builder.add_node("unknown_agent", unknown_agent_node)

    # --- Entry point ---
    builder.set_entry_point("supervisor")

    # --- Conditional routing from supervisor ---
    builder.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "price_agent": "price_agent",
            "pnl_agent": "pnl_agent",
            "detail_agent": "detail_agent",
            "general_agent": "general_agent",
            "unknown_agent": "unknown_agent",
        },
    )

    # --- All specialist agents terminate ---
    for node in ["price_agent", "pnl_agent", "detail_agent", "general_agent", "unknown_agent"]:
        builder.add_edge(node, END)

    return builder.compile() # type: ignore


# Module-level compiled graph (singleton)
graph = build_graph()
