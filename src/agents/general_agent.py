"""
agents/general_agent.py
Handles general real estate knowledge questions and unknown requests.
"""

from langchain_core.messages import HumanMessage

from state.agent_state import AgentState
from utils.llm_client import get_llm

llm = get_llm(model_type="openai", model_name="gpt-4o-mini")


def general_agent_node(state: AgentState) -> dict:
    """
    Answer generic real estate questions using the LLM's built-in knowledge.
    Used when no specific property data lookup is needed.
    """
    prompt = f"""
You are a knowledgeable real estate asset management assistant.
Answer the following question concisely and accurately.

Question: {state['user_query']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


def unknown_agent_node(state: AgentState) -> dict:
    """
    Fallback handler for ambiguous, unsupported, or malformed requests.
    Provides a friendly help message rather than a cryptic error.
    """
    return {
        "final_response": (
            "I'm sorry, I couldn't understand your request. "
            "Here's what I can help you with:\n\n"
            "- **Price comparison**: 'What is the value of 123 Main St?'\n"
            "- **P&L calculation**: 'What is the P&L for all my properties?'\n"
            "- **Property details**: 'Show details for 456 Oak Ave'\n"
            "- **General questions**: 'What is a cap rate?'\n\n"
            "Please rephrase your question or provide more details."
        )
    }
