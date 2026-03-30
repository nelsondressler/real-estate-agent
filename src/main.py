"""
main.py
CLI entry point for the Real Estate Asset Management multi-agent system.
Run: python main.py
"""

from dotenv import load_dotenv

load_dotenv()  # Load API_KEYs before importing graph

from workflows.graph import graph
from state.agent_state import AgentState

def run_query(query: str) -> str:
    """
    Execute the multi-agent graph for a natural-language query.

    Parameters
    ----------
    query : str
        User's natural-language question.

    Returns
    -------
    str
        Final response from the appropriate specialist agent.
    """
    initial_state: AgentState = {
        "user_query": query,
        "intent": None,
        "extracted_addresses": [],
        "retrieved_data": None,
        "final_response": None,
        "error": None,
    }
    result = graph.invoke(initial_state) # type: ignore
    return result["final_response"]


def main() -> None:
    """Interactive REPL loop for the asset management assistant."""
    print("=" * 60)
    print("  Real Estate Asset Management Assistant")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = run_query(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
