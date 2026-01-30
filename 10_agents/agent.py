import sys
from planner import plan
from tools import rag_tool, calculator_tool, llm_tool


def run_agent(question: str):
    print("\n=== USER QUESTION ===")
    print(question)

    decision = plan(question)

    print("\n=== PLANNER DECISION ===")
    print(decision)

    action = decision["action"]

    if action == "rag":
        rag_tool(question)

    elif action == "calculator":
        result = calculator_tool(question)
        print("\n=== ANSWER ===")
        print(result)

    else:
        result = llm_tool(question)
        print("\n=== ANSWER ===")
        print(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"your question\"")
        sys.exit(1)

    run_agent(sys.argv[1])
