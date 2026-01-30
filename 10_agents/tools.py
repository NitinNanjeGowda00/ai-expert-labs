import subprocess
import sys
from openai import OpenAI


def rag_tool(question: str) -> str:
    """
    Call Segment 09 RAG pipeline as a subprocess.
    """
    cmd = [
        sys.executable,
        "09_llm_rag/ask.py",
        question
    ]

    print("\n[TOOL] RAG search")
    subprocess.run(cmd, check=True)
    return "RAG answer returned above."


def calculator_tool(question: str) -> str:
    """
    Extremely simple calculator tool.
    """
    print("\n[TOOL] Calculator")

    try:
        expression = (
            question.lower()
            .replace("calculate", "")
            .replace("what is", "")
            .strip()
        )
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation failed: {e}"


def llm_tool(question: str) -> str:
    """
    General LLM response (no retrieval).
    """
    print("\n[TOOL] General LLM")

    client = OpenAI()
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=question,
    )

    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    return c.text

    return "No response generated."
