def plan(question: str) -> dict:
    """
    Decide which tool to use based on the question.
    """

    q = question.lower()

    # Very simple heuristics (intentionally transparent)
    if any(word in q for word in ["grant thornton", "document", "notes", "services"]):
        return {
            "action": "rag",
            "reason": "Question requires document grounding"
        }

    if any(word in q for word in ["calculate", "sum", "add", "multiply", "divide"]):
        return {
            "action": "calculator",
            "reason": "Question requires numerical computation"
        }

    return {
        "action": "llm",
        "reason": "General knowledge question"
    }
