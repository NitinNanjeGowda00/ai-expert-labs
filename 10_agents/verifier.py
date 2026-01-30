"""
Verifier: checks answer quality and grounding.
"""

SCORE_THRESHOLD = 0.35


def verify(result: dict) -> dict:
    if result["status"] != "ok":
        return {"final": "I couldn't retrieve a reliable answer.", "safe": True}

    if not result.get("citations"):
        return {"final": "I don't know based on the provided documents.", "safe": True}

    if (result.get("best_score") or 0.0) < SCORE_THRESHOLD:
        return {"final": "I don't know based on the provided documents.", "safe": True}

    return {"final": result["answer"], "safe": True}
