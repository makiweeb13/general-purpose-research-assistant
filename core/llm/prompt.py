from typing import Any

TEMPLATE = (
    "You are a helpful research assistant.\n\n"
    "Use ONLY the information in the context below to answer the question.\n"
    "If the answer cannot be found in the context, say \"I don't know based on the provided sources.\"\n\n"
    "Context:\n"
    "{context}\n\n"
    "Question:\n"
    "{question}\n\n"
    "Answer:\n"
)


def build_prompt(context: str, question: str) -> str:
    """Return a prompt string ready to send to an LLM.

    Example:
        from core.llm.prompt import build_prompt
        prompt = build_prompt(context_text, "What is the plot?")
    """
    safe_context = context.strip() if context is not None else ""
    safe_question = question.strip() if question is not None else ""
    return TEMPLATE.format(context=safe_context, question=safe_question)


__all__ = ["build_prompt"]
