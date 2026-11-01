"""Prompt construction.

The prompt is the instruction template supplied by the project owner::

    You are a senior business and finance analyst ... answer the user's question accurately and concisely.

    **Document Content:**
    {content}

    **Question:**
    {question}

    **Instructions:**
    - ...

    **Answer:**

``{content}`` is the whole document as structure-preserving Markdown (headings as
``#``/``##``, tables as ``| a | b |`` rows, list items as ``- item``). The text ends right after
``**Answer:**`` and a newline; the model continues from there.
"""

from __future__ import annotations

NOT_AVAILABLE_ANSWER = "The requested information is not available in the provided document."

_INTRO = (
    "You are a senior business and finance analyst at a reputed organization. "
    "Your task is to analyze the provided document content and answer the user's question "
    "accurately and concisely.\n\n"
)
CONTENT_HEADER = _INTRO + "**Document Content:**\n"
QUESTION_HEADER = "\n\n**Question:**\n"
INSTRUCTIONS_AND_ANSWER_HEADER = (
    "\n\n**Instructions:**\n"
    "- Answer the question using only the information provided in the document.\n"
    "- Provide a precise, crisp, and professional answer that directly addresses the question.\n"
    "- Avoid unnecessary explanations, background information, or repetition.\n"
    "- Do not make assumptions or introduce information that is not supported by the document.\n"
    f'- If the answer is not available in the provided content, respond with: "{NOT_AVAILABLE_ANSWER}"\n'
    "\n**Answer:**\n"
)


def build_prompt(context: str, question: str) -> str:
    """Return the full prompt (ending right after ``**Answer:**\\n``)."""
    prefix, suffix = prompt_parts(question)
    return f"{prefix}{context.strip()}{suffix}"


def prompt_parts(question: str) -> tuple[str, str]:
    """Return ``(prefix, suffix)`` so that ``prefix + context + suffix == build_prompt(...)``.

    The model service uses this to shorten only the document part when the prompt would
    exceed the token budget, keeping the question, the instructions and ``**Answer:**`` intact.
    """
    return CONTENT_HEADER, f"{QUESTION_HEADER}{question.strip()}{INSTRUCTIONS_AND_ANSWER_HEADER}"


def first_line(generated: str) -> str:
    """Use the first line of the generated text as the answer."""
    stripped = generated.strip()
    return stripped.splitlines()[0].strip() if stripped else ""


def is_unanswerable(answer: str) -> bool:
    """True when the model says the document does not contain the answer.

    Matches the sentence the prompt asks for, and the bare label ``unanswerable``.
    """
    normalized = answer.strip().strip('"').strip().rstrip(".").lower()
    return normalized in {"unanswerable", NOT_AVAILABLE_ANSWER.rstrip(".").lower()}
