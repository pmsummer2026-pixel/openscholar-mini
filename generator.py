"""
Build a citation-grounded prompt from retrieved chunks and call an LLM to answer.

Mirrors OpenScholar's approach: numbered references, an instruction to answer
only from those references, and inline [i] citations pointing back to them.
"""
import os

SYSTEM_INSTRUCTIONS = (
    "You are a research assistant answering questions about a small set of "
    "scientific papers. Use ONLY the numbered references provided below to "
    "answer the question. Every factual sentence in your answer must end "
    "with one or more citations in the form [i], where i is the reference "
    "number it is based on. If the references do not contain enough "
    "information to answer, say so explicitly instead of guessing."
)

SELF_CHECK_INSTRUCTIONS = (
    "Review the draft answer below against the same references. "
    "Respond in exactly two sections using these exact markdown headings.\n\n"
    "## Issues\n"
    "List any sentence that lacks a citation or is not actually supported "
    "by the cited reference. If there are no issues, write 'None found.'\n\n"
    "## Revised Answer\n"
    "Always include this section in full, using the same [i] citation style. "
    "Write the complete corrected answer here, fixing any issues found above. "
    "If no issues were found, copy the draft answer here unchanged."
    )


def format_references(chunks):
    lines = []
    for i, c in enumerate(chunks):
        heading = f" ({c['heading']})" if c.get("heading") else ""
        lines.append(f"[{i}] Title: {c['title']}{heading}\nText: {c['text']}")
    return "\n\n".join(lines)


def build_answer_prompt(question, chunks):
    references = format_references(chunks)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"References:\n{references}\n\n"
        f"Question: {question}\n"
        f"Answer (with inline [i] citations):"
    )


def build_selfcheck_prompt(question, chunks, draft_answer):
    references = format_references(chunks)
    return (
        f"{SELF_CHECK_INSTRUCTIONS}\n\n"
        f"References:\n{references}\n\n"
        f"Question: {question}\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Revised answer (with inline [i] citations):"
    )


def call_claude(prompt, model="claude-sonnet-5", api_key_env="ANTHROPIC_API_KEY"):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ[api_key_env])
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")



def generate_answer(question, chunks, model="claude-sonnet-5", self_check=True):
    draft_prompt = build_answer_prompt(question, chunks)
    draft = call_claude(draft_prompt, model=model)

    if not self_check:
        return draft

    check_prompt = build_selfcheck_prompt(question, chunks, draft)
    revised = call_claude(check_prompt, model=model)
    if "## Revised Answer" in revised:
        return revised.split("## Revised Answer", 1)[1].lstrip(": \n#")
    return revised
