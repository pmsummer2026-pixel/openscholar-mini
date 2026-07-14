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
    "Review the draft answer below against the same references. List any "
    "sentence that lacks a citation or is not actually supported by the "
    "cited reference, then produce a corrected final answer with the same "
    "citation style."
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


def call_openai(prompt, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[api_key_env])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def generate_answer(question, chunks, model="gpt-4o-mini", self_check=True):
    draft_prompt = build_answer_prompt(question, chunks)
    draft = call_openai(draft_prompt, model=model)

    if not self_check:
        return draft

    check_prompt = build_selfcheck_prompt(question, chunks, draft)
    revised = call_openai(check_prompt, model=model)
    return revised
