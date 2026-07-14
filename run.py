"""
CLI entry point: ask a question against your mini paper corpus.

Usage:
    python run.py --question "How do these papers define X?" \
        --store_dir index/store --top_k 20 --top_n 8 --model gpt-4o-mini

Requires:
    - An index already built with index/build_index.py
    - OPENAI_API_KEY set in your environment (or adapt generator.py for a
      local/open model instead of the OpenAI API)
"""
import argparse
import json

from generator import generate_answer
from retriever import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Question to ask about your paper corpus")
    parser.add_argument("--store_dir", default="index/store")
    parser.add_argument("--top_k", type=int, default=20, help="Candidates pulled from vector search")
    parser.add_argument("--top_n", type=int, default=8, help="Passages kept after reranking")
    parser.add_argument("--max_per_paper", type=int, default=2)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no_rerank", action="store_true", help="Skip the cross-encoder reranker")
    parser.add_argument("--no_self_check", action="store_true", help="Skip the self-critique revision pass")
    parser.add_argument("--output_file", default=None, help="Optional path to save the result as JSON")
    args = parser.parse_args()

    retriever = Retriever(store_dir=args.store_dir, use_reranker=not args.no_rerank)
    chunks = retriever.retrieve(
        args.question,
        top_k=args.top_k,
        top_n=args.top_n,
        max_per_paper=args.max_per_paper,
    )

    if not chunks:
        print("No relevant passages found in the index.")
        return

    answer = generate_answer(
        args.question,
        chunks,
        model=args.model,
        self_check=not args.no_self_check,
    )

    print("\n=== Answer ===\n")
    print(answer)

    print("\n=== References ===\n")
    for i, c in enumerate(chunks):
        print(f"[{i}] {c['title']} - {c.get('heading', '')} (paper_id={c.get('paper_id')})")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({
                "question": args.question,
                "answer": answer,
                "references": chunks,
            }, f, indent=2)
        print(f"\nSaved full result to {args.output_file}")


if __name__ == "__main__":
    main()
