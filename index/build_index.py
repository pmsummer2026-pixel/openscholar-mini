"""
Build a small local retrieval index over a personal corpus of papers.

Usage:
    python index/build_index.py --papers_dir data/parsed --out_dir index/store

Expects one JSON file per paper in --papers_dir, shaped like:
    {
      "paper_id": "2402.xxxxx",
      "title": "...",
      "authors": ["..."],
      "year": 2024,
      "abstract": "...",
      "sections": [{"heading": "Introduction", "text": "..."}, ...]
    }

Produces:
    <out_dir>/chunks.jsonl   one retrievable chunk per line, with metadata
    <out_dir>/index.faiss    FAISS flat index of chunk embeddings
"""
import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORD_TARGET = 220
CHUNK_WORD_OVERLAP = 40


def load_papers(papers_dir):
    papers = []
    for path in sorted(Path(papers_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            papers.append(json.load(f))
    return papers


def chunk_text(text, target_words=CHUNK_WORD_TARGET, overlap_words=CHUNK_WORD_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def build_chunks(papers):
    chunks = []
    for paper in papers:
        title = paper.get("title", "")
        for section in paper.get("sections", []) or []:
            heading = section.get("heading", "")
            for piece in chunk_text(section.get("text", "")):
                chunks.append({
                    "paper_id": paper.get("paper_id"),
                    "title": title,
                    "year": paper.get("year"),
                    "heading": heading,
                    "text": piece,
                })
        abstract = paper.get("abstract")
        if abstract:
            chunks.append({
                "paper_id": paper.get("paper_id"),
                "title": title,
                "year": paper.get("year"),
                "heading": "Abstract",
                "text": abstract,
            })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers_dir", default="data/parsed")
    parser.add_argument("--out_dir", default="index/store")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    papers = load_papers(args.papers_dir)
    if not papers:
        raise SystemExit(f"No paper JSON files found in {args.papers_dir}")
    print(f"Loaded {len(papers)} papers")

    chunks = build_chunks(papers)
    print(f"Built {len(chunks)} chunks")

    model = SentenceTransformer(args.model)
    texts = [f"{c['title']} - {c['heading']}: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(args.out_dir, "index.faiss"))

    with open(os.path.join(args.out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Saved index and chunks to {args.out_dir}")


if __name__ == "__main__":
    main()
