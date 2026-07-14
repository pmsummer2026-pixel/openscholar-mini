# OpenScholar Mini

A small-scale, educational replication of [AkariAsai/OpenScholar](https://github.com/AkariAsai/OpenScholar) — a retrieval-augmented language model system for synthesizing scientific literature. This project reproduces the same overall pipeline (chunk papers, embed and index them, retrieve and rerank relevant passages, then generate a cited answer) but scoped down to a personal corpus of about 10 to 15 papers so it can run on a laptop with free or low-cost tools.

## Why this exists

The original OpenScholar is built on a datastore of roughly 45 million papers, a custom-trained retriever and reranker, and a fine-tuned Llama-3.1-8B generator. That scale is not reproducible outside a well-resourced lab. This repo keeps the same architecture and citation-driven prompting strategy, but swaps in off-the-shelf, small-scale components: a local vector index over a handful of papers, an open sentence-embedding model, an existing cross-encoder reranker, and any instruction-following LLM (local or API) for generation.

## Pipeline overview

1. Ingest - collect 10-15 PDFs on a single sub-topic and parse them into structured JSON (title, authors, year, abstract, section text).
2. Chunk - split each paper into paragraph-sized chunks with metadata retained for citation.
3. Index - embed all chunks and store them in a lightweight local vector store.
4. Retrieve - embed the user's question, pull the top candidates by similarity.
5. Rerank - score candidates with a cross-encoder and cap how many chunks come from a single paper.
6. Generate - prompt an LLM with numbered references and the question, requiring inline [i] citations grounded only in the provided references.
7. Self-check (optional) - ask the model to review its own draft for uncited or unsupported claims and revise.

## Repository layout

- data/ - raw PDFs and parsed per-paper JSON files (not committed if large; see .gitignore).
- index/build_index.py - chunks papers and builds the local embedding index.
- retriever.py - similarity search plus reranking over the index.
- generator.py - prompt construction and LLM call for cited answer generation.
- run.py - command-line entry point tying retrieval and generation together.
- requirements.txt - Python dependencies.

## Status

Scaffold only. Fill in data/ with your own paper set and implement/adjust the scripts to taste.

## Credit

Architecture and prompting approach adapted from the original [OpenScholar](https://github.com/AkariAsai/OpenScholar) project by Asai et al. This is an independent, unaffiliated educational replication, not the official codebase.
