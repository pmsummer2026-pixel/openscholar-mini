"""
Retrieve and rerank chunks from the local index built by index/build_index.py.
"""
import json
import os
from collections import defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class Retriever:
    def __init__(self, store_dir="index/store", embed_model=DEFAULT_MODEL,
                 reranker_model=DEFAULT_RERANKER, use_reranker=True):
        self.store_dir = store_dir
        self.index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
        self.chunks = []
        with open(os.path.join(store_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        self.embed_model = SentenceTransformer(embed_model)

        self.reranker = None
        if use_reranker:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(reranker_model, use_fp16=True)

    def search(self, query, top_k=20):
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, idxs = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["retrieval_score"] = float(score)
            results.append(chunk)
        return results

    def rerank(self, query, candidates, top_n=8, max_per_paper=2):
        if self.reranker is not None:
            pairs = [[query, f"{c['title']}: {c['text']}"] for c in candidates]
            scores = self.reranker.compute_score(pairs, batch_size=32)
            if isinstance(scores, float):
                scores = [scores]
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        else:
            candidates.sort(key=lambda c: c["retrieval_score"], reverse=True)

        # cap how many chunks come from the same paper so one source doesn't dominate
        per_paper_count = defaultdict(int)
        capped = []
        for c in candidates:
            pid = c.get("paper_id")
            if per_paper_count[pid] >= max_per_paper:
                continue
            capped.append(c)
            per_paper_count[pid] += 1
            if len(capped) >= top_n:
                break
        return capped

    def retrieve(self, query, top_k=20, top_n=8, max_per_paper=2):
        candidates = self.search(query, top_k=top_k)
        return self.rerank(query, candidates, top_n=top_n, max_per_paper=max_per_paper)
