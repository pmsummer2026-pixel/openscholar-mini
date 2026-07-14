Place your 10-15 source papers here.

Suggested layout:
  data/pdfs/         raw PDF files you have rights to use (e.g. arXiv papers)
  data/parsed/        one JSON file per paper after running index/build_index.py's parsing step, e.g.:
    {
      "paper_id": "2402.xxxxx",
      "title": "...",
      "authors": ["..."],
      "year": 2024,
      "abstract": "...",
      "sections": [{"heading": "Introduction", "text": "..."}, ...]
    }

Large PDFs and generated JSON are excluded from git via .gitignore - only this placeholder is tracked.
