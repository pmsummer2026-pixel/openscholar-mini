import json
from pathlib import Path

SRC_DIR = Path("data/parsed")
OUT_DIR = Path("data/normalized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TITLE_CHARS = 300


def clean_title(title, fallback):
    if not title:
        return fallback
    title = " ".join(title.split())
    if len(title) > MAX_TITLE_CHARS:
        cut = title[:MAX_TITLE_CHARS]
        last_period = cut.rfind(". ")
        if last_period > 50:
            cut = cut[:last_period]
        return cut.strip() + "..."
    return title


def extract_full_text(raw):
    ft = raw.get("fullText")
    if isinstance(ft, str):
        return ft
    if isinstance(ft, dict):
        return ft.get("fullText", "") or ""
    return ""


count = 0
for path in sorted(SRC_DIR.glob("*.json")):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("metadata", {})
    fallback_id = path.stem.replace("_metadata", "")
    paper_id = meta.get("arxiv") or fallback_id
    title = clean_title(meta.get("title", ""), fallback_id)
    authors = meta.get("authors") or []
    year = meta.get("year")
    abstract = meta.get("abstract") or ""
    full_text = extract_full_text(raw)

    sections = []
    if full_text.strip():
        sections.append({"heading": "Full Text", "text": full_text})

    normalized = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "sections": sections,
    }

    out_name = fallback_id.replace(" ", "_") + ".json"
    out_path = OUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    count += 1

print(f"Normalized {count} papers into {OUT_DIR}")
