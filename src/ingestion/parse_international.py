from pathlib import Path
import json
import re
import pandas as pd
from src.utils.logger import logger

RAW_DIR = Path("data/raw/international")
PROCESSED_DIR = Path("data/processed")

# Article boundary patterns for English legal text
ARTICLE_PATTERN = re.compile(
    r"(Article\s+\d+[\.\:\-]?\s*\w*)",
    re.IGNORECASE
)

def split_into_articles(text: str, doc_id: str, metadata: dict) -> list[dict]:
    if not text:
        return []

    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        return [{
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_0",
            "article_number": "N/A",
            "text": text[:5000],
            **metadata
        }]

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 30:
            continue

        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_{i}",
            "article_number": match.group().strip(),
            "text": chunk_text,
            **metadata
        })

    return chunks

def parse_international():
    json_files = list(RAW_DIR.glob("*.json"))
    logger.info(f"Found {len(json_files)} international chapter files")

    all_chunks = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            chapter = json.load(f)

        metadata = {
            "title": f"{chapter['agreement']} {chapter['chapter']}: {chapter['topic']}",
            "so_ky_hieu": f"{chapter['agreement']}-{chapter['chapter'].replace(' ', '')}",
            "loai_van_ban": "FTA Chapter",
            "ngay_ban_hanh": "",
            "ngay_co_hieu_luc": "",
            "ngay_het_hieu_luc": "",
            "tinh_trang_hieu_luc": "in_effect",
            "co_quan_ban_hanh": chapter["agreement"],
            "domain": chapter["domain"],
            "source": "international",
            "language": chapter["language"],
            "agreement": chapter["agreement"],
        }

        chunks = split_into_articles(chapter["text"], chapter["id"], metadata)
        all_chunks.extend(chunks)
        logger.success(f"[{chapter['id']}] {len(chunks)} chunks")

    df_chunks = pd.DataFrame(all_chunks)
    save_path = PROCESSED_DIR / "international_chunks.parquet"
    df_chunks.to_parquet(str(save_path), index=False)
    logger.success(f"Total: {len(all_chunks)} international chunks → {save_path}")

    for domain, group in df_chunks.groupby("domain"):
        logger.info(f"  [{domain}] {len(group)} chunks")

    return df_chunks

if __name__ == "__main__":
    parse_international()