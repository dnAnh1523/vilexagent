from pathlib import Path
import pandas as pd
import re
from bs4 import BeautifulSoup
from src.utils.logger import logger

PROCESSED_DIR = Path("data/processed")

# Vietnamese article boundary pattern
# Matches: "Điều 1.", "Điều 12:", "Điều 1 -", "ĐIỀU 1."
ARTICLE_PATTERN = re.compile(
    r"(Đ[Ii][Ềề][Uu]\s+\d+[\.\:\-])",
    re.UNICODE
)

def extract_text_from_html(html: str) -> str:
    """Strip HTML tags and return clean plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)

def split_into_articles(text: str, doc_id: str, metadata: dict) -> list[dict]:
    """
    Split document text at article (Điều) boundaries.
    Returns list of chunk dicts with metadata attached.
    """
    if not text:
        return []

    # Find all article boundary positions
    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        # No article markers found — treat whole doc as single chunk
        return [{
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_0",
            "article_number": "N/A",
            "text": text[:5000],  # cap at 5000 chars for non-structured docs
            **metadata
        }]

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 30:
            # Skip near-empty chunks
            continue

        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_{i}",
            "article_number": match.group().strip(),
            "text": chunk_text,
            **metadata
        })

    return chunks

def parse_documents():
    logger.info("Loading joined documents...")
    df = pd.read_parquet(PROCESSED_DIR / "joined_documents.parquet")

    # Only process documents that have HTML content
    df_with_content = df[df["content_html"].notna() & (df["content_html"] != "")]
    logger.info(f"Documents to parse: {len(df_with_content)}")

    all_chunks = []
    failed = []

    for i, (_, row) in enumerate(df_with_content.iterrows()):
        try:
            text = extract_text_from_html(str(row["content_html"]))

            metadata = {
                "title": str(row.get("title", "")),
                "so_ky_hieu": str(row.get("so_ky_hieu", "")),
                "loai_van_ban": str(row.get("loai_van_ban", "")),
                "ngay_ban_hanh": str(row.get("ngay_ban_hanh", "")),
                "ngay_co_hieu_luc": str(row.get("ngay_co_hieu_luc", "")),
                "ngay_het_hieu_luc": str(row.get("ngay_het_hieu_luc", "")),
                "tinh_trang_hieu_luc": str(row.get("tinh_trang_hieu_luc", "")),
                "co_quan_ban_hanh": str(row.get("co_quan_ban_hanh", "")),
                "domain": str(row.get("domain", "")),
                "source": "domestic",
                "language": "vi",
            }

            chunks = split_into_articles(text, str(row["id"]), metadata)
            all_chunks.extend(chunks)

        except Exception as e:
            failed.append(str(row["id"]))
            logger.warning(f"Failed to parse doc {row['id']}: {e}")

        if (i + 1) % 1000 == 0:
            logger.info(f"Parsed {i + 1}/{len(df_with_content)} documents, {len(all_chunks)} chunks so far...")

    logger.success(f"Parsing complete: {len(all_chunks)} chunks from {len(df_with_content)} documents")
    logger.warning(f"Failed documents: {len(failed)}")

    df_chunks = pd.DataFrame(all_chunks)
    save_path = PROCESSED_DIR / "parsed_chunks.parquet"
    df_chunks.to_parquet(str(save_path), index=False)
    logger.success(f"Saved chunks → {save_path}")

    # Stats
    logger.info(f"Average chunks per document: {len(all_chunks) / len(df_with_content):.1f}")
    logger.info(f"Chunks with article markers: {(df_chunks['article_number'] != 'N/A').sum()}")
    logger.info(f"Chunks without article markers: {(df_chunks['article_number'] == 'N/A').sum()}")

    # Domain breakdown
    for domain, group in df_chunks.groupby("domain"):
        logger.info(f"  [{domain}] {len(group)} chunks")

    return df_chunks

if __name__ == "__main__":
    parse_documents()