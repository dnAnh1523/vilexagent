from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from src.utils.logger import logger

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

def join_metadata_content():
    logger.info("Loading filtered metadata...")
    df_meta = pd.read_parquet(PROCESSED_DIR / "filtered_metadata.parquet")
    df_meta["id"] = df_meta["id"].astype(str)
    logger.info(f"Filtered metadata: {len(df_meta)} documents")

    logger.info("Loading content parquet...")
    df_content = pq.read_table(RAW_DIR / "vietnamese_legal_content.parquet").to_pandas()
    df_content["id"] = df_content["id"].astype(str)
    logger.info(f"Content records: {len(df_content)}")

    # Check for duplicate IDs before joining
    meta_dupes = df_meta["id"].duplicated().sum()
    content_dupes = df_content["id"].duplicated().sum()
    logger.info(f"Duplicate IDs in metadata: {meta_dupes}")
    logger.info(f"Duplicate IDs in content: {content_dupes}")

    # Deduplicate both sides before joining
    df_meta = df_meta.drop_duplicates(subset="id", keep="first")
    df_content = df_content.drop_duplicates(subset="id", keep="first")
    logger.info(f"After dedup — metadata: {len(df_meta)}, content: {len(df_content)}")

    logger.info("Joining on document id...")
    df_joined = df_meta.merge(df_content, on="id", how="left")

    has_content = df_joined["content_html"].notna() & (df_joined["content_html"] != "")
    no_content = ~has_content

    logger.success(f"Total joined documents: {len(df_joined)}")
    logger.success(f"Documents with HTML content: {has_content.sum()}")
    logger.warning(f"Documents without content (PDF-only scans): {no_content.sum()}")

    # Save both
    save_path = PROCESSED_DIR / "joined_documents.parquet"
    df_joined.to_parquet(str(save_path), index=False)
    logger.success(f"Saved joined documents → {save_path}")

    pdf_only = df_joined[no_content][["id", "title", "so_ky_hieu", "domain"]]
    pdf_save_path = PROCESSED_DIR / "pdf_only_documents.parquet"
    pdf_only.to_parquet(str(pdf_save_path), index=False)
    logger.info(f"PDF-only document list saved → {pdf_save_path}")

    logger.info("--- Domain breakdown ---")
    for domain, group in df_joined.groupby("domain"):
        with_content = group["content_html"].notna().sum()
        logger.info(f"  [{domain}] {len(group)} total, {with_content} with content")

    return df_joined

if __name__ == "__main__":
    join_metadata_content()