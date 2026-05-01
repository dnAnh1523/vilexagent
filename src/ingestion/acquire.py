from pathlib import Path
from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import hf_hub_download
from src.utils.logger import logger

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_metadata_and_relationships():
    for config in ["metadata", "relationships"]:
        logger.info(f"Downloading config: '{config}'...")
        dataset = load_dataset(
            "th1nhng0/vietnamese-legal-documents",
            config,
            split="data"
        )
        save_path = RAW_DIR / f"vietnamese_legal_{config}"
        dataset.save_to_disk(str(save_path))
        logger.success(f"[{config}] {len(dataset)} records → {save_path}")
        logger.info(f"[{config}] Columns: {list(dataset.features.keys())}")

def download_content_direct():
    """
    Bypass datasets library for content config — parquet uses large_string
    which causes ArrowInvalid cast error. Read directly with PyArrow instead.
    """
    logger.info("Downloading config: 'content' (direct PyArrow method)...")
    parquet_path = hf_hub_download(
        repo_id="th1nhng0/vietnamese-legal-documents",
        filename="data/content.parquet",
        repo_type="dataset"
    )
    logger.info(f"Parquet file cached at: {parquet_path}")

    # Read with PyArrow, preserving large_string as-is
    table = pq.read_table(parquet_path)
    logger.info(f"Schema: {table.schema}")
    logger.info(f"Total rows: {table.num_rows}")

    # Save as parquet directly — no casting
    save_path = RAW_DIR / "vietnamese_legal_content.parquet"
    pq.write_table(table, str(save_path))
    logger.success(f"[content] {table.num_rows} records → {save_path}")
    return table

def download_vietnamese_legal_docs():
    download_metadata_and_relationships()
    download_content_direct()
    logger.success("All 3 configs downloaded and saved locally.")

if __name__ == "__main__":
    download_vietnamese_legal_docs()