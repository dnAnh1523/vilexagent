from pathlib import Path
from datasets import load_from_disk
import pyarrow.parquet as pq
import pandas as pd
from src.utils.logger import logger

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Target domains in Vietnamese as they appear in the dataset
# linh_vuc = field/sector, nganh = industry/branch
LABOR_KEYWORDS = [
    "lao động", "việc làm", "tiền lương", "bảo hiểm xã hội",
    "công đoàn", "hợp đồng lao động", "an toàn lao động",
    "quan hệ lao động", "người lao động"
]

FOOD_SAFETY_KEYWORDS = [
    "an toàn thực phẩm", "vệ sinh thực phẩm", "thực phẩm",
    "kiểm dịch", "kiểm nghiệm thực phẩm", "phụ gia thực phẩm",
    "an toàn vệ sinh"
]

def keyword_match(text: str, keywords: list[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

def filter_by_domain():
    logger.info("Loading metadata dataset...")
    meta = load_from_disk(str(RAW_DIR / "vietnamese_legal_metadata"))
    df = meta.to_pandas()
    logger.info(f"Total documents: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")

    # Combine searchable text fields
    df["search_text"] = (
        df["title"].fillna("") + " " +
        df["linh_vuc"].fillna("") + " " +
        df["nganh"].fillna("")
    ).str.lower()

    # Filter by domain
    labor_mask = df["search_text"].apply(
        lambda x: keyword_match(x, LABOR_KEYWORDS)
    )
    food_mask = df["search_text"].apply(
        lambda x: keyword_match(x, FOOD_SAFETY_KEYWORDS)
    )

    df_labor = df[labor_mask].copy()
    df_food = df[food_mask].copy()
    df_combined = df[labor_mask | food_mask].copy()

    df_labor["domain"] = "labor"
    df_food["domain"] = "food_safety"

    # For overlapping docs, label as labor (priority)
    df_combined["domain"] = "food_safety"
    df_combined.loc[labor_mask[labor_mask | food_mask], "domain"] = "labor"

    logger.success(f"Labor law documents: {len(df_labor)}")
    logger.success(f"Food safety documents: {len(df_food)}")
    logger.success(f"Combined (deduplicated): {len(df_combined)}")

    # Save filtered metadata
    save_path = PROCESSED_DIR / "filtered_metadata.parquet"
    df_combined.to_parquet(str(save_path), index=False)
    logger.success(f"Saved filtered metadata → {save_path}")

    # Show sample from each domain
    logger.info("--- Labor sample ---")
    for _, row in df_labor.head(3).iterrows():
        logger.info(f"  [{row['loai_van_ban']}] {row['title'][:80]}")

    logger.info("--- Food safety sample ---")
    for _, row in df_food.head(3).iterrows():
        logger.info(f"  [{row['loai_van_ban']}] {row['title'][:80]}")

    return df_combined

if __name__ == "__main__":
    filter_by_domain()