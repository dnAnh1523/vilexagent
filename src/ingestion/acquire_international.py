import requests
from pathlib import Path
from bs4 import BeautifulSoup
import json
import time
import fitz  # PyMuPDF
from src.utils.logger import logger

RAW_DIR = Path("data/raw/international")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Direct official sources — PDFs from NZ MFAT and HTML from EU Commission
CHAPTERS = [
    {
        "id": "evfta_ch6_sps",
        "agreement": "EVFTA",
        "chapter": "Chapter 6",
        "topic": "Sanitary and Phytosanitary Measures",
        "domain": "food_safety",
        "language": "en",
        "type": "html",
        "url": "https://thuvienphapluat.vn/tintuc/vn/hiep-dinh-evfta/12851/hiep-dinh-evfta-chapter-6-sanitary-and-phytosanitary-measures"
    },
    {
        "id": "evfta_ch13_labor",
        "agreement": "EVFTA",
        "chapter": "Chapter 13",
        "topic": "Trade and Sustainable Development",
        "domain": "labor",
        "language": "en",
        "type": "pdf",
        "url": "https://trungtamwto.vn/upload/files/fta/196-chua-ky-ket/199-viet-nam---eu-evfta/248-noi-dung-hiep-dinh/13%20CHAPTER%2013%20Trade%20and%20Sustainable%20Development.pdf"
    },
    {
        "id": "cptpp_ch19_labor",
        "agreement": "CPTPP",
        "chapter": "Chapter 19",
        "topic": "Labour",
        "domain": "labor",
        "language": "en",
        "type": "pdf",
        "url": "https://www.mfat.govt.nz/assets/Trade-agreements/TPP/Text-ENGLISH/19.-Labour-Chapter.pdf"
    },
    {
        "id": "cptpp_ch7_sps",
        "agreement": "CPTPP",
        "chapter": "Chapter 7",
        "topic": "Sanitary and Phytosanitary Measures",
        "domain": "food_safety",
        "language": "en",
        "type": "pdf",
        "url": "https://www.mfat.govt.nz/assets/Trade-agreements/TPP/Text-ENGLISH/7.-Sanitary-and-Phytosanitary-Measures-Chapter.pdf"
    },
]

def fetch_html(chapter: dict) -> str:
    response = requests.get(chapter["url"], headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    content = None
    for selector in ["article", ".content", ".main-content", "#content", "main"]:
        content = soup.select_one(selector)
        if content:
            break
    if not content:
        content = soup.find("body")
    text = content.get_text(separator="\n") if content else ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def fetch_pdf(chapter: dict) -> str:
    response = requests.get(chapter["url"], headers=HEADERS, timeout=60)
    response.raise_for_status()

    # Save PDF temporarily
    pdf_path = RAW_DIR / f"{chapter['id']}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    # Extract text with PyMuPDF
    doc = fitz.open(str(pdf_path))
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()

    text = "\n".join(pages_text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def fetch_chapter(chapter: dict) -> dict | None:
    logger.info(f"Fetching {chapter['agreement']} {chapter['chapter']}: {chapter['topic']} ({chapter['type'].upper()})...")
    try:
        if chapter["type"] == "pdf":
            text = fetch_pdf(chapter)
        else:
            text = fetch_html(chapter)

        if len(text) < 500:
            logger.warning(f"Very short content for {chapter['id']}: {len(text)} chars")

        result = {
            **chapter,
            "text": text,
            "char_count": len(text),
            "source": "international",
            "language": "en",
        }

        save_path = RAW_DIR / f"{chapter['id']}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.success(f"Saved {chapter['id']} ({len(text):,} chars) → {save_path}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch {chapter['id']}: {e}")
        return None

def acquire_international():
    results = []
    for chapter in CHAPTERS:
        result = fetch_chapter(chapter)
        if result:
            results.append(result)
        time.sleep(2)

    logger.info("\n--- Summary ---")
    for r in results:
        status = "✓" if r["char_count"] > 1000 else "⚠ SHORT"
        logger.info(f"  [{r['agreement']} {r['chapter']}] {r['char_count']:,} chars {status}")

    logger.success(f"Downloaded {len(results)}/{len(CHAPTERS)} chapters")
    return results

if __name__ == "__main__":
    acquire_international()