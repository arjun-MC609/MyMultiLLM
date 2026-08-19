"""Download a Tamil text corpus from Wikipedia for training."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def download_tamil_corpus(output_path, target_size_mb=15.0, min_article_chars=200):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "This script requires the `datasets` library. Install it with:\n"
            "    pip install datasets --break-system-packages\n"
            "(In Colab: !pip install datasets -q)"
        ) from e

    logger.info("Streaming Tamil Wikipedia (this does not download the full dump)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ta", split="train", streaming=True)

    target_bytes = int(target_size_mb * 1024 * 1024)
    written_bytes = 0
    n_articles = 0
    n_skipped_short = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "").strip()

            if len(text) < min_article_chars:
                n_skipped_short += 1
                continue

            for paragraph in text.split("\n"):
                paragraph = paragraph.strip()
                if len(paragraph) < 20:
                    continue
                f.write(paragraph + "\n")
                written_bytes += len(paragraph.encode("utf-8")) + 1

            n_articles += 1

            if written_bytes >= target_bytes:
                break

    size_mb = written_bytes / (1024 * 1024)
    logger.info(
        "Done. Wrote %.2f MB from %d articles (skipped %d too-short) to %s",
        size_mb, n_articles, n_skipped_short, output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Tamil Wikipedia text corpus.")
    parser.add_argument("--output", type=str, default="data/raw/tamil_wiki.txt")
    parser.add_argument("--target-size-mb", type=float, default=15.0)
    parser.add_argument("--min-article-chars", type=int, default=200)
    args = parser.parse_args()

    download_tamil_corpus(
        output_path=args.output,
        target_size_mb=args.target_size_mb,
        min_article_chars=args.min_article_chars,
    )
