import os
from pathlib import Path

import requests


GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dst.write_bytes(resp.content)


def main() -> None:
    """
    Download GSM8K train/test splits into the location expected by MaAS.

    This creates:
        maas/ext/maas/data/gsm8k_train.jsonl
        maas/ext/maas/data/gsm8k_test.jsonl
    relative to the repository root.
    """
    repo_root = Path(__file__).resolve().parents[4]
    data_dir = repo_root / "maas" / "ext" / "maas" / "data"

    train_path = data_dir / "gsm8k_train.jsonl"
    test_path = data_dir / "gsm8k_test.jsonl"

    print(f"Saving GSM8K data under: {data_dir}")

    if not train_path.exists():
        print("Downloading GSM8K train split...")
        download_file(GSM8K_TRAIN_URL, train_path)
    else:
        print("GSM8K train split already exists, skipping.")

    if not test_path.exists():
        print("Downloading GSM8K test split...")
        download_file(GSM8K_TEST_URL, test_path)
    else:
        print("GSM8K test split already exists, skipping.")

    print("Done.")


if __name__ == "__main__":
    main()


