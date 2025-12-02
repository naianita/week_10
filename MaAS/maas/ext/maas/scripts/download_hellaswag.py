import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    """
    Download a small validation subset of HellaSwag and save it as JSONL in the
    format expected by the HellaSwagBenchmark.

    We only keep 100 validation examples to keep training/eval cheap.
    """
    ds = load_dataset("hellaswag", split="validation")
    subset = ds.select(range(100))

    repo_root = Path(__file__).resolve().parents[4]
    data_dir = repo_root / "maas" / "ext" / "maas" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "hellaswag_val.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for ex in subset:
            rec = {
                "context": ex["ctx"],
                # HellaSwag does not have a separate 'question' field; we treat
                # 'ctx' as the full prompt and leave question empty.
                "question": "",
                "endings": ex["endings"],
                "label": int(ex["label"]),
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Saved HellaSwag subset to {out_path}")


if __name__ == "__main__":
    main()


