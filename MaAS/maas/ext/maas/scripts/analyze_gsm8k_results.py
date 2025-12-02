import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def load_latest_results(log_dir: Path) -> pd.DataFrame:
    csv_files = sorted(log_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV results found under {log_dir}")
    latest = csv_files[-1]
    return pd.read_csv(latest)


def load_traces(trace_path: Path) -> List[Dict[str, Any]]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    traces = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))
    return traces


def build_trace_index(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["question"]: t for t in traces}


def select_easiest_failures(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    failures = df[df["score"] == 0].copy()
    failures["q_len"] = failures["question"].str.len()
    return failures.sort_values("q_len").head(k)


def select_hardest_successes(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    successes = df[df["score"] == 1].copy()
    successes["q_len"] = successes["question"].str.len()
    return successes.sort_values("q_len", ascending=False).head(k)


def pretty_print_examples(
    title: str, rows: pd.DataFrame, trace_index: Dict[str, Dict[str, Any]]
) -> None:
    print(f"\n=== {title} ===")
    for _, row in rows.iterrows():
        q = row["question"]
        print("\n---")
        print("Question:")
        print(q)
        print("\nExpected:", row["expected_output"])
        print("Prediction:", row["prediction"])
        print("Score:", row["score"])
        trace = trace_index.get(q)
        if trace:
            print("Multi-agent system (layers -> operators):")
            for layer in trace["layers"]:
                print(f"  Layer {layer['layer']}: {', '.join(layer['operators'])}")
        else:
            print("Multi-agent system: <no trace found>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GSM8K MaAS results: easiest failures and hardest successes."
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="maas/ext/maas/scripts/optimized/GSM8K/test/workflows",
        help="Directory containing GSM8K CSV result files.",
    )
    parser.add_argument(
        "--trace_file",
        type=str,
        default="maas/ext/maas/data/gsm8k_traces.jsonl",
        help="Path to the operator trace JSONL file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many easiest failures / hardest successes to show.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_root)
    trace_path = Path(args.trace_file)

    df = load_latest_results(log_dir)
    traces = load_traces(trace_path)
    trace_index = build_trace_index(traces)

    easiest_failures = select_easiest_failures(df, args.top_k)
    hardest_successes = select_hardest_successes(df, args.top_k)

    pretty_print_examples(
        "5 easiest GSM8K failures (shortest questions where MaAS fails)",
        easiest_failures,
        trace_index,
    )
    pretty_print_examples(
        "5 hardest GSM8K successes (longest questions where MaAS succeeds)",
        hardest_successes,
        trace_index,
    )


if __name__ == "__main__":
    main()


