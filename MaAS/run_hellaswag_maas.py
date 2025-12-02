import argparse
import os
from pathlib import Path
from typing import List, Optional

from maas.configs.llm_config import LLMConfig, LLMType
from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
from maas.ext.maas.scripts.optimizer import Optimizer


def build_llm_config() -> LLMConfig:
    """
    Build an LLMConfig from environment variables.

    Two modes are supported:

    1) Groq (OpenAI-compatible)
       - Set GROQ_API_KEY to enable this mode.
       - Optional:
           GROQ_BASE_URL (defaults to https://api.groq.com/openai/v1)
           GROQ_MODEL (defaults to llama-3.3-70b-versatile)

    2) OpenAI (fallback)
       - Required: OPENAI_API_KEY
       - Optional:
           OPENAI_BASE_URL (defaults to https://api.openai.com/v1)
           OPENAI_MODEL (defaults to gpt-4o-mini)
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return LLMConfig(
            api_key=groq_key,
            api_type=LLMType.OPENAI,
            base_url=base_url,
            model=model,
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    return LLMConfig(
        api_key=api_key,
        api_type=LLMType.OPENAI,
        base_url=base_url,
        model=model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MaAS on HellaSwag without the MetaGPT config stack."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="Graph",
        choices=["Graph", "Test"],
        help="Graph = train controller, Test = evaluate with saved controller.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Number of training repetitions (few is cheaper).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for HellaSwag evaluation.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Optimization round index (used in output paths).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Maximum number of HellaSwag problems to evaluate (subset for cost).",
    )
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="maas/ext/maas/scripts/optimized",
        help="Where to store optimized workflows and logs.",
    )
    return parser.parse_args()


def build_indices(max_examples: int) -> Optional[List[int]]:
    if max_examples is None or max_examples <= 0:
        return None
    return list(range(max_examples))


def main() -> None:
    args = parse_args()

    config = EXPERIMENT_CONFIGS["HellaSwag"]

    opt_llm_config = build_llm_config()
    exec_llm_config = build_llm_config()

    indices = build_indices(args.max_examples)

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        round=args.round,
        batch_size=args.batch_size,
        lr=0.01,
        is_textgrad=False,
    )

    optimizer.indices = indices

    if args.mode == "Test":
        optimizer.optimize("Test")
    else:
        optimizer.optimize("Graph")


if __name__ == "__main__":
    main()


