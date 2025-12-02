import asyncio
import re
from typing import Any, Callable, List, Optional, Tuple

import torch
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger


class HellaSwagBenchmark(BaseBenchmark):
    """
    HellaSwag multiple-choice benchmark.

    Each example is a context (and optional question), four candidate endings,
    and a gold label 0–3. We ask the graph to answer with a letter A–D and
    score 1.0 if it matches the gold option.
    """

    LETTERS = ["A", "B", "C", "D"]

    def _format_input(self, problem: dict) -> str:
        context = problem.get("context", "")
        question = problem.get("question", "")
        endings: List[str] = problem.get("endings", [])

        parts = [context.strip()]
        if question:
            parts.append(f"Question: {question.strip()}")
        parts.append("Choices:")
        for i, end in enumerate(endings):
            if i >= 4:
                break
            parts.append(f"{self.LETTERS[i]}) {end.strip()}")
        parts.append("\nAnswer with the single letter (A, B, C, or D) of the correct choice.")
        return "\n".join(parts)

    def _extract_letter(self, text: str) -> Optional[str]:
        """Extract the first A/B/C/D letter from the model output."""
        if not text:
            return None
        m = re.search(r"\b([ABCD])\b", text.upper())
        if m:
            return m.group(1)
        # fallback: look for 'answer is X'
        m2 = re.search(r"answer is\s*([ABCD])", text, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).upper()
        return None

    def calculate_score(self, expected_output: int, prediction: Optional[str]) -> Tuple[float, Optional[str]]:
        if prediction is None:
            return 0.0, prediction
        try:
            gold_idx = int(expected_output)
        except (TypeError, ValueError):
            return 0.0, prediction
        if 0 <= gold_idx < len(self.LETTERS):
            gold_letter = self.LETTERS[gold_idx]
        else:
            return 0.0, prediction
        return (1.0, prediction) if prediction == gold_letter else (0.0, prediction)

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_output(self, graph, input_text):
        """Thin wrapper around the graph call with retry and timeout."""
        return await asyncio.wait_for(graph(input_text), timeout=1500)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        input_text = self._format_input(problem)
        expected_output = problem.get("label", -1)

        try:
            output, cost, logprob = await self._generate_output(graph, input_text)
            if not output:
                raise ValueError("output is empty")

            pred_letter = self._extract_letter(output)
            score, extracted_output = self.calculate_score(expected_output, pred_letter)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost, logprob

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return (
                input_text,
                str(e),
                expected_output,
                0.0,
                0.0,
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
            )

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]


