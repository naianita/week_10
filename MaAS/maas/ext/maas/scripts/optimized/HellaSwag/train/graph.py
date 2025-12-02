import json
from pathlib import Path

import torch
import maas.ext.maas.scripts.optimized.GSM8K.train.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.GSM8K.train.template.operator as operator
from maas.ext.maas.scripts.optimized.GSM8K.train.template.operator_registry import (
    operator_mapping,
    operator_names,
)
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager
from maas.logs import logger


class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset,
        controller: torch.nn.Module,
        operator_embeddings,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.cost_manager = CostManager()
        self.custom = operator.Generate(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names

    def _log_trace(self, problem: str, final_solution: str, selected_names_layers):
        """Persist the sampled multi-agent architecture for later analysis."""
        trace = {
            "question": problem,
            "final_solution": final_solution,
            "layers": [
                {"layer": idx, "operators": names}
                for idx, names in enumerate(selected_names_layers)
            ],
        }
        log_dir = Path("maas") / "ext" / "maas" / "data"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "hellaswag_traces.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    async def __call__(self, problem: str):
        log_probs_layers, selected_names_layers = self.controller.forward(
            problem, self.operator_embeddings, self.selection_operator_names
        )

        current_solution = ""
        solutions = []
        sum_log_prob = 0.0

        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_name in selected_names:
                selected_operator = self.selection_operator_instances[op_name]

                if op_name in ["Generate", "GenerateCoT"]:
                    result = await selected_operator(
                        input=problem, instruction=prompt_custom.MATH_SOLVE_PROMPT
                    )
                    new_solution = result.get("response", "")
                    solutions.append(new_solution)
                elif op_name == "SelfRefine":
                    result = await selected_operator(
                        problem=problem, solution=current_solution
                    )
                    new_solution = result.get("response", "")
                    solutions.append(new_solution)
                elif op_name == "Programmer":
                    result = await selected_operator(
                        problem=problem, analysis=current_solution
                    )
                    refined_solution = await self.custom(
                        input=problem + f"\nCode output: {result['code']}",
                        instruction=prompt_custom.REFINE_ANSWER_PROMPT,
                    )

                    new_solution = refined_solution["response"]
                    solutions.append(new_solution)
                elif op_name == "ScEnsemble":
                    result = await selected_operator(problem=problem, solutions=solutions)
                    solutions = []
                    new_solution = result.get("response", "")
                    solutions.append(new_solution)
                elif op_name == "MultiGenerateCoT":
                    result = await selected_operator(
                        input=problem, instruction=prompt_custom.MATH_SOLVE_PROMPT
                    )
                    if isinstance(result, dict) and "response" in result:
                        for res in result["response"]:
                            new_solution = res.get("response", "")
                            solutions.append(new_solution)
                    else:
                        logger.error(
                            "Expected dict with 'responses' from MultiGenerateCoT, "
                            f"got {type(result)}"
                        )
                        new_solution = current_solution
                else:
                    new_solution = current_solution

                current_solution = new_solution

            sum_log_prob += log_probs_layers[layer_idx].item()

        if len(solutions) > 1:
            final_solution = await self.sc_ensemble(
                solutions=solutions, problem=problem
            )
            final_solution = final_solution["response"]
        else:
            final_solution = current_solution

        verification = await self.programmer(problem=problem, analysis=final_solution)

        # Log the sampled multi-agent architecture for this problem.
        self._log_trace(problem, final_solution, selected_names_layers)

        if verification["output"] and verification["output"] != "No code generated":
            return (
                verification["output"],
                self.llm.cost_manager.total_cost,
                sum_log_prob,
            )
        else:
            return final_solution, self.llm.cost_manager.total_cost, sum_log_prob
