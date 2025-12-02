# MaAS: Multi-agent Architecture Search via Agentic Supernet

## 📰 News

- 🎉 Updates (2025-05-03) MaAS is accepted as **ICML'25 Oral (Top ~1% among 12,107 submissions)**!
- 🚩 Updates (2025-02-06) Initial upload to arXiv (see [PDF](https://arxiv.org/abs/2502.04180)).


## 🤔 What is Agentic Supernet?

We *for the first time* shift the paradigm of automated multi-agent system design from seeking a (possibly non-existent) single optimal system to optimizing a probabilistic, continuous distribution of agentic architectures, termed the **agentic supernet**. 

![MaAS](assets/MaAS.png)

## 👋🏻 Method Overview

Building on this concept, we propose **MaAS**, which dynamically samples multi-agent systems that deliver satisfactory performance and token efficiency for user queries across different domains and varying levels of difficulty. Concretely, MaAS takes diverse and varying difficulty queries as input and leverages a controller to sample a subnetwork from the agentic supernet for each query, corresponding to a customized multi-agent system. After the sampled system executes the query, MaAS receives environment feedback and jointly optimizes the supernet’s parameterized distribution and agentic operators.

![framework](assets/framework.png)

## 🏃‍♂️‍➡️ Quick Start

### 🔬 Course Project: HellaSwag Experiment (This Repo)

This fork has been adapted to run MaAS on the **HellaSwag** benchmark (small validation subset) for a course project. The key additions are:

- `maas/ext/maas/benchmark/hellaswag.py` – HellaSwag benchmark class.
- `maas/ext/maas/scripts/download_hellaswag.py` – downloads and prepares a small HellaSwag validation subset.
- `run_hellaswag_maas.py` – lightweight runner that bypasses the MetaGPT config stack.
- `maas/ext/maas/scripts/optimized/HellaSwag/**` – train/test graphs and operators for HellaSwag.
- `maas/ext/maas/scripts/analyze_hellaswag_results.py` – finds 5 easiest failures and 5 hardest successes.
- `hellaswag_analysis.md` – written analysis of the HellaSwag runs and multi‑agent systems.

From the `MaAS` directory on Windows:

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) Download a small HellaSwag validation subset
python -m maas.ext.maas.scripts.download_hellaswag

# 4) Set OpenAI API key (example)
$env:OPENAI_API_KEY = "sk-..."          # replace with your key
$env:OPENAI_MODEL   = "gpt-4o-mini"     # or another compatible model

# 5) Train MaAS controller on HellaSwag (Graph mode)
python run_hellaswag_maas.py --mode Graph --sample 1 --batch_size 1 --max_examples 20

# 6) Evaluate on HellaSwag (Test mode)
python run_hellaswag_maas.py --mode Test  --sample 1 --batch_size 1 --max_examples 20

# 7) Analyze results: 5 easiest failures & 5 hardest successes
python -m maas.ext.maas.scripts.analyze_hellaswag_results `
  --log_root "maas/ext/maas/scripts/optimized/HellaSwag/test/round_1" `
  --trace_file "maas/ext/maas/data/hellaswag_traces.jsonl" `
  --top_k 5
```

The last command prints the problems, expected labels, MaAS predictions, and the sampled multi‑agent architectures (layers and operators) used for each example. The written discussion of these results lives in `hellaswag_analysis.md`.

---

### 📊 Original Datasets (Upstream MaAS)

For the original MaAS experiments, please download the `GSM8K`, `HumanEval`, and `MATH` datasets and place them in the `maas\ext\maas\data` folder. The file structure should be organized as follows:

```text
data
└── gsm8k_train.jsonl
└── gsm8k_test.jsonl
└── ...
```

### 🔑 Add API keys (Upstream MaAS)

You can configure `~/.metagpt/config2.yaml` according to the example.yaml, or `~/config/config2.yaml`:

```yaml
llm:
  api_type: "openai"
  model: "gpt-4o-mini"
  base_url: ""
  api_key: ""
```

### 🐹 Run the original HumanEval example

The code below (from the upstream project) verifies the experimental results of the `HumanEval` dataset:

```bash
python -m examples.maas.optimize --dataset HumanEval --round 1 --sample 4 --exec_model_name "gpt-4o-mini"
python -m examples.maas.optimize --dataset HumanEval --round 1 --sample 4 --exec_model_name "gpt-4o-mini" --is_test True
```

## 📚 Citation

If you find this repo useful, please consider citing our paper as follows:

```bibtex
@article{zhang2025agentic-supernet,
  title={Multi-agent Architecture Search via Agentic Supernet},
  author={Zhang, Guibin and Niu, Luyang and Fang, Junfeng and Wang, Kun and Bai, Lei and Wang, Xiang},
  journal={arXiv preprint arXiv:2502.04180},
  year={2025}
}
```

## 🙏 Acknowledgement

Special thanks to the following repositories for their invaluable code and prompt.

Our prompt is partially adapted from [ADAS](https://github.com/ShengranHu/ADAS), [AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare/tree/main), and [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow). Our code and operators are partially adapted from [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow).
