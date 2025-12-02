## Week 10 Course Project – MaAS on HellaSwag

This repository contains a course project built around **MaAS (Multi-agent Architecture Search via Agentic Supernet)**, applied to a **new benchmark**:

- **Benchmark**: HellaSwag (small validation subset)
- **Goal**: Run MaAS on a benchmark *not used in the original MaAS paper*, then
  - Find the **5 easiest examples where MaAS fails** and analyze why.
  - Find the **5 hardest examples where MaAS succeeds** and describe the multi‑agent systems MaAS discovers.

The original MaAS codebase lives in the `MaAS/` folder; this project adds a custom HellaSwag integration and analysis pipeline on top of it.

---

## Repository Structure

- `MaAS/` – upstream MaAS codebase (lightly modified).
  - `maas/ext/maas/benchmark/hellaswag.py` – HellaSwag benchmark class used by MaAS.
  - `maas/ext/maas/scripts/download_hellaswag.py` – downloads and formats a small HellaSwag validation subset to JSONL.
  - `run_hellaswag_maas.py` – minimal runner to train and test MaAS on HellaSwag without the MetaGPT config stack.
  - `maas/ext/maas/scripts/optimized/HellaSwag/` – optimized workflows, including:
    - `train/graph.py` – controller training graph for HellaSwag.
    - `test/graph.py` – evaluation graph for HellaSwag.
    - `test/round_1/*.csv` – HellaSwag result CSVs (scores and predictions).
  - `maas/ext/maas/scripts/analyze_hellaswag_results.py` – script to select:
    - 5 easiest failures (shortest questions with score 0).
    - 5 hardest successes (longest questions with score 1).
  - `maas/ext/maas/data/hellaswag_val.jsonl` – prepared HellaSwag subset.
  - `maas/ext/maas/data/hellaswag_traces.jsonl` – logged multi‑agent architecture traces for each problem.
- `MaAS/hellaswag_analysis.md` – written analysis of the 5 easiest failures and 5 hardest successes (root causes, operators, multi‑agent systems).

The original upstream MaAS documentation is in `MaAS/README.md`.

---

## Environment and Requirements

- **OS**: Windows 10
- **Python**: 3.12 (used in development)
- **GPU**: ~15 GB VRAM (project is tuned for small runs; no huge models loaded locally)
- **External services**:
  - OpenAI‑compatible LLM endpoint (e.g., OpenAI API)

### Python setup (recommended)

From the repo root:

```powershell
cd MaAS

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the HellaSwag MaAS Experiment

All commands below assume you are in the `MaAS` directory with the virtual environment activated:

```powershell
cd MaAS
.\.venv\Scripts\Activate.ps1
```

### 1. Download the HellaSwag subset

```powershell
python -m maas.ext.maas.scripts.download_hellaswag
```

This creates:

- `maas/ext/maas/data/hellaswag_val.jsonl`

### 2. Configure your LLM (OpenAI example)

In the same PowerShell session:

```powershell
$env:OPENAI_API_KEY = "sk-..."      # replace with your real key
$env:OPENAI_MODEL   = "gpt-4o-mini" # or another compatible chat model
```

If Groq or other OpenAI‑compatible providers are used instead, you can configure their base URL and key via environment variables as wired up in `run_hellaswag_maas.py`.

### 3. Train the MaAS controller on HellaSwag (Graph mode)

```powershell
python run_hellaswag_maas.py --mode Graph --sample 1 --batch_size 1 --max_examples 20
```

This:

- Trains the MaAS controller on a small subset of HellaSwag.
- Saves the trained controller parameters under:
  - `maas/ext/maas/scripts/optimized/HellaSwag/train/round_1/HellaSwag_controller_sample1.pth`
- Logs sampled multi‑agent architectures (operator layers) to:
  - `maas/ext/maas/data/hellaswag_traces.jsonl`

### 4. Evaluate MaAS on HellaSwag (Test mode)

```powershell
python run_hellaswag_maas.py --mode Test --sample 1 --batch_size 1 --max_examples 20
```

This:

- Loads the saved controller checkpoint.
- Evaluates on (up to) 20 HellaSwag problems.
- Writes a CSV of results (question, prediction, expected, score, logprob) into:
  - `maas/ext/maas/scripts/optimized/HellaSwag/test/round_1/*.csv`

### 5. Analyze easiest failures and hardest successes

```powershell
python -m maas.ext.maas.scripts.analyze_hellaswag_results `
  --log_root "maas/ext/maas/scripts/optimized/HellaSwag/test/round_1" `
  --trace_file "maas/ext/maas/data/hellaswag_traces.jsonl" `
  --top_k 5
```

The script prints:

- The 5 **easiest failures** (shortest questions with score 0), and
- The 5 **hardest successes** (longest questions with score 1),

including the **multi‑agent system structure** for each:

- Layers and operators (e.g., `Layer 0: Generate`).

The detailed written discussion of these 10 examples is in `MaAS/hellaswag_analysis.md`.

---

## What to Read for the Report

For writing up the project:

- **High‑level method and original MaAS design**:  
  - `MaAS/README.md` and the MaAS paper referenced there.
- **Your HellaSwag integration**:  
  - `maas/ext/maas/benchmark/hellaswag.py`  
  - `run_hellaswag_maas.py`  
  - `maas/ext/maas/scripts/analyze_hellaswag_results.py`
- **Final analysis**:  
  - `MaAS/hellaswag_analysis.md` – explains:
    - Root causes of the 5 easiest failures.
    - Why the 5 hardest successes work.
    - Whether failures are due to missing operators vs. search / base LLM behavior.

