# CommunityFact

**A Dynamic, Multilingual, Multi-domain Benchmark for Real-World Claim Verification**

CommunityFact is a fact-checking benchmark built from X (Twitter) Community Notes. Each claim is paired with a community-sourced verdict and supporting evidence URLs, enabling evaluation under realistic, time-stamped, multilingual conditions.

The first release (**v1, 2025**) covers:

- **Languages:** English (`en`), Spanish (`es`), French (`fr`), Japanese (`ja`), Portuguese (`pt`)
- **Domains:** Politics, Finance

> The public dataset card will be hosted on Hugging Face: `<Masked>`

Because Community Notes is updated continuously, the pipeline can be re-run on a newer snapshot to produce an updated CommunityFact release.

---

## Repository Layout

```
CommunityFact/
├── data/             # Dataset construction pipeline
├── experiments/      # Baselines and inference scripts
├── analysis/         # Notebooks and figures
└── requirements.txt
```

---

## Setup
Install the required packages/modules using: `requirements.txt`

Add API keys under `config/` (one plain-text file per provider: `openai_key.txt`, `google_key.txt`, `xai_key.txt`).

---

## Dataset

To run the benchmark, download `cf_v1_test.csv` from the Hugging Face dataset card into [data/CF/](data/CF/).

To build an updated snapshot from the latest Community Notes release, run the pipeline in [data/](data/) — see [data/run.sh](data/run.sh). Snapshot scope (date range, languages, sample sizes) is configurable via the arguments to `collector.py`.

Each row contains: `claimId`, `tweetId`, `noteId`, `claim`, `language`, `domain`, `label`, `noteTimeStamp`, and `evidenceURLs`.

---

## Benchmark

Baselines in [experiments/baselines.py](experiments/baselines.py) evaluate both open-weight and closed-source models under different settings.

```bash
cd experiments
python3 baselines.py <model_name> <prompt_type> <reasoning_effort> <web_search> <cuda_device>
```

Model outputs are in [experiments/outputs/](experiments/outputs/) and aggregated scores in [experiments/scores.csv](experiments/scores.csv). Evaluation is performed in [experiments/evaluation.ipynb](experiments/evaluation.ipynb).

---

## Analysis

Notebooks for dataset statistics, evidence-source analysis, and search-strategy comparisons live in [analysis/](analysis/), with paper figures in [analysis/graphs/](analysis/graphs/).

---

## Citation

```bibtex
<Masked for anonymity during review>
```