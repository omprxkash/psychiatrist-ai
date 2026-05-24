# Psychiatrist

A mental-health triage and clinical decision-support system built around a multi-agent pipeline. You fill in a PHQ-9 / GAD-7 screening and a short clinical note; the system gives you back a severity assessment, a safety verdict, and a suggested care plan — with the reasoning surfaced so a clinician could verify (or override) any of it.

> ⚠ Read [SAFETY.md](SAFETY.md) before going further. This is a portfolio engineering project, not a medical device. If you or someone you know is in crisis, the crisis-line directory is in SAFETY.md.

![Psychiatrist clinical UI](docs/images/hero.png)

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent_framework-FF6B35?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-severity_model-337AB7?style=flat)
![Safety tests](https://img.shields.io/badge/safety_tests-16%2F16_passing-brightgreen?style=flat)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)
![Not a medical device](https://img.shields.io/badge/NOT_a_medical_device-%E2%9A%A0_portfolio_only-orange?style=flat)

---

## Contents

- [Quickstart](#quickstart--three-commands-no-gpu-required)
- [What it does](#what-it-does)
- [Tech stack](#tech-stack)
- [How it works](#how-it-works)
- [Why I built it](#why-i-built-it)
- [Running it](#running-it)
- [What's inside](#whats-inside)
- [Honest limitations](#honest-limitations)
- [Skill matrix](#skill-matrix)
- [Tests](#tests)

---

## Quickstart — three commands, no GPU required

```bash
pip install -e ".[dev]"
python data/generate.py --n 50000 --out data/processed
streamlit run serving/app.py --server.port 8501
```

Open `http://localhost:8501`. The UI works in heuristic-fallback mode without Ollama, trained models, or a Java runtime. See [Running it](#running-it) for the full pipeline.

---

## What it does

Fill in the PHQ-9 + GAD-7 screening forms, add a brief clinical note, and submit. Behind the scenes:

- A severity classifier predicts a PHQ-9 band (none / mild / moderate / moderately severe / severe).
- A fine-tuned MentalBERT model reads the narrative for clinical signals and flags suicidal ideation if it's present.
- A hybrid RAG retriever pulls DSM-5 criteria and PubMed abstracts relevant to the detected signals.
- A care-plan agent suggests next steps based on the severity, signals, and retrieved literature.
- A **Safety-Critic agent** runs last and can override any of the above — combining deterministic rules (regex patterns for SI language, plan/intent keywords) with an LLM verifier that double-checks the reasoning. If it escalates, the UI surfaces a crisis banner and helpline numbers.

The Safety-Critic is the part I spent the most time on. The hard rule is that the suicide-risk regression suite has to hit **100% recall**, every run, no exceptions. There is a curated test set in `tests/safety/` that gates every commit.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Agent orchestration | LangGraph | Deterministic DAG — testable node-by-node; interviewers ask about it |
| LLM | Llama-3.2-3B via Ollama | Fully local; zero paid API calls in any code path |
| NLP / SI detection | MentalBERT int8 | Domain-pretrained on mental health text; outperforms generic 7B on SI detection on CPU |
| Severity model | XGBoost + PyTorch MLP | XGBoost wins on tabular PHQ-9/GAD-7; MLP included as a neural baseline |
| RAG | FAISS + BM25 hybrid | Dense retrieval for semantic match + BM25 for clinical keyword precision |
| UI | Streamlit | Rapid prototype with oklch design tokens — less generic than stock Streamlit |
| API | FastAPI | Pydantic validates PHQ-9/GAD-7 item ranges (0–3); per-IP rate limiting |
| MLOps | MLflow + Evidently | Experiment registry + drift detection; wired for auto-retrain on drift |
| Data | Synthetic generator (PySpark ETL optional) | No real PHI; calibrated to NHANES/NIMHANS prevalence tables |

## How it works

![Architecture](images/architecture.png)

<details>
<summary>ASCII fallback (terminal / no image rendering)</summary>

```
                    ┌─────────────────────────────────────────────┐
                    │              Streamlit UI                    │
                    │   structured intake + narrative input        │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │           LangGraph orchestrator             │
                    │                                              │
                    │  Screening → Risk → DSM/Lit → Care Plan      │
                    │                                  │           │
                    │                                  ▼           │
                    │                         ┌────────────────┐   │
                    │                         │ Safety-Critic  │   │
                    │                         │  (det + LLM)   │   │
                    │                         └────────────────┘   │
                    └──┬──────────┬──────────┬──────────┬──────────┘
                       │          │          │          │
                  ┌────▼───┐ ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
                  │XGBoost │ │Mental- │ │ FAISS  │ │Ollama  │
                  │severity│ │ BERT   │ │ DSM +  │ │Llama3.2│
                  │        │ │        │ │ PubMed │ │  3B    │
                  └────────┘ └────────┘ └────────┘ └────────┘
```

</details>

The orchestration is a straight LangGraph DAG: screening → risk → DSM/literature → care plan → safety critic → end. The Safety-Critic always runs last. Its verdict (`routine` / `monitor` / `escalate`) is what the UI surfaces; everything else is supporting evidence.

For a deeper technical walkthrough — per-agent responsibilities, the two-layer Safety-Critic design, fallback patterns — see [ARCHITECTURE.md](ARCHITECTURE.md).

## Why I built it

I wanted to learn agentic AI properly — not just by reading about LangGraph but by building something where the agents actually have to coordinate around a real constraint. Psychiatry turned out to be a good fit: it is heavily text-based (so it runs on a laptop without a GPU), the clinical structure is well-defined (PHQ-9, GAD-7, DSM-5), and there is a clear safety constraint that forces you to think about agent failure modes instead of just chaining LLM calls.

The other reason: I wanted one project that exercises the whole stack — classical ML, NLP, RAG, agent orchestration, MLOps — coherently, instead of five disconnected demos.

## Running it

**Minimal path** (no Ollama, no Java, no GPU — heuristic fallback mode):

```bash
pip install -e ".[dev]"
pytest tests/safety/ -v -m safety   # 16/16 must pass
python data/generate.py --n 50000 --out data/processed
streamlit run serving/app.py --server.port 8501
```

**Full pipeline** (LLM-generated care plans + live PubMed citations):

```bash
ollama pull llama3.2
python -m models.train --task severity --model xgboost
python -m models.train --task clinical_nlp --model mentalbert
python -m rag.ingest_dsm_summaries --out rag/indexes/dsm
```

`make train` runs on CPU. The MentalBERT fine-tune takes a few hours for a full pass — use `--max-train-samples 5000` for a quick dev run.

**Or use the CLI** (installed with `pip install -e "."`):

```bash
psychiatrist serve       # Streamlit on :8501
psychiatrist api         # FastAPI on :8000
psychiatrist safety      # safety regression suite
psychiatrist data        # generate synthetic records
```

## What's inside

| Path | Purpose |
|---|---|
| [SAFETY.md](SAFETY.md) | Read first — ethics, limitations, crisis lines |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep-dive: per-agent design, two-layer Safety-Critic, fallback patterns |
| [agents/](agents/) | LangGraph DAG + 5 agent nodes including `safety_critic.py` |
| [models/](models/) | Severity classifier (XGBoost + MLP) and clinical NLP (MentalBERT fine-tune) |
| [rag/](rag/) | Hybrid FAISS + BM25 retriever over DSM-5 summaries and PubMed |
| [serving/](serving/) | Streamlit UI (`app.py`) + FastAPI service (`api.py`) |
| [spark_jobs/](spark_jobs/) | PySpark ETL for synthetic data generation and Reddit cohort |
| [monitoring/](monitoring/) | Evidently drift reports + immutable per-request safety audit log |
| [tests/safety/](tests/safety/) | 16-case SI regression suite — 100% recall is the release gate |
| [data/](data/) | Synthetic PHQ-9/GAD-7 generator + DVC-tracked data sources |
| [psychiatrist/](psychiatrist/) | CLI entry point (`psychiatrist serve / api / safety / data`) |

## Honest limitations

- **All training data is synthetic.** PHQ-9 / GAD-7 distributions calibrated to published prevalence tables (Kroenke 2001; Manea 2012), but there is no real patient data in this repo. The severity classifier learns the synthetic distribution, not the real-world one.
- **Not clinically validated.** Every piece of output should be read as "what a model trained on synthetic data thinks", not as a clinical recommendation.
- **Fallback heuristics matter.** When Ollama / MentalBERT / the trained XGBoost are not available, the system falls back to keyword matching and rule-based scoring. The Safety-Critic deterministic rules always fire — that is by design.
- **No PII anywhere.** The audit log stores feature vectors and predictions only; the Streamlit app does not persist anything you type into the narrative box.

## What's still in flight

| Component | Status | Notes |
|---|---|---|
| Model training | Not run | `make train` to produce XGBoost + MentalBERT checkpoints |
| RAG indexes | Not built | `make rag-index` for FAISS + BM25 over DSM + PubMed |
| Docker | Scaffolded | `docker/` dir exists; Dockerfile + compose TBD |
| Deployment | Not started | Options: Hugging Face Spaces, Render, AWS ECS — decision pending |
| MLflow server | Referenced only | Dependency installed; no remote server configured |
| DVC remote | Referenced only | Dependency installed; no `.dvc/` config committed |

## Tests

```bash
pytest tests/safety/ -v -m safety   # 16-case safety regression — must be 16/16
pytest tests/ -v --cov=agents       # full suite
ruff check . && mypy agents models  # lint
```

CI runs `make safety` and `make test` on every push (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Skill matrix

Where to find evidence for common data-scientist JD bullets:

| Skill area | Where to look |
|---|---|
| ML model training & evaluation | [models/severity/](models/severity/) — XGBoost, LightGBM, Torch MLP; [MODEL_CARD.md](models/severity/MODEL_CARD.md) for fairness slices |
| Transformer fine-tuning | [models/clinical_nlp/](models/clinical_nlp/) — MentalBERT fine-tune + int8 quantization |
| Retrieval-augmented generation | [rag/](rag/) — hybrid FAISS + BM25 retriever; DSM-5 and PubMed ingestion |
| LLM agent orchestration | [agents/](agents/) — 5-node LangGraph DAG with graceful dependency fallback |
| Safety-critical system design | [agents/safety_critic.py](agents/safety_critic.py) + [tests/safety/](tests/safety/) — deterministic-first, 100% recall gate |
| MLOps / experiment tracking | [models/train.py](models/train.py) + [monitoring/](monitoring/) — MLflow registry, Evidently drift |
| Data engineering / PySpark | [spark_jobs/](spark_jobs/) — synthetic data generator + Reddit mental-health ETL |
| REST API design | [serving/api.py](serving/api.py) — FastAPI with Pydantic validation + rate limiting |
| UI / product sense | [serving/app.py](serving/app.py) — Streamlit with oklch design tokens, sticky safety banner |

## License

MIT — see [LICENSE](LICENSE). DSM-5-TR is copyrighted by the American Psychiatric Association and is not redistributed here; see SAFETY.md for details.
