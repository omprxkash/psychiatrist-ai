<div align="center">

# Psychiatrist

A mental-health triage and clinical decision-support system built around a multi-agent pipeline.
Fill in a PHQ-9 / GAD-7 screening form and a short clinical note — get back a severity assessment, a safety verdict, and a suggested care plan.

![Psychiatrist clinical UI](docs/images/hero.png)

[![CI](https://github.com/omprxkash/data-scientist/actions/workflows/ci.yml/badge.svg)](https://github.com/omprxkash/data-scientist/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent_framework-FF6B35?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-severity_model-337AB7?style=flat)
![Safety tests](https://img.shields.io/badge/safety_tests-16%2F16_passing-brightgreen?style=flat)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

</div>

---

## Quickstart — three commands, no GPU required

```bash
pip install -e ".[dev]"
python data/generate.py --n 50000 --out data/processed
streamlit run serving/app.py --server.port 8501
```

Open `http://localhost:8501`. Works in fallback mode without Ollama, trained models, or a Java runtime. See [Running the full pipeline](#running-the-full-pipeline) for the complete setup.

---

## What it does

You fill in a standard depression and anxiety screening form (PHQ-9 and GAD-7 — two widely used questionnaires in mental health clinics) along with a short clinical note. A five-step pipeline runs automatically and returns three things: a severity label, a safety verdict, and a suggested care plan.

The pipeline runs in this order:

```
Screening  →  Risk Assessment  →  DSM + PubMed lookup  →  Care Plan  →  Safety Critic
```

| What you get | How it's produced |
|:---|:---|
| Severity band (none / mild / moderate / severe) | Machine learning model trained on PHQ-9 and GAD-7 scores |
| Safety verdict (routine / monitor / escalate) | Deterministic rules checked first; language model adds context |
| Care plan suggestions | Generated from severity, detected symptoms, and retrieved clinical literature |

A bit more detail on each component:

- **Severity classifier** — XGBoost (a gradient-boosted decision tree model) trained on PHQ-9 and GAD-7 item scores predicts which severity band the patient falls into.
- **MentalBERT** — a BERT-based language model pre-trained on mental health text, quantized to run on CPU. Reads the clinical narrative and flags suicidal ideation signals.
- **Hybrid retrieval (RAG)** — when generating the care plan, the system searches a local index of DSM-5 criteria summaries and PubMed abstracts to ground the suggestions in clinical literature. Uses FAISS for semantic search and BM25 for keyword matching, then merges the results.
- **Care plan agent** — uses a local language model (Llama 3.2, run via Ollama) to synthesise severity, symptoms, and retrieved literature into actionable suggestions. Falls back to templated recommendations if Ollama isn't running.
- **Safety Critic** — always runs last. Deterministic regex rules check the narrative for suicidal ideation keywords first; if nothing fires, the language model adds a nuanced second pass. The deterministic layer cannot be overridden by the LLM. A 16-case regression suite runs on every commit and must pass at 100% recall.

---

## How it works

![Architecture](images/architecture.png)

<details>
<summary>Architecture diagram (text version)</summary>

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

The pipeline is built with LangGraph, which lets you define the agent flow as an explicit directed graph — each node is a separate Python class that reads from shared state, does its job, and writes back. This makes it straightforward to test individual nodes in isolation and to see exactly what information flows between agents.

Every node fails gracefully. If Ollama isn't running, the care plan agent uses templates. If the XGBoost checkpoint hasn't been trained yet, the severity model falls back to rule-based PHQ-9 banding. The Safety Critic's deterministic rules always fire regardless of what else is available.

For the full technical walkthrough — per-agent design, the two-layer Safety Critic, fallback patterns — see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Agent orchestration | LangGraph | Explicit graph topology — each node is independently testable |
| LLM | Llama-3.2-3B via Ollama | Fully local; no paid API calls in any code path |
| NLP / SI detection | MentalBERT int8 | Domain-pretrained on mental health text; runs on CPU |
| Severity model | XGBoost + PyTorch MLP | XGBoost wins on tabular PHQ-9/GAD-7; MLP included as a neural baseline |
| Retrieval | FAISS + BM25 hybrid | Dense semantic search + keyword matching, merged with reciprocal rank fusion |
| UI | Streamlit | Custom CSS design tokens — less generic than default Streamlit styling |
| API | FastAPI | Pydantic validates PHQ-9/GAD-7 item ranges (0–3); per-IP rate limiting |
| MLOps | MLflow + Evidently | Experiment tracking + data drift detection; wired for auto-retrain on drift |
| Data | Synthetic pandas generator + PySpark ETL | No real patient data anywhere; distributions calibrated to published prevalence tables |

---

## Why I built it

I wanted to get hands-on with agentic AI rather than just reading about it — something where the agents have to coordinate around a real constraint rather than just chain together LLM calls.

Mental health triage turned out to be a good fit. The clinical structure is well-defined (PHQ-9, GAD-7, DSM-5 criteria), the domain is text-heavy so it runs on a laptop without a GPU, and there is a concrete safety constraint — a patient flagged for suicidal ideation must be escalated, full stop. That constraint forced me to think about agent failure modes properly: what happens when the LLM is offline? What happens when the model hasn't been trained yet? Can the safety layer be overridden by a hallucinating care-plan agent? The answer to that last one is no — deterministic rules fire first and cannot be reversed downstream.

The other reason: I wanted one project that exercises the full stack — classical ML, clinical NLP, retrieval-augmented generation, agent orchestration, and MLOps — in a coherent way, rather than five disconnected toy demos.

---

## What's inside

| Path | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep-dive: per-agent design, two-layer Safety-Critic, fallback patterns |
| [agents/](agents/) | LangGraph DAG + 5 agent nodes including `safety_critic.py` |
| [models/](models/) | Severity classifier (XGBoost + MLP) and clinical NLP (MentalBERT fine-tune) |
| [rag/](rag/) | Hybrid FAISS + BM25 retriever over DSM-5 summaries and PubMed |
| [serving/](serving/) | Streamlit UI (`app.py`) + FastAPI service (`api.py`) |
| [spark_jobs/](spark_jobs/) | PySpark ETL for synthetic data generation and Reddit cohort |
| [monitoring/](monitoring/) | Evidently drift reports + immutable per-request safety audit log |
| [tests/safety/](tests/safety/) | 16-case suicidal ideation regression suite — 100% recall is the release gate |
| [data/](data/) | Synthetic PHQ-9/GAD-7 generator + DVC-tracked data sources |
| [psychiatrist/](psychiatrist/) | CLI entry point (`psychiatrist serve / api / safety / data`) |

---

## Tests

```bash
pytest tests/safety/ -v -m safety   # 16-case safety regression — must be 16/16
pytest tests/ -v --cov=agents       # full suite
ruff check . && mypy agents models  # lint
```

CI runs on every push — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

<details>
<summary>Running the full pipeline (Ollama + trained models)</summary>

```bash
ollama pull llama3.2
python -m models.train --task severity --model xgboost
python -m models.train --task clinical_nlp --model mentalbert
python -m rag.ingest_dsm_summaries --out rag/indexes/dsm
```

`make train` runs on CPU. The MentalBERT fine-tune takes a few hours for a full pass — use `--max-train-samples 5000` for a quick dev run.

Or use the CLI (installed with `pip install -e "."`):

```bash
psychiatrist serve       # Streamlit on :8501
psychiatrist api         # FastAPI on :8000
psychiatrist safety      # safety regression suite
psychiatrist data        # generate synthetic records
```

</details>

<details>
<summary>Honest limitations</summary>

- **All training data is synthetic.** PHQ-9 / GAD-7 distributions are calibrated against published prevalence tables (Kroenke et al. 2001; Manea et al. 2012), but there is no real patient data in this repo.
- **Not clinically validated.** Every output from this system should be read as "what a model trained on synthetic data produces" — not as a clinical recommendation.
- **Fallbacks matter.** When Ollama, MentalBERT, or the trained XGBoost checkpoint are unavailable, the system falls back to keyword matching and rule-based scoring. The Safety Critic's deterministic rules always fire regardless.
- **No PII anywhere.** The audit log stores only feature vectors and model outputs — no patient-identifiable data.

</details>

<details>
<summary>What's still in progress</summary>

| Component | Status | Notes |
|---|---|---|
| Model training | Not run | `make train` to produce XGBoost + MentalBERT checkpoints |
| RAG indexes | Not built | `make rag-index` for FAISS + BM25 over DSM + PubMed |
| Docker | Scaffolded | `docker/` dir exists; Dockerfile + compose TBD |
| Deployment | Not started | Options: Hugging Face Spaces, Render, AWS ECS |
| MLflow server | Referenced only | Dependency installed; no remote server configured |

</details>

---

## License

MIT — see [LICENSE](LICENSE).
