<div align="center">

# Psychiatrist

Fill in a depression/anxiety screening form and write a short clinical note.
Get back a severity label, a safety verdict, and a suggested care plan.

![Psychiatrist clinical UI](docs/images/hero.png)

[![CI](https://github.com/omprxkash/data-scientist-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/omprxkash/data-scientist-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent_framework-FF6B35?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-severity_model-337AB7?style=flat)
![Safety tests](https://img.shields.io/badge/safety_tests-16%2F16_passing-brightgreen?style=flat)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

</div>

---

## Why I built this

I got tired of reading about agentic AI and wanted to actually build something with it — something where the agents have a real reason to exist, not just a chain of LLM calls dressed up as a pipeline.

Mental health triage clicked for a few reasons. The clinical structure is already well-defined (PHQ-9 and GAD-7 are standard questionnaires used in actual clinics, DSM-5 criteria are public knowledge). The domain is text-heavy, so it runs fine on CPU. And there's one constraint that can't be fudged: if someone mentions suicidal ideation, the system has to escalate — full stop, no negotiation with a language model.

That constraint turned out to be the most interesting part to build. What if Ollama is offline? What if the XGBoost model hasn't been trained yet? Can a hallucinating care-plan agent talk the safety layer into a quieter verdict? I spent more time thinking through those failure modes than any other part of the project, and the answers — deterministic rules that fire before any LLM sees the text, fallbacks at every step, a regression suite that blocks merges — ended up shaping the whole design.

The other thing I wanted was one project that touches everything end-to-end: classical ML, clinical NLP, RAG, agent orchestration, and MLOps. Not five disconnected notebooks — one thing that actually holds together.

---

## Quickstart

No GPU needed. No Ollama. No pre-trained models. Everything degrades gracefully to rules and templates.

```bash
pip install -e ".[dev]"
python data/generate.py --n 50000 --out data/processed
streamlit run serving/app.py --server.port 8501
```

Open `http://localhost:8501` and you're in. For the full setup with Ollama and trained models, see [Running the full pipeline](#running-the-full-pipeline) at the bottom.

---

## How it works

Five agents run in a fixed sequence on every request. No branching, no retries — each one reads from a shared state object, does its job, and passes the baton. The state fills up as it moves through, so by the time the Safety Critic runs it has everything the previous agents produced.

```mermaid
flowchart TD
    UI([Streamlit UI\nPHQ-9 · GAD-7 · clinical note])
    UI --> screening

    subgraph pipeline["LangGraph pipeline"]
        screening[Screening\nvalidate inputs · compute totals · flag Q9]
        risk[Risk Assessment\nXGBoost severity · MentalBERT SI detection]
        dsm[DSM + Literature\nFAISS semantic + BM25 keyword retrieval]
        care[Care Plan\nOllama Llama-3.2 · templated fallback]
        safety[Safety Critic\nLayer 1: deterministic rules\nLayer 2: LLM verifier]

        screening --> risk --> dsm --> care --> safety
    end

    safety --> verdict([Verdict\nroutine · monitor · escalate])

    subgraph models["Models"]
        xgb[XGBoost\nseverity classifier]
        bert[MentalBERT int8\nSI detection]
        faiss[FAISS + BM25\nDSM · PubMed]
        llm[Ollama\nLlama-3.2-3B]
    end

    risk -.-> xgb
    risk -.-> bert
    dsm -.-> faiss
    care -.-> llm
```

### The five agents

**Screening** is the simplest one — it validates that all the questionnaire scores are in the right range (0–3 per item), adds up the totals, and checks whether PHQ-9 item 9 is non-zero. That last item asks specifically about suicidal ideation, and if it's anything above zero, a flag gets set that the Safety Critic picks up regardless of what else happens.

**Risk Assessment** runs two things. First, an XGBoost classifier on the questionnaire scores to predict severity band: none, mild, moderate, or severe. Second, MentalBERT — a BERT model pre-trained specifically on mental health text, compressed to run on CPU — reads the clinical note and produces a suicidal ideation probability. If MentalBERT isn't loaded, it falls back to regex keyword matching. Not ideal, but it doesn't silently return nothing.

**DSM + Literature** searches two local indexes — paraphrased DSM-5 criteria summaries and PubMed psychiatry abstracts — using a combination of FAISS (semantic) and BM25 (keyword), then merges and re-ranks the results. This is what gives the care plan something to cite rather than generating suggestions from scratch.

**Care Plan** takes the severity band, the detected symptoms, and the retrieved passages, and asks Llama 3.2 (via Ollama) to turn that into actionable suggestions. If Ollama isn't running, it uses templated recommendations keyed to severity band. Either way you get something out, not an error.

**Safety Critic** is the one I spent the most time on. It always runs last and it runs two layers, in order:

- *Deterministic first:* checks the narrative sentence by sentence for suicidal ideation keywords, with exclusions for reported speech patterns ("the patient denied any ideation"). If this fires, the verdict is `escalate`, Layer 2 is skipped, and nothing downstream can change it.
- *LLM second:* if Layer 1 didn't fire, the LLM reviews the care plan for hallucinated symptoms, overconfident claims, and anything it shouldn't be asserting. It can upgrade a `routine` to `monitor`, but it can't downgrade an `escalate`. For moderate or above severity, the floor is `monitor`.

There's a 16-case regression suite covering both obvious suicidal ideation ("I want to end my life") and subtle patterns ("I've been thinking there's no point anymore"). It runs on every push and has to hit 100% recall. That suite is what I actually trust — not the LLM.

---

## What's under the hood

| Layer | What I used | Why I chose it |
|---|---|---|
| Agent orchestration | LangGraph | Lets you define the pipeline as an explicit graph — each node is a Python class you can test on its own |
| LLM | Llama-3.2-3B via Ollama | Fully local, no API keys, no cost per call |
| NLP / SI detection | MentalBERT int8 | Pre-trained on mental health text, not general-purpose BERT. Actually better for this domain. |
| Severity model | XGBoost + PyTorch MLP | XGBoost is genuinely better on tabular questionnaire data. MLP is there as a comparison baseline. |
| Retrieval | FAISS + BM25 hybrid | Semantic search alone misses exact terminology; BM25 alone misses meaning. Both together works better. |
| UI | Streamlit | Fast to build, and with custom CSS it doesn't look like every other Streamlit app |
| API | FastAPI | Pydantic validates item ranges; per-IP rate limiting keeps it from being hammered |
| MLOps | MLflow + Evidently | Experiment tracking + drift monitoring; wired to flag when the input distribution shifts |
| Training data | Synthetic PHQ-9/GAD-7 | No real patient data anywhere in this repo |

---

## What's in here

| Path | What it is |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Goes deeper — per-agent design, Safety Critic internals, retriever implementation |
| [agents/](agents/) | LangGraph DAG and all five agent classes |
| [models/](models/) | XGBoost + MLP training, MentalBERT fine-tuning and quantization |
| [rag/](rag/) | FAISS + BM25 retriever, DSM and PubMed ingestion |
| [serving/](serving/) | Streamlit UI and FastAPI service |
| [spark_jobs/](spark_jobs/) | PySpark ETL for synthetic data generation |
| [monitoring/](monitoring/) | Evidently drift reports and a per-request safety audit log |
| [tests/safety/](tests/safety/) | The 16-case suicidal ideation suite — 100% recall is the release gate |
| [data/](data/) | Synthetic data generator |

---

## Tests

```bash
pytest tests/safety/ -v -m safety   # the suite that matters most — needs to be 16/16
pytest tests/ -v --cov=agents       # full test run
ruff check . && mypy agents models  # lint and type check
```

The safety suite runs in CI without Ollama or any trained checkpoints — the agents fall back to rules automatically, which is the whole point.

---

<details>
<summary>Running the full pipeline (with Ollama and trained models)</summary>

```bash
ollama pull llama3.2

# Train the severity models
python -m models.train --task severity --model xgboost
python -m models.train --task clinical_nlp --model mentalbert

# Build the RAG indexes
python -m rag.ingest_dsm_summaries --out rag/indexes/dsm
```

`make train` wraps all of this. MentalBERT fine-tuning takes a few hours on CPU — add `--max-train-samples 5000` if you just want to verify it runs.

After `pip install -e "."`, there's also a CLI:

```bash
psychiatrist serve    # Streamlit on :8501
psychiatrist api      # FastAPI on :8000
psychiatrist safety   # run the safety regression suite
psychiatrist data     # generate synthetic records
```

</details>

<details>
<summary>Honest caveats</summary>

**It's trained on synthetic data.** The PHQ-9 and GAD-7 score distributions are calibrated against published prevalence tables (Kroenke et al. 2001, Manea et al. 2012), but there's no real patient data here. The outputs reflect what a model trained on plausible synthetic records produces — which isn't the same as clinical validation.

**Don't use this for actual clinical decisions.** It's a portfolio project demonstrating how these components fit together, not a medical tool.

**The fallbacks matter.** Without Ollama, without MentalBERT, without the trained XGBoost checkpoint — the system still runs. The Safety Critic's deterministic rules still fire. It's worse, but it's not broken.

**No patient data anywhere.** The audit log stores feature vectors and model predictions only.

</details>

<details>
<summary>What's still left to build</summary>

| Thing | Where it's at |
|---|---|
| Model checkpoints | Not in the repo — run `make train` to produce them locally |
| RAG indexes | Same — run `make rag-index` |
| Docker | Directory scaffolded, Compose file in progress |
| Deployment | Deciding between Hugging Face Spaces and Render |

</details>

---

## License

MIT — see [LICENSE](LICENSE).
