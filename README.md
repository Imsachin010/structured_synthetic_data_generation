# Reliability-Aware Structured Synthetic Data Generation via Schema Enforcement and Layered Repair

**Author:** Sachin Mishra  
**Affiliation:** International Institute of Information Technology Bangalore (IIIT-B)  
**Email:** sachin.mishra@iiitb.ac.in
**ORCID:** [0009-0006-1184-8455](https://orcid.org/0009-0006-1184-8455)  
**Contact:** [GitHub @Imsachin010](https://github.com/Imsachin010)  
**License:** MIT  
**Paper:** *Reliability-Aware Structured Synthetic Data Generation via Schema Enforcement and Layered Repair* — submitted to SynthAI@SIGMOD 2026, Workshop on Synthetic Data Generation and Management for Building AI Systems, ACM SIGMOD 2026, Bengaluru, India.

> If you use this work, code, pipeline design, or experimental framework in any form, please cite the author and this repository. All design decisions, experimental methodology, evaluation metrics, and results in this repository are original work by Sachin Mishra.

---

## Overview

This repository contains the full implementation of a **reliability-aware structured synthetic data generation pipeline** — a controlled empirical study of how schema enforcement, multi-layer post-hoc repair, retrieval grounding, and multimodal conditioning affect structured output quality in locally-deployed LLMs.

The system is built around a single research question:

> *Can post-hoc schema enforcement and repair reliably produce valid structured JSON from a black-box local LLM — and what is the cost to semantic alignment?*

The answer, backed by controlled experiments across 100 queries and 4 pipeline conditions, is: **yes, with a +24 to +28 percentage point validity lift over unguided generation, and a measurable but modest alignment trade-off under retrieval and multimodal conditioning.**

All experiments run on a **consumer-grade NVIDIA RTX 3050 6 GB GPU** using fully local inference via Ollama — no cloud APIs, no fine-tuning.

---

## Key Results

| Stage | Retrieval | VLM | Enforce | Valid % | Alignment |
|---|---|---|---|---|---|
| Stage 0 — Baseline | None | — | ✗ | **70.0%** | — |
| Stage 1 — Embedding RAG | BGE embed | — | ✓ | **98.0%** | 0.938 |
| Stage 2 — Embedding + VLM | BGE embed | LLaVA 7B | ✓ | **98.0%** | 0.932 |
| Stage 3 — TF-IDF RAG | TF-IDF | — | ✓ | **94.0%** | 0.899 |

**Complex nested schema repair (N=30):**

| Metric | Value |
|---|---|
| Valid after repair | 22 / 30 (73.3%) |
| JSON parse repairs | 18 |
| Schema validation repairs | 7 |
| Unrecoverable | 8 / 30 (26.7%) |

**Three concrete findings:**
1. Schema enforcement raises validity from 70% to 94–98% without model fine-tuning (+24 to +28 pp)
2. Complex nested schemas trigger the repair loop on 83% of queries, with a clear two-layer failure taxonomy
3. Both TF-IDF retrieval (Δ=−0.039) and VLM conditioning (Δ=−0.006) introduce consistent alignment degradation

---

## System Architecture

```
Text Query
    │
    ├── (Optional) Matched Image → LLaVA 7B → VLM Caption
    │
    ▼
HyDE Query Expansion
    │
    ▼
Retrieval Module
    ├── Embedding Retrieval (BAAI/bge-small-en-v1.5)
    └── TF-IDF Retrieval (scikit-learn)
    │
    ▼
LLM Structured Generation (LLaMA3 8B via Ollama)
    │
    ▼
JSON Extraction
    │
    ▼
┌─────────────────────────────────────┐
│         Multi-Layer Repair Loop     │
│  Layer 1: JSON parse repair         │
│      → bracket fix, quote norm      │
│  Layer 2: Schema validation repair  │
│      → re-prompt with error trace   │
│  Fallback: Deterministic fill       │
│  Logging: repair_trace per attempt  │
└─────────────────────────────────────┘
    │
    ▼
Evaluation
    ├── JSON Validity Rate
    ├── Structural Consistency Score
    └── Query-Object Alignment Score (cosine, BGE embeddings)
```

---

## Core Contributions

### 1. Multi-Layer Post-Hoc Repair Loop
A two-layer repair mechanism that operates without white-box model access — making it deployable on any local inference backend (Ollama, llama.cpp, etc.):
- **Layer 1** — JSON parse repair: bracket balancing, quote normalization, trailing-comma removal
- **Layer 2** — Schema validation repair: re-prompts the model with explicit jsonschema error feedback
- **Fallback** — Deterministic minimum-field enforcement ensures schema-compliant output even when repair fails
- **Tracing** — Every repair attempt is logged with `layer`, `raw_output`, and `error` for post-hoc analysis

### 2. Strict Schema Enforcement with `additionalProperties: false`
The validator uses `Draft7Validator` with strict mode — rejecting any output containing fields outside the defined schema. This is more aggressive than lenient field-checking and directly causes repair loop activation, ensuring output cleanliness.

### 3. Controlled Ablation Framework
Four experimental conditions on identical 100-query benchmark with the same LLaMA3 8B Ollama backend:
- Baseline (no enforcement), Embedding RAG, Embedding + VLM, TF-IDF RAG
- Enables clean measurement of each component's independent contribution

### 4. Quantified Multimodal Drift
First controlled measurement of VLM-induced alignment degradation in a structured generation pipeline under matched image-query conditions. Demonstrates that even semantically relevant captions introduce distributional shift (Δ=−0.006).

### 5. Retrieval Noise Quantification in Structured Generation
Extends known retrieval noise findings specifically to structured output quality — TF-IDF lexical matching reduces alignment by Δ=−0.039 compared to embedding-based dense retrieval.

---

## Repository Structure

```
vlm_dataset_generation/
│
├── pipeline/
│   ├── generator.py          # Main generation function, repair loop, schema enforcement
│   ├── validator.py          # Draft7Validator schema validation (strict mode)
│   ├── evaluator.py          # Consistency score, alignment score (BGE cosine)
│   ├── rag_module.py         # TF-IDF retrieval (SimpleRAG)
│   ├── embedding_module.py   # Dense retrieval (BAAI/bge-small-en-v1.5)
│   ├── vlm_module.py         # LLaVA 7B caption extraction via Ollama
│   ├── hyde.py               # HyDE query expansion
│   └── prompt_templates.py   # BASE_PROMPT template
│
├── scripts/
│   ├── run_multimodal_stages.py      # Runs all 4 experimental conditions
│   ├── run_nested_schema_experiment.py  # Complex schema repair study (N=30)
│   ├── summarize_multimodal.py       # Generates comparison summary + Markdown
│   └── eval_metrics.py               # Computes per-stage metrics from JSON logs
│
├── pipeline/
│   └── complex_schema.py     # 3-level nested schema for repair stress test
│
├── images/
│   ├── dog.jpg               # Matched image for Stage 2 VLM queries
│   ├── car.jpg
│   └── football.jpg
│
├── data/
│   ├── raw_queries.json      # 100 structured scene-description queries
│   └── logs/
│       └── errors.txt        # Repair error trace log
│
├── experiments/
│   ├── stage0_baseline_ollama.json         # N=100, enforce=False
│   ├── stage1_embedding_only.json          # N=100, embedding RAG
│   ├── stage2_embedding_plus_vlm_matched.json  # N=15, matched pairs
│   ├── stage3_tfidf_rag.json               # N=100, TF-IDF RAG
│   ├── nested_schema_results.json          # N=30, complex schema
│   ├── multimodal_comparison_summary.json
│   └── multimodal_comparison.md
│
└── figures/
    ├── pipeline_architecture_figure1.svg
    ├── figure1_validity_rate.png
    ├── figure2_scores.png
    ├── figure3_alignment_repair.png
    └── figure4_failure_modes.png
```

---

## Setup

### Requirements

```
Python 3.10+
Ollama (latest) with llama3:8b and llava:7b pulled
NVIDIA GPU with 6GB+ VRAM
```

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- `ollama` — local LLM inference
- `sentence-transformers` — BGE embedding model
- `scikit-learn` — TF-IDF retrieval
- `jsonschema` — Draft7 schema validation
- `numpy`, `tqdm`

### Pull models via Ollama

```bash
ollama pull llama3:8b
ollama pull llava:7b
```

---

## Running Experiments

### Run all 4 stages (standard schema, N=100)

```bash
python scripts/run_multimodal_stages.py
```

Stages 0 and 2 are commented out by default — uncomment as needed in the script.

### Run complex nested schema repair study (N=30)

```bash
python scripts/run_nested_schema_experiment.py
```

### Compute metrics for any stage

```bash
python scripts/eval_metrics.py experiments/stage1_embedding_only.json
```

### Generate comparison summary

```bash
python scripts/summarize_multimodal.py
```

---

## Experimental Design Notes

- **HyDE expansion** is active in all stages including Stage 0 — Stage 0 is therefore a "no enforcement, HyDE-expanded" baseline, not a zero-context baseline
- **Stage 2** uses keyword-based image matching (`dog`, `car`, `football` in query text) — only the 15 intentionally constructed queries trigger VLM conditioning
- **Repair counts** in results are cumulative totals across all queries (up to `MAX_RETRIES=2` per layer per query)
- **Strict validation** uses `additionalProperties: false` — outputs with any fields beyond the schema fail validation and enter the repair loop
- **Structural consistency score** measures presence and type correctness of `objects` and `actions` arrays (0.5 each); `scene_description` is validated by the schema validator separately

---

## System Configuration Used

| Component | Specification |
|---|---|
| LLM | LLaMA3 8B via Ollama (temp=0.3, num_predict=256) |
| VLM | LLaVA 7B via Ollama |
| Embedding model | BAAI/bge-small-en-v1.5 |
| GPU | NVIDIA RTX 3050 6 GB |
| RAM | 16 GB |
| OS | Windows 11 |

---

<!-- ## Citation

If you use this code, pipeline design, experimental framework, evaluation metrics, or findings in any academic or commercial work, please cite:

```bibtex
@inproceedings{mishra2026reliability,
  title     = {Reliability-Aware Structured Synthetic Data Generation},
  author    = {Mishra, Sachin},
  booktitle = {Proceedings of the Workshop on Synthetic Data Generation and Management
               for Building AI Systems (SynthAI@SIGMOD '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Bengaluru, India},
  note      = {\url{https://github.com/Imsachin010/vlm_dataset_generation}}
}
```

--- -->

## Limitations and Future Work

This study has four acknowledged limitations:

1. Standard schema benchmark uses N=100 queries across a single schema type
2. Stage 2 multimodal experiment uses N=15 matched pairs — pilot only, not statistically significant
3. No formal comparison against constrained decoding systems (Outlines, Guidance) on the same queries — this is the primary direction for future work
4. Repair loop convergence is characterized empirically but not formally bounded

**Planned extensions:**
- Formal comparison against Outlines/Guidance on identical query set
- Hallucination rate metric (object not present in query or caption)
- Scale to 500+ queries across diverse schema families
- Schema complexity threshold analysis (nesting depth vs repair rate curve)
- Formal repair convergence bound

---

## License

MIT License — see `LICENSE` for details.

This repository and all its contents, including pipeline architecture, experimental design, evaluation metrics, and results, are the original work of Sachin Mishra (IIIT Bangalore). Any reproduction, derivative work, or academic use must include attribution to the author and a citation to the associated paper.

## Citation

If you use this work, please cite:

Sachin Mishra. 2026. Reliability-Aware Structured Synthetic Data Generation via Schema Enforcement and Layered Repair. In Workshop on Synthetic Data Generation and Management for Building AI Systems (SynthAI '26), May 31-June 05, 2026, Bengaluru, India. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3814574.3816747

---