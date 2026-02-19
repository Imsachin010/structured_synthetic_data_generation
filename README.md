# Reliability-Aware Multimodal Structured Generation Framework

A modular, experiment-driven framework for **structured JSON generation using LLMs and VLMs**, with schema enforcement, multimodal grounding, retrieval control, and automated evaluation.

---

## 📌 Motivation

LLMs are increasingly used for synthetic dataset generation and structured annotation. However, raw LLM outputs suffer from:

- ❌ Malformed JSON
- ❌ Missing schema fields
- ❌ Hallucinated entities
- ❌ Retrieval contamination
- ❌ Multimodal drift

This project converts uncontrolled LLM/VLM generation into a:

> Controlled, validated, repair-aware, and experimentally measurable system.

---

## 🎯 Project Objective

Design a **robust multimodal structured generation pipeline** that:

1. Enforces strict JSON schema compliance  
2. Automatically repairs malformed outputs  
3. Controls retrieval grounding  
4. Integrates image-based VLM context  
5. Quantifies semantic alignment  
6. Enables controlled ablation experiments  

---

## 🏗️ System Architecture

```

Text Query
↓
(Optional) Matched Image
↓
VLM Caption Extraction (LLaVA 7B)
↓
HyDE Query Expansion
↓
Retrieval Module
├── TF-IDF
└── Embedding Retrieval (BGE-small)
↓
LLM Structured JSON Generation (LLaMA3 8B)
↓
JSON Extraction
↓
Automatic Repair Loop
↓
Schema Validation
↓
Deterministic Enforcement
↓
Evaluation Metrics
↓
Stage-wise Experiment Logging

```

---

## ⚙️ Core Features

### ✅ Strict Structured Generation
- JSON schema enforcement
- Nested attribute validation
- Required field guarantees
- Deterministic fallback filling

### 🔁 Multi-Layer Repair Mechanism
- Malformed JSON self-correction
- Validation-aware repair loop
- Deterministic schema completion

### 🔍 Retrieval Grounding
- TF-IDF retrieval
- Embedding-based retrieval (BAAI/bge-small-en-v1.5)
- Similarity threshold filtering
- Retrieval ablation support

### 🖼 Multimodal Integration
- LLaVA 7B image caption extraction
- Matched image-query experiments
- Cross-modal grounding evaluation
- Multimodal drift analysis

### 📊 Evaluation Metrics
- Structural consistency score
- Query-object alignment score
- JSON validity rate
- Stage-wise comparison statistics

### 🧪 Controlled Experimental Framework
- Stage-based execution
- Embedding vs TF-IDF comparison
- Vision vs non-vision comparison
- Automated Markdown summary generation

---

## 📂 Updated Project Structure

```

vlm_dataset_generation/
│
├── pipeline/
│   ├── generator.py
│   ├── rag_module.py
│   ├── embedding_module.py
│   ├── vlm_module.py
│   ├── hyde.py
│   ├── validator.py
│   ├── evaluator.py
│
├── scripts/
│   ├── run_multimodal_stages.py
│   ├── summarize_multimodal.py
│
├── images/
│   ├── dog.jpg
│   ├── car.jpg
│   └── football.jpg
│
├── data/
│   ├── raw_queries.json
│   └── logs/
│
├── experiments/
│   ├── stage1_embedding_only.json
│   ├── stage2_embedding_plus_vlm_matched.json
│   ├── stage3_tfidf_rag.json
│   ├── multimodal_comparison_summary.json
│   └── multimodal_comparison.md
│
└── README.md

````

---

## 🚀 How to Run Multimodal Study

### 1️⃣ Run Controlled Stage Experiments

```bash
python scripts/run_multimodal_stages.py
````

This executes:

* Stage 1 — Embedding Retrieval Only
* Stage 2 — Embedding + Matched VLM Grounding
* Stage 3 — TF-IDF Retrieval

---

### 2️⃣ Generate Stage Comparison

```bash
python scripts/summarize_multimodal.py
```

Outputs:

```
experiments/multimodal_comparison_summary.json
experiments/multimodal_comparison.md
```

---

## 🔬 Multimodal Grounding Study

We conduct controlled experiments across three configurations:

| Stage   | Retrieval | Vision | Purpose                       |
| ------- | --------- | ------ | ----------------------------- |
| Stage 1 | Embedding | ❌      | Baseline semantic grounding   |
| Stage 2 | Embedding | ✅      | Multimodal grounding impact   |
| Stage 3 | TF-IDF    | ❌      | Retrieval comparison baseline |

### Example Observations

* Structural reliability remains stable (repair system effective).
* Naïve multimodal fusion can introduce semantic drift.
* Controlled retrieval improves alignment stability.
* Vision grounding requires relevance filtering to prevent contamination.

---

## 🧠 Key Experimental Insight

This project demonstrates that:

> Structural reliability can be enforced deterministically, but semantic grounding requires careful multimodal and retrieval control.

The study highlights trade-offs between:

* Retrieval breadth vs semantic precision
* Vision fusion vs contextual contamination
* Embedding retrieval vs lexical retrieval

---

## 💻 System Configuration

* **LLM:** LLaMA3 8B (Ollama)
* **VLM:** LLaVA 7B
* **Embedding Model:** BAAI/bge-small-en-v1.5
* **GPU:** RTX 3050 6GB
* **RAM:** 16GB
* Designed for consumer-grade hardware experimentation

---

## 📌 Scope

This project focuses on:

* Structured scene annotation
* Multimodal grounding analysis
* Retrieval-controlled generation
* Reliability-aware design

It does NOT:

* Fine-tune foundation models
* Claim production deployment
* Use large-scale datasets

---

## 🔮 Future Directions

* Embedding similarity gating for caption inclusion
* Hallucination rate metric
* Object-level grounding verification
* Hybrid retrieval fusion
* Workshop paper submission

---

## 📜 License

MIT License

```
