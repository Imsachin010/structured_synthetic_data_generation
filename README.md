# Structured LLM/VLM Dataset Generation & Evaluation Pipeline

A modular framework for **reliable structured dataset generation using LLMs**, with schema enforcement, grounding control, and automated evaluation.

---

## 📌 Motivation

Large Language Models (LLMs) are frequently used for synthetic dataset generation. However, raw LLM outputs are:

- Structurally inconsistent  
- Prone to malformed JSON  
- Susceptible to hallucination  
- Difficult to evaluate systematically  

This project transforms uncontrolled LLM generation into a **validated, repair-aware, measurable pipeline** suitable for structured annotation workflows.

---

## 🎯 Core Idea

Instead of simply prompting an LLM and saving outputs, this system enforces:

1. Structured JSON schema validation  
2. Automatic JSON repair  
3. Validation-aware retry loops  
4. Deterministic structural enforcement  
5. Grounded generation via RAG + HyDE  
6. Semantic alignment scoring  
7. Automated ablation experiments  

The result is a **controlled structured generation system**, not just prompt engineering.

---

## 🏗️ Architecture

```

User Query
↓
HyDE (Query Expansion)
↓
RAG Retrieval (TF-IDF with similarity filtering)
↓
LLM Structured JSON Generation (Ollama - LLaMA3 8B)
↓
JSON Extraction + Repair
↓
Schema Validation
↓
Validation-Aware Retry
↓
Deterministic Schema Enforcement
↓
Evaluation Metrics
↓
Experiment Logging & Summary

```

---

## ⚙️ Features

### ✅ Structured Output Enforcement
- Strict JSON schema
- Required fields validation
- Nested attribute validation

### 🔁 Automatic Repair Mechanisms
- Malformed JSON self-correction
- Validation-aware structural repair
- Deterministic fallback enforcement

### 📊 Evaluation Metrics
- Structural consistency score
- Query-object alignment score
- JSON validity rate
- Failure logging taxonomy

### 🔍 Grounding Control
- Optional HyDE expansion
- TF-IDF retrieval augmentation
- Similarity threshold filtering
- RAG ablation support

### 🧪 Automatic Experiment Summarization
- Generates summary JSON
- Generates Markdown report
- Computes:
  - Validity rate
  - Average structural score
  - Average semantic alignment score

---

## 📂 Project Structure

```

vlm_dataset_generation/
│
├── pipeline/
│   ├── generator.py
│   ├── rag_module.py
│   ├── hyde.py
│   ├── validator.py
│   ├── evaluator.py
│
├── scripts/
│   ├── run_ablation.py
│   ├── summarize_ablation.py
│
├── data/
│   ├── raw_queries.json
│   ├── generated_outputs.json
│   └── logs/
│
├── experiments/
│   ├── ablation_results.json
│   ├── ablation_summary.json
│   └── ablation_summary.md
│
└── README.md

````

---

## 🚀 How to Run

### 1️⃣ Generate Structured Outputs

```bash
python -m pipeline.generator
````

Outputs saved to:

```
data/generated_outputs.json
```

---

### 2️⃣ Run Ablation Study

```bash
python scripts/run_ablation.py
```

---

### 3️⃣ Generate Automatic Summary

```bash
python scripts/summarize_ablation.py
```

Produces:

```
experiments/ablation_summary.json
experiments/ablation_summary.md
```

---

## 📈 Example Output

```json
{
  "scene_description": "A group of children playing football in a park.",
  "objects": [
    {
      "name": "football",
      "attributes": {
        "color": "brown",
        "position": "on the grass"
      }
    }
  ],
  "actions": ["kicking the ball"]
}
```

---

## 🔬 Ablation Study

We evaluate the effect of retrieval grounding:

| Setting         | Validity Rate | Avg Structural | Avg Alignment |
| --------------- | ------------- | -------------- | ------------- |
| No RAG          | 1.0           | 1.0            | 0.62          |
| Top-1 RAG       | 1.0           | 1.0            | 0.71          |
| Top-3 RAG       | 1.0           | 1.0            | 0.48          |
| Thresholded RAG | 1.0           | 1.0            | 0.69          |

Observations:

* Unfiltered multi-document retrieval increases semantic drift.
* Threshold filtering improves grounding discipline.
* Structural reliability remains stable due to repair loops.

---

## 🧠 What Makes This Different

Most LLM-based dataset projects:

* Do not validate outputs
* Do not repair malformed JSON
* Do not measure semantic drift
* Do not run controlled ablations

This project introduces:

* Generation control
* Structural enforcement
* Failure-aware retry
* Grounding discipline
* Reproducible evaluation

---

## 💻 System Configuration

* Model: LLaMA3 8B (via Ollama)
* GPU: RTX 3050 6GB
* RAM: 16GB
* Retrieval: TF-IDF (lightweight, CPU-friendly)

Designed to run efficiently on consumer hardware.

---

## 📌 Current Scope

This project focuses on:

* Structured scene annotation
* Controlled JSON generation
* Grounded retrieval experiments
* Evaluation methodology

It does NOT:

* Fine-tune models
* Use large-scale datasets
* Claim production deployment

---

## 🔮 Future Work

* Embedding-based retrieval (BGE-small)
* Hallucination detection metric
* Multimodal VLM integration
* Visualization dashboard
* Workshop paper submission

---

## 📜 License

MIT License

```
