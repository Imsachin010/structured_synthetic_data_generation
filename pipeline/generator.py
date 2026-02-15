# pipeline/generator.py
import json
import os
from prompt_templates import BASE_PROMPT
from hyde import generate_hyde
from rag_module import SimpleRAG
from validator import validate_output
from evaluator import consistency_score, log_error

DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "generated_outputs.json")
RAW_QUERIES = os.path.join(DATA_DIR, "raw_queries.json")

os.makedirs(DATA_DIR, exist_ok=True)

import ollama

def llm_generate(prompt):
    try:
        response = ollama.chat(
            model="llama3:8b",
            messages=[
                {"role": "system", "content": "You must output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.3,
                "num_predict": 512
            }
        )

        return response["message"]["content"]

    except Exception as e:
        print("Ollama error:", e)
        raise


def generate_scene(query, use_hyde=True, use_rag=True, rag=None, llm_fn=None):
    llm_fn = llm_fn or llm_generate
    rag = rag or SimpleRAG.from_file(RAW_QUERIES)

    hyde_doc = generate_hyde(query, llm_fn) if use_hyde else None
    retrieved = rag.retrieve(query, top_k=3) if use_rag else []

    # combine context
    context_parts = []
    if hyde_doc:
        context_parts.append(hyde_doc)
    if retrieved:
        context_parts += [r["doc"] for r in retrieved]
    context_text = "\n".join(context_parts)
    prompt = BASE_PROMPT.format(query=query + ("\nContext:\n" + context_text if context_text else ""))

    raw_response = llm_fn(prompt)

    # Try parse JSON
    try:
        parsed = json.loads(raw_response)
    except Exception as e:
        log_error("json_parse_error", str(e), output=raw_response, metadata={"query": query})
        return None, {"error": "json_parse_error", "details": str(e)}

    # Validate
    is_valid, errors = validate_output(parsed)
    if not is_valid:
        log_error("validation_error", "; ".join(errors), output=parsed, metadata={"query": query})
        return parsed, {"error": "validation_error", "details": errors}

    # Evaluate
    score, issues = consistency_score(parsed)
    return parsed, {"score": score, "issues": issues}

def run_demo():
    # write a tiny sample raw_queries.json if not present
    if not os.path.exists(RAW_QUERIES):
        sample = [
            {"id": 1, "query": "A person walking a dog in the park"},
            {"id": 2, "query": "A red car parked near a tree"},
            {"id": 3, "query": "A group of kids playing football"}
        ]
        with open(RAW_QUERIES, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)

    rag = SimpleRAG.from_file(RAW_QUERIES)
    outputs = []
    with open(RAW_QUERIES, "r", encoding="utf-8") as f:
        queries = json.load(f)

    for item in queries:
        print(f"Generating for: {item['query']}")
        parsed, info = generate_scene(item['query'], use_hyde=True, use_rag=True, rag=rag)
        outputs.append({"query": item['query'], "output": parsed, "info": info})

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"Demo done — outputs written to {OUT_FILE}")

if __name__ == "__main__":
    run_demo()
