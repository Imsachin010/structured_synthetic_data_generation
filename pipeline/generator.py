import json
import os
import re
import ollama

from prompt_templates import BASE_PROMPT
from hyde import generate_hyde
from rag_module import SimpleRAG
from validator import validate_output
from evaluator import consistency_score, log_error, query_alignment_score

DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "generated_outputs.json")
RAW_QUERIES = os.path.join(DATA_DIR, "raw_queries.json")

os.makedirs(DATA_DIR, exist_ok=True)

MAX_RETRIES = 2


# ----------------------------
# JSON Extraction Utility
# ----------------------------
def extract_json_block(text):
    """
    Extract first JSON block from LLM output.
    Protects against extra commentary or markdown.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


# ----------------------------
# LLM Adapter (Ollama)
# ----------------------------
def llm_generate(prompt):
    try:
        response = ollama.chat(
            model="llama3:8b",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No explanation."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.3,
                "num_predict": 1024   # increased to avoid truncation
            }
        )

        return response["message"]["content"]

    except Exception as e:
        log_error("ollama_runtime_error", str(e))
        raise


# ----------------------------
# Main Generation Function
# ----------------------------
def generate_scene(query, use_hyde=True, use_rag=True, rag=None, llm_fn=None, rag_top=3, rag_threshold=0.1):
    llm_fn = llm_fn or llm_generate
    rag = rag or SimpleRAG.from_file(RAW_QUERIES)

    hyde_doc = generate_hyde(query, llm_fn) if use_hyde else None
    if use_rag:
        raw_retrieved = rag.retrieve(query, top_k=rag_top)
        retrieved = [r for r in raw_retrieved if r["score"] >= rag_threshold]
    else:
        retrieved = []


    context_parts = []
    if hyde_doc:
        context_parts.append(hyde_doc)
    if retrieved:
        context_parts += [r["doc"] for r in retrieved]

    context_text = "\n".join(context_parts)

    prompt = BASE_PROMPT.format(
        query=query + ("\nContext:\n" + context_text if context_text else "")
    )

    # ----------------------------
    # Retry Loop with Self-Repair
    # ----------------------------
    for attempt in range(MAX_RETRIES + 1):

        raw_response = llm_fn(prompt)
        raw_response = extract_json_block(raw_response)

        try:
            parsed = json.loads(raw_response)
            break  # success

        except Exception as e:
            if attempt == MAX_RETRIES:
                log_error(
                    "json_parse_error",
                    str(e),
                    output=raw_response,
                    metadata={"query": query}
                )
                return None, {"error": "json_parse_error", "details": str(e)}

            # Ask model to fix malformed JSON
            prompt = f"""
The following JSON is malformed.
Fix it and return ONLY valid JSON.

Malformed JSON:
{raw_response}
"""

    # ----------------------------
    # Schema Validation
    # ----------------------------
    is_valid, errors = validate_output(parsed)
    if not is_valid:
        log_error(
            "validation_error",
            "; ".join(errors),
            output=parsed,
            metadata={"query": query}
        )
        return parsed, {"error": "validation_error", "details": errors}

    # ----------------------------
    # Evaluation
    # ----------------------------
    score, issues = consistency_score(parsed)
    alignment = query_alignment_score(query, parsed)
    return parsed, {
    "score": score,
    "alignment_score": alignment,
    "issues": issues
    }



# ----------------------------
# Demo Runner
# ----------------------------
def run_demo():
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
        parsed, info = generate_scene(
            item['query'],
            use_hyde=True,
            use_rag=True,
            rag=rag, rag_top=3, rag_threshold=0.2
        )
        outputs.append({
            "query": item['query'],
            "output": parsed,
            "info": info
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"Demo done — outputs written to {OUT_FILE}")


if __name__ == "__main__":
    run_demo()
