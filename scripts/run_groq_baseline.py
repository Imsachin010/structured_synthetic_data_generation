import json
import os
import time
import concurrent.futures
from groq import Groq

# Set up API client
os.environ["GROQ_API_KEY"] = "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"
client = Groq()

# Paths
DATA_DIR = "data"
RAW_QUERIES = os.path.join(DATA_DIR, "raw_queries.json")
OUT_FILE = "experiments/stage0_baseline.json"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.prompt_templates import BASE_PROMPT
from pipeline.validator import validate_output

def generate_groq(query):
    # Same prompt construction as Local Llama
    prompt = BASE_PROMPT.format(query=query)
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON. "
                        "Do not use markdown. "
                        "Do not invent names or brands. "
                        "Do not introduce entities not in the query or provided context. "
                        "Keep descriptions literal and minimal."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model="llama3-8b-8192",  # Equivalent model
            temperature=0.3,
            max_tokens=256,
        )
        output = response.choices[0].message.content
        return output
    except Exception as e:
        return str(e)

def process_single_query(query):
    raw_output = generate_groq(query)
    
    # Baseline exactly matches no enforcement
    info = {
        "failure_modes": [],
        "query": query,
        "raw_output": raw_output
    }
    
    # Try parsing
    try:
        parsed = json.loads(raw_output)
    except Exception as e:
        info["error"] = "json_parse_error"
        info["failure_modes"].append("malformed_json")
        return {"query": query, "output": {}, "info": info}
        
    # Validate structure silently (without repairing) to check failure modes
    is_valid, error_msg = validate_output(parsed)
    if not is_valid:
        info["error"] = "schema_validation_error"
        if "required property" in error_msg.lower():
            info["failure_modes"].append("missing_required_field")
        elif "is not of type" in error_msg.lower():
            info["failure_modes"].append("wrong_type")
        else:
            info["failure_modes"].append("schema_mismatch")
            
    return {"query": query, "output": parsed, "info": info}

def run_groq_baseline():
    print("Loading queries...")
    with open(RAW_QUERIES, "r") as f:
        queries = json.load(f)
        
    print(f"Loaded {len(queries)} queries. Running via Groq API concurrently...")
    
    results = []
    start_time = time.time()
    
    # Run all 100 queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single_query, q): q for q in queries}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"\rCompleted: {i+1}/{len(queries)}", end="")
            
    print(f"\nFinished in {time.time() - start_time:.2f} seconds.")
    
    # Save the exactly identical output format
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {OUT_FILE}")

if __name__ == "__main__":
    run_groq_baseline()
