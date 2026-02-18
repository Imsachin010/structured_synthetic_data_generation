import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from pipeline.generator import generate_scene
from pipeline.rag_module import SimpleRAG
from pipeline.embedding_module import EmbeddingRetriever

DATA_PATH = "data/raw_queries.json"
OUT_DIR = "experiments"

os.makedirs(OUT_DIR, exist_ok=True)


def load_queries():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def run_stage(stage_name, config):
    print(f"\nRunning {stage_name}...")

    queries = load_queries()
    rag = SimpleRAG.from_file(DATA_PATH)

    documents = [
        "A person walking a dog in a park.",
        "A red car parked near a tree.",
        "Children playing football in a field."
    ]
    embedding_retriever = EmbeddingRetriever(documents)

    results = []

    for item in queries:
        parsed, info = generate_scene(
            query=item["query"],
            image_path=config.get("image_path"),
            use_hyde=True,
            use_rag=config.get("use_rag", False),
            use_embedding_rag=config.get("use_embedding_rag", False),
            rag=rag,
            embedding_retriever=embedding_retriever,
            rag_top=3,
            rag_threshold=0.3,
        )

        results.append({
            "query": item["query"],
            "output": parsed,
            "info": info
        })

    with open(os.path.join(OUT_DIR, f"{stage_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"{stage_name} saved.")


if __name__ == "__main__":

    run_stage(
        "stage1_embedding_only",
        {
            "use_rag": False,
            "use_embedding_rag": True,
            "image_path": None
        }
    )

    run_stage(
        "stage2_embedding_plus_vlm",
        {
            "use_rag": False,
            "use_embedding_rag": True,
            "image_path": "./images/car.jpg"  # provide a valid image path for testing VLM
        }
    )

    run_stage(
        "stage3_tfidf_rag",
        {
            "use_rag": True,
            "use_embedding_rag": False,
            "image_path": None
        }
    )
