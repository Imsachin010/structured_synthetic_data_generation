# pipeline/prompt_templates.py
BASE_PROMPT = """
You are asked to produce a JSON object that strictly conforms to the schema described below.
Schema:
- scene_description: string
- objects: array of objects, each with fields:
    - name: string
    - attributes: object with keys: color (string), position (string)
- actions: array of strings

Generate only valid JSON (no extra commentary). Example:
{{
  "scene_description": "A red car parked next to a tree.",
  "objects": [
    { "name": "car", "attributes": { "color": "red", "position": "left" } },
    { "name": "tree", "attributes": { "color": "green", "position": "right" } }
  ],
  "actions": ["parked"]
}}

User query:
{query}
"""

HYDE_PROMPT = """
Produce a short hypothetical document (1-2 lines) that elaborates on the query and may help the model produce the structured JSON.
Return plain text (not JSON).
Query:
{query}
"""
