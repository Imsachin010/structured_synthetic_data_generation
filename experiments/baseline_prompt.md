# Baseline Prompt

This baseline prompt was used before iteration. It often produced malformed JSON or extra commentary.

Prompt:

"Describe the scene in JSON. Include scene_description, objects (array), actions (array)."

Problems observed:
- LLM appended commentary after JSON
- Missing required fields ~23% of the time
- Incorrect attribute keys
