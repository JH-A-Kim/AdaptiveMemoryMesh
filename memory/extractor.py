import json
from llm import chat

EXTRACT_PROMPT = """
Extract entities and relationships from this message. 
Return ONLY valid JSON, nothing else. No explanation, no markdown, no backticks.

Format:
{{
  "entities": [
    {{"id": "unique_slug", "label": "Entity Name", "type": "person|project|decision|tool|fact"}}
  ],
  "relationships": [
    {{"from": "slug_a", "to": "slug_b", "relation": "short verb phrase"}}
  ]
}}

Message: {message}
"""

def extract(message: str) -> dict:
    response = chat(EXTRACT_PROMPT.format(message=message), max_tokens=500)
    
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except Exception:
        return {"entities": [], "relationships": []}
    