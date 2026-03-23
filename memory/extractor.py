import re
import json
from gliner import GLiNER
from llm import chat

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

LOW_CONFIDENCE_THRESHOLD = 0.75
FALLBACK_TYPE = "concept"

LABELS = [
    # people & teams
    "person", "team", "organization", "role",
    # software / tech
    "software tool", "programming language", "framework", "system", "api", "repository", "bug", "feature",
    # gaming / esports
    "game", "game mode", "champion", "character", "item", "ability", "strategy", "tournament",
    # decisions & action items
    "decision", "action item", "goal", "deadline", "blocker", "outcome",
    # general knowledge
    "location", "event", "concept", "metric", "date", "resource",
]

RELATION_PROMPT = """
You are extracting relationships between entities.

Entities found: {entities}
Message: "{message}"

Return ONLY a valid JSON array, no explanation, no markdown:
[{{"from": "slug_a", "to": "slug_b", "relation": "short verb phrase"}}]

Rules:
- "from" and "to" must be slugs from the entity list above
- "relation" should be a short active verb phrase (e.g. "built with", "assigned to", "blocks", "won using", "has", "located in", "scheduled for")
- Always link metrics and quantities to the entity they describe (e.g. event "has" attendee count)
- Always link dates and locations to the entity they belong to
- Only include relationships clearly implied by the message
- If no relationships exist, return []
"""

def normalize_id(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text

BLOCKLIST = {"i", "we", "you", "they", "he", "she", "it", "me", "us", "my", "our"}

import re

def extract_numbers(message: str) -> list:
    pattern = r'\b(\d+(?:,\d+)?(?:\.\d+)?)\s*(attendees|users|points|ms|seconds|percent|%|people|teams|players)?\b'
    matches = re.finditer(pattern, message, re.IGNORECASE)
    results = []
    for m in matches:
        number = m.group(1)
        unit = m.group(2) or "quantity"
        slug = normalize_id(f"{number}_{unit}")
        results.append({
            "id": slug,
            "label": f"{number} {unit}".strip(),
            "type": "metric",
            "score": 1.0
        })
    return results

def extract(message: str) -> dict:
    raw_entities = model.predict_entities(message, LABELS, threshold=0.45)
    seen = set()
    entities = []
    for e in raw_entities:
        slug = normalize_id(e["text"])
        if slug in seen or slug in BLOCKLIST:
            continue
        seen.add(slug)
        entities.append({
            "id": slug,
            "label": e["text"],
            "type": e["label"] if e["score"]>= LOW_CONFIDENCE_THRESHOLD else FALLBACK_TYPE,
            "score": round(e["score"], 3)
        })
    number_entities = extract_numbers(message)

    for e in number_entities:
        if e["id"] not in seen:
            seen.add(e["id"])
            entities.append(e)

    if len(entities) < 2:
        return {"entities": entities, "relationships": []}

    entity_ref = {e["label"]: e["id"] for e in entities}
    entity_list = ", ".join([f'"{e["label"]}" (slug: {e["id"]})' for e in entities])

    response = chat(RELATION_PROMPT.format(
        entities=entity_list,
        message=message
    ))

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        raw_rels = json.loads(response[start:end])
        valid_slugs = {e["id"] for e in entities}
        relationships = [
            r for r in raw_rels
            if r.get("from") in valid_slugs and r.get("to") in valid_slugs
        ]
    except Exception:
        relationships = []

    return {"entities": entities, "relationships": relationships}