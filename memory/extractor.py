import re
import json
from gliner import GLiNER
from llm import chat
import re

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

LOW_CONFIDENCE_THRESHOLD = 0.75
FALLBACK_TYPE = "concept"

LABELS = [
    # people & identity
    "person", "team", "organization", "company", "government", "institution",
    "role", "job title", "department", "community", "ethnic group", "nationality",

    # places & geography
    "location", "city", "country", "region", "venue", "address", "landmark",
    "building", "online platform", "website", "server", "environment",

    # time
    "date", "time", "deadline", "duration", "period", "season", "era",

    # events & activities
    "event", "meeting", "conference", "hackathon", "competition", "tournament",
    "project", "initiative", "campaign", "mission", "task", "action item",

    # decisions & outcomes
    "decision", "goal", "objective", "outcome", "result", "achievement",
    "blocker", "risk", "issue", "problem", "solution", "requirement",

    # software & tech
    "software tool", "programming language", "framework", "library", "api",
    "repository", "database", "system", "architecture", "algorithm", "model",
    "bug", "feature", "deployment", "infrastructure", "cloud service",
    "protocol", "data format", "operating system", "hardware", "device",

    # gaming & esports
    "game", "game mode", "champion", "character", "item", "ability",
    "strategy", "mechanic", "patch", "meta", "rank", "tournament",
    "team composition", "skill", "quest", "level", "achievement",

    # finance & business
    "financial concept", "accounting method", "business process", "revenue",
    "expense", "asset", "liability", "investment", "market", "industry",
    "product", "service", "contract", "metric", "kpi", "budget",
    "transaction", "currency", "tax", "regulation", "policy",

    # science & academia
    "scientific concept", "theory", "hypothesis", "experiment", "study",
    "field of study", "research area", "institution", "publication",
    "chemical", "element", "organism", "disease", "treatment", "drug",
    "phenomenon", "law of nature", "mathematical concept", "formula",

    # arts & culture
    "music genre", "artist", "musician", "song", "album", "instrument",
    "art movement", "artwork", "film", "tv show", "book", "genre",
    "cultural practice", "tradition", "language", "religion", "philosophy",
    "sport", "athlete", "team sport", "recreational activity",

    # media & communication
    "news event", "media outlet", "social media platform", "content",
    "brand", "advertisement", "message", "topic", "theme", "narrative",

    # preferences & opinions
    "preference", "opinion", "favourite", "belief", "value",
    "hobby", "interest", "skill", "expertise",

    # general knowledge
    "concept", "idea", "fact", "statistic", "quantity", "measurement",
    "relationship", "process", "method", "principle", "rule", "standard",
    "resource", "material", "object", "tool", "technology",
]

RELATION_PROMPT = """
You are extracting relationships between entities.

Entities found: {entities}
Message: "{message}"

Return ONLY a valid JSON array, no explanation, no markdown:
[{{"from": "slug_a", "to": "slug_b", "relation": "short verb phrase"}}]

Rules:
- "from" and "to" must be slugs from the entity list above
- "relation" should be a short active verb phrase
- For preference and opinion language, ALWAYS link User directly to the named entity:
  "X is my favourite Y" → {{"from": "speaker", "to": "x_slug", "relation": "considers favourite"}}
  "X is the best Y to me" → {{"from": "speaker", "to": "x_slug", "relation": "considers best"}}
  "I love X" → {{"from": "speaker", "to": "x_slug", "relation": "loves"}}
  NEVER link User to a generic concept like "musician" or "game" — link to the specific named entity
- Always link metrics to the entity they describe
- Always link dates and locations to their parent entity
- If no relationships exist, return []
"""

def normalize_id(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text

BLOCKLIST = {
    "i", "we", "you", "they", "he", "she", "it", "me", "us", "my", "our",
    "game", "thing", "stuff", "something", "anything", "everything",
    "speaker", "musician", "artist", "person", "player"
}

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
            "type": e["label"] if e["score"] >= LOW_CONFIDENCE_THRESHOLD else FALLBACK_TYPE,
            "score": round(e["score"], 3)
        })

    number_entities = extract_numbers(message)
    for e in number_entities:
        if e["id"] not in seen:
            seen.add(e["id"])
            entities.append(e)

    if len(entities) < 1:
        return {"entities": entities, "relationships": []}

    speaker_entity = {"id": "speaker", "label": "User", "type": "person", "score": 1.0}
    all_entities_for_relations = [speaker_entity] + entities

    entity_list = ", ".join([
        f'"{e["label"]}" (slug: {e["id"]})' 
        for e in all_entities_for_relations
    ])

    response = chat(RELATION_PROMPT.format(
        entities=entity_list,
        message=message
    ))

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        raw_rels = json.loads(response[start:end])
        valid_slugs = {e["id"] for e in all_entities_for_relations}
        relationships = [
            r for r in raw_rels
            if r.get("from") in valid_slugs and r.get("to") in valid_slugs
        ]
    except Exception:
        relationships = []

    return {"entities": entities, "relationships": relationships}