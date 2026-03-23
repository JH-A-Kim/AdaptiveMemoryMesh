from memory.graph import get_context_string

def build_prompt(user_message: str) -> str:
    context = get_context_string()
    return f"""You are a memory-aware assistant. You have access to this knowledge graph from past conversations:

{context}

Strict rules:
- ONLY reference facts that are explicitly in the knowledge graph above
- Do NOT infer, guess, or connect entities unless the graph has an explicit edge between them
- If the graph has no relevant information for the query, just answer normally without referencing the graph
- Never fabricate relationships between graph nodes

User: {user_message}"""