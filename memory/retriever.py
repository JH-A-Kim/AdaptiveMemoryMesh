from memory.graph import get_context_string

def build_prompt(user_message: str) -> str:
    context = get_context_string()
    return f"""You have memory of past conversations stored as a knowledge graph: 
    
    {context}
    
    Use this context where relevant. Be concise.
    User: {user_message}
    """