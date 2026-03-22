from fastapi import FastAPI
from pydantic import BaseModel
from memory.extractor import extract
from memory.graph import add_from_extraction, get_all_nodes
from memory.retriever import build_prompt
from llm import chat

app = FastAPI()

class Message(BaseModel):
    content: str

@app.post("/chat")
def chat_endpoint(msg: Message):
    extracted = extract(msg.content)
    add_from_extraction(extracted)
    prompt = build_prompt(msg.content)
    reply = chat(prompt)
    return {
        "reply": reply,
        "memory_added": extracted
    }

@app.get("/memory")
def get_memory():
    return {"nodes": get_all_nodes()}