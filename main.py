from fastapi import FastAPI
from pydantic import BaseModel
from memory.extractor import extract
from memory.graph import add_from_extraction, get_all_nodes, load_graph
from memory.retriever import build_prompt
from llm import chat
from memory.graph import G, GRAPH_PATH
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_graph()

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

@app.delete("/memory") 
def clear_memory():
    G.clear()
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)
    return {"status": "Memory cleared."}