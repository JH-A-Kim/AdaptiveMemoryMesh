import networkx as nx 
import os 
from dotenv import load_dotenv
import pickle

load_dotenv()

GRAPH_PATH = os.getenv("GRAPH_PATH", "memory/graph.gpickle")

G = nx.DiGraph()

def load_graph():
    global G

    if os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH, "rb") as f:
            G=pickle.load(f)
        print(f"Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    else:
        print("No existing graph found. Starting with an empty graph.")
def save_graph():
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
        
def add_from_extraction(extracted: dict):
    for entity in extracted.get("entities", []):
        slug = entity["id"]
        if slug in G.nodes:
            existing_score = G.nodes[slug].get("score", 0)
            if entity.get("score", 0) > existing_score:
                G.nodes[slug].update(entity)
        else:
            G.add_node(slug, **entity)

    for rel in extracted.get("relationships", []):
        if rel["from"] in G and rel["to"] in G:
            G.add_edge(rel["from"], rel["to"], relation=rel["relation"])

    save_graph()

def get_all_nodes() -> list:
    return [dict(G.nodes[n]) for n in G.nodes]

def get_context_string() -> str:
    lines = []

    for u, v, data in G.edges(data=True):
        u_label = G.nodes[u].get("label", u)
        v_label = G.nodes[v].get("label", v)
        lines.append(f"{u_label} → {data['relation']} → {v_label}")

    connected = {n for edge in G.edges() for n in edge}
    for node_id in G.nodes:
        if node_id not in connected:
            node = G.nodes[node_id]
            lines.append(f"{node.get('label', node_id)} [type: {node.get('type', 'unknown')}]")

    return "\n".join(lines) if lines else "No memory yet."
