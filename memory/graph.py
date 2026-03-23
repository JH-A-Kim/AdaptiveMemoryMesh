import networkx as nx 

G = nx.DiGraph()

def add_from_extraction(extracted: dict):
    for entity in extracted.get("entities", []):
        G.add_node(entity["id"], **entity)
    
    for rel in extracted.get("relationships", []):
        if rel["from"] in G and rel["to"] in G:
            G.add_edge(rel["from"], rel["to"], relation=rel["relation"])

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
