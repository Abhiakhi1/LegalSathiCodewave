import json
from langchain_groq import ChatGroq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def load_glove_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings[word] = vector
    return embeddings

def get_embedding(word):
    word = word.lower()
    if word in glove_embeddings:
        return glove_embeddings[word]
    else:
        print(f"⚠ '{word}' not found in GloVe vocabulary.")
        return np.zeros(300)  
    
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_relation(llm, keywords):
    relations = {}
    for i in range(len(keywords)):
        main = keywords[i]
        relations[main] = {}

        for j in range(len(keywords)):
            if i == j:
                continue

            other = keywords[j]
            prompt = (
                f"Give a one-line *short* direct relation between '{main}' and '{other}'. "
                f"Only give the relation if it exists. If no relation, reply exactly with 'None'."
            )

            response = llm.invoke(prompt).content.strip()

            if response.lower() != "none" and len(response.split()) <= 12 and "." in response:
                relations[main][other] = response

        if not relations[main]:
            del relations[main]

    with open(r"C:\Hackathon\LegalSathi\graph builder\keyword_relations.json", "w") as f:
        json.dump(relations, f, indent=2)

    print("✅ Clean relations saved to keyword_relations.json")
    return relations

def build_graph_with_weights(relations):
    G = nx.DiGraph()

    for main, subs in relations.items():
        main_emb = get_embedding(main)
        for sub, relation_text in subs.items():
            sub_emb = get_embedding(sub)

            weight = round(cosine_similarity(main_emb, sub_emb), 2)

            G.add_edge(main, sub, label=relation_text, weight=weight)

    return G

def vis_graph_with_legend(G, relations):
    pos = nx.spring_layout(G, seed=42)

    edge_labels = {}
    legend_text = []
    edge_number = 1

    for u, v, data in G.edges(data=True):
        relation = None
        if u in relations and v in relations[u]:
            relation = relations[u][v]
        elif v in relations and u in relations[v]:
            relation = relations[v][u]

        edge_labels[(u, v)] = str(edge_number)
        if relation:
            legend_text.append(f"{edge_number}. {relation} (weight: {data['weight']})")
        else:
            legend_text.append(f"{edge_number}. Weight: {data['weight']}")
        edge_number += 1

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    legend_str = "\n".join(legend_text)
    plt.gcf().text(0.75, 0.95, legend_str, fontsize=9, verticalalignment='top')

    plt.title("LLM Relations + GloVe Similarity Graph (Numbered Edges + Legend)")
    plt.show()

def main():
    global glove_embeddings
    keywords = ["technology", "education", "health", "economy", "agriculture", "energy"]

    glove_path = r"C:\Hackathon\LegalSathi\graph builder\glove.6B.300d.txt"
    glove_embeddings = load_glove_embeddings(glove_path)

    with open(r"C:\Hackathon\LegalSathi\graph builder\creds.json") as f:
        api_data = json.load(f)
        api_key = api_data["api_key"]

    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192")

    relations = get_relation(llm, keywords)

    G = build_graph_with_weights(relations)
    vis_graph_with_legend(G, relations)

    with open(r"C:\Hackathon\LegalSathi\graph builder\graph.pkl", "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()