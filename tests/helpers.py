import networkx as nx
import numpy as np
from src.tratamento import Grafo


def grafo_aleatorio(n=60, p=0.08, seed=1, pesos=True):
    G = nx.gnp_random_graph(n, p, seed=seed)
    rng = np.random.default_rng(seed)
    ar = [(u, v, float(rng.integers(1, 9)) if pesos else 1.0) for u, v in G.edges()]
    for u, v, w in ar:
        G[u][v]['weight'] = w
    return G, Grafo(n, ar)


def conexo(n=60, seed=1):
    for s in range(seed, seed + 50):
        G, g = grafo_aleatorio(n, 0.1, s)
        if nx.is_connected(G):
            return G, g
