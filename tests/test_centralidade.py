import networkx as nx
import numpy as np
from src import centralidade as C
from tests.helpers import conexo


def test_betweenness():
    G, g = conexo()
    ref = nx.betweenness_centrality(G)
    mine = C.betweenness(g.adj)
    assert np.allclose(mine, [ref[i] for i in range(g.n)])


def test_pagerank():
    D = nx.gnp_random_graph(80, 0.05, seed=3, directed=True)
    out = [list(D.successors(i)) for i in range(80)]
    ref = nx.pagerank(D, alpha=0.85, tol=1e-12)
    assert np.allclose(C.pagerank(out), [ref[i] for i in range(80)], atol=1e-9)


def test_ranking_desempate():
    assert C.ranking([1, 3, 3, 2]) == [1, 2, 3, 0]
