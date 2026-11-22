import networkx as nx
import numpy as np
from src import estrutura as S
from tests.helpers import grafo_aleatorio, conexo


def test_triangulos_clusterizacao_transitividade():
    G, g = grafo_aleatorio(80, 0.1, 3)
    r = S.triangulos_e_clusterizacao(g.adj)
    assert r['triangulos'] == sum(nx.triangles(G).values()) // 3
    assert abs(r['clust_medio'] - nx.average_clustering(G)) < 1e-12
    assert abs(r['transitividade'] - nx.transitivity(G)) < 1e-12


def test_distancias_exatas():
    G, g = conexo()
    r = S.distancias_exatas(g.adj)
    assert r['diametro'] == nx.diameter(G)
    assert r['raio'] == nx.radius(G)
    assert abs(r['caminho_medio'] - nx.average_shortest_path_length(G)) < 1e-12


def test_kcore_e_assortatividade():
    G, g = grafo_aleatorio(100, 0.08, 5)
    ref = nx.core_number(G)
    assert S.kcore(g.adj) == [ref[i] for i in range(100)]
    assert abs(S.assortatividade_grau(g.adj) - nx.degree_assortativity_coefficient(G)) < 1e-9


def test_densidade_e_reciprocidade():
    assert abs(S.densidade(10, 9) - 0.2) < 1e-12
    D = nx.gnp_random_graph(40, 0.15, seed=2, directed=True)
    out = [list(D.successors(i)) for i in range(40)]
    assert abs(S.reciprocidade(out) - nx.reciprocity(D)) < 1e-12
