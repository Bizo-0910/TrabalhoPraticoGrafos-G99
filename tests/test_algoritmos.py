import networkx as nx
import numpy as np
from src import algoritmos as A
from tests.helpers import grafo_aleatorio, conexo


def test_bfs_dfs():
    G, g = grafo_aleatorio(70, 0.07, 2)
    ref = nx.single_source_shortest_path_length(G, 0)
    d = A.bfs(g.adj, 0)
    assert all(d[v] == ref.get(v, -1) for v in range(70))
    ordem, desc, term = A.dfs(g.adj, 0)
    assert set(ordem) == set(nx.node_connected_component(G, 0))
    for u in ordem:                      # intervalos de descoberta/término aninhados ou disjuntos
        assert desc[u] < term[u]


def test_euler():
    ciclo = [[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]]
    assert A.eulerianidade_nao_dirigido(ciclo)[0] == 'euleriano (circuito)'
    caminho = [[1], [0, 2], [1]]
    assert A.eulerianidade_nao_dirigido(caminho)[0] == 'semi-euleriano (caminho)'
    estrela = [[1, 2, 3], [0], [0], [0]]
    assert A.eulerianidade_nao_dirigido(estrela)[0] == 'não euleriano'
    assert A.eulerianidade_dirigido([[1], [2], [0]])[0] == 'euleriano (circuito)'
    assert A.eulerianidade_dirigido([[1, 2], [], []])[0] == 'não euleriano'


def test_dijkstra_bellman_ford():
    G, g = conexo()
    ref = nx.single_source_dijkstra_path_length(G, 0)
    dj = A.dijkstra(g.adjw, 0)
    bf, _, neg = A.bellman_ford(g.n, A.arcos_de(g.arestas), 0)
    assert not neg
    for v in range(g.n):
        assert abs(dj[v] - ref[v]) < 1e-9 and abs(bf[v] - ref[v]) < 1e-9


def test_bellman_ford_ciclo_negativo():
    arcos = [(0, 1, 1.0), (1, 2, -3.0), (2, 0, 1.0)]
    assert A.bellman_ford(3, arcos, 0)[2] is True


def test_floyd_warshall():
    G, g = grafo_aleatorio(40, 0.15, 4)
    W = A.matriz_pesos(40, g.arestas)
    ref = dict(nx.all_pairs_dijkstra_path_length(G))
    for fw in (A.floyd_warshall_np(W), np.array(A.floyd_warshall_py(W.tolist()))):
        for u in range(40):
            for v in range(40):
                r = ref[u].get(v, np.inf)
                assert fw[u, v] == r or abs(fw[u, v] - r) < 1e-9


def test_tarjan_scc():
    D = nx.gnp_random_graph(120, 0.02, seed=7, directed=True)
    out = [list(D.successors(i)) for i in range(120)]
    mine = {frozenset(c) for c in A.tarjan_scc(out)}
    assert mine == {frozenset(c) for c in nx.strongly_connected_components(D)}


def test_pontes_articulacoes():
    G, g = grafo_aleatorio(90, 0.03, 9, pesos=False)
    pontes, artic = A.pontes_e_articulacoes(g.adj)
    assert {frozenset(p) for p in pontes} == {frozenset(p) for p in nx.bridges(G)}
    assert artic == set(nx.articulation_points(G))


def test_mst():
    G, g = conexo()
    ref = sum(d['weight'] for *_, d in nx.minimum_spanning_tree(G).edges(data=True))
    pk, ak = A.kruskal(g.n, g.arestas)
    pp, ap = A.prim(g.adjw, 0)
    assert abs(pk - ref) < 1e-9 and abs(pp - ref) < 1e-9
    assert len(ak) == len(ap) == g.n - 1
