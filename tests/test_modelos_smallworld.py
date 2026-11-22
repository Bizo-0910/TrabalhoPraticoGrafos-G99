import collections
import numpy as np
from src import modelos as M, smallworld as SW
from tests.helpers import conexo


def _graus(n, ar):
    c = collections.Counter()
    for u, v in ar:
        c[u] += 1; c[v] += 1
    return [c[i] for i in range(n)]


def test_er_numero_de_arestas_e_simples():
    rng = np.random.default_rng(1)
    ar = M.erdos_renyi_gnm(200, 500, rng)
    assert len(ar) == 500 == len(set(ar)) and all(u < v for u, v in ar)


def test_ba_grau_minimo_e_hubs():
    rng = np.random.default_rng(2)
    ar = M.barabasi_albert(500, 3, rng)
    g = _graus(500, ar)
    assert min(g[3:]) >= 3 and max(g) > 20


def test_ws_grau_medio_e_p0_e_anel():
    rng = np.random.default_rng(3)
    ar = M.watts_strogatz(100, 6, 0.0, rng)
    assert set(_graus(100, ar)) == {6}
    ar2 = M.watts_strogatz(100, 6, 0.3, rng)
    assert len(ar2) == len(ar)


def test_reconexao_preserva_graus_e_arestas():
    rng = np.random.default_rng(4)
    ar = M.erdos_renyi_gnm(150, 600, rng)
    novo = M.reconexao_preserva_grau(ar, rng, 5)
    assert _graus(150, ar) == _graus(150, novo)
    assert len(novo) == len(ar) and set(novo) != set(ar)


def test_smallworld_er_nao_e_small_world_forte():
    rng = np.random.default_rng(5)
    ar = M.erdos_renyi_gnm(300, 1200, rng)
    adj = SW._maior_componente(M.lista_adj(300, ar))
    r = SW.avaliar(adj, rng, n_amostras=3, k_fontes=100)
    assert 0.3 < r['sigma_er'] < 3          # um grafo ER é seu próprio nulo: sigma ≈ 1
