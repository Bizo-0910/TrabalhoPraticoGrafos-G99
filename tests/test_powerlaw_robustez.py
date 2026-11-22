import numpy as np
from src import powerlaw as P, robustez as R, modelos as M
from tests.helpers import conexo


def test_mle_recupera_gamma():
    rng = np.random.default_rng(1)
    x = P.amostrar_pl(5000, 2.5, 3, rng)
    a = P.ajustar(x, kmin=3)
    assert abs(a['gamma'] - 2.5) < 0.1


def test_gof_nao_rejeita_lei_de_potencia_verdadeira():
    rng = np.random.default_rng(2)
    x = P.amostrar_pl(1500, 2.3, 2, rng)
    a = P.ajustar(x)
    p, _ = P.gof_bootstrap(x, a, 40, rng)
    assert p > 0.1


def test_comparacao_prefere_pl_a_exponencial_em_dados_pl():
    rng = np.random.default_rng(3)
    x = P.amostrar_pl(3000, 2.2, 2, rng)
    a = P.ajustar(x, kmin=2)
    assert P.comparar(x, a)['exponencial']['R'] > 0


def test_exponencial_rejeita_pl():
    rng = np.random.default_rng(4)
    x = np.floor(rng.exponential(8, 4000)).astype(int) + 1
    a = P.ajustar(x)
    assert P.comparar(x, a)['exponencial']['R'] < 0 or a['n_cauda'] < 200


def test_metricas_robustez_grafo_integro_e_remocao():
    G, g = conexo(60)
    m0 = R.metricas(g.adj, [])
    assert m0['A'] == 1 and m0['B'] == 1 and m0['D'] == 0
    # remover o centro de uma estrela isola todo mundo
    estrela = [[i for i in range(1, 6)]] + [[0] for _ in range(5)]
    m = R.metricas(estrela, [0])
    assert m['B'] == 5 and m['D'] == 1.0 and abs(m['A'] - 1 / 6) < 1e-12


def test_tamanho_remocao_e_aleatorio_reprodutivel():
    assert R.tamanho_remocao(1893, 0.05) == 95
    G, g = conexo(60)
    a = R.aleatorio(g.adj, 0.05, 3, np.random.default_rng(1))
    b = R.aleatorio(g.adj, 0.05, 3, np.random.default_rng(1))
    assert a == b
