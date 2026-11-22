"""Propriedade small-world (Parte III, questão 1), implementada sem bibliotecas prontas.

Critério do enunciado: L(G) ≈ L_rand e C(G) >> C_rand, com o grafo aleatório de mesmo
número de vértices e arestas. Usamos também o índice de Humphries & Gurney:
    sigma = (C / C_rand) / (L / L_rand)      (sigma > 1 indica small-world)
Dois modelos nulos: Erdős-Rényi G(n, m) e rede com mesma sequência de graus (reconexão).
"""
import numpy as np
from . import estrutura as S
from . import modelos as M


def _medidas(adj, rng, k_fontes):
    c = S.triangulos_e_clusterizacao(adj)['clust_medio']
    n = len(adj)
    if k_fontes is None or k_fontes >= n:
        L = S.distancias_exatas(adj)['caminho_medio']
    else:
        L = S.caminho_medio_amostrado(adj, rng, k_fontes)
    return c, L


def _maior_componente(adj):
    from .tratamento import componentes
    comp = componentes(adj)[0]
    pos = {v: i for i, v in enumerate(comp)}
    novo = [[pos[v] for v in adj[u]] for u in comp]
    return novo


def avaliar(adj, rng, n_amostras=20, k_fontes=500):
    """adj: lista de adjacência da maior componente conexa. Devolve dicionário com os resultados."""
    n = len(adj)
    m = sum(len(a) for a in adj) // 2
    arestas = [(u, v) for u in range(n) for v in adj[u] if u < v]
    C, L = _medidas(adj, rng, None)                      # observado: exato
    k_med = 2 * m / n
    res = {'n': n, 'm': m, 'grau_medio': k_med, 'C': C, 'L': L,
           'C_er_teorico': k_med / n, 'L_er_teorico': float(np.log(n) / np.log(k_med))}
    for nome, gerar in [('er', lambda: M.erdos_renyi_gnm(n, m, rng)),
                        ('grau', lambda: M.reconexao_preserva_grau(arestas, rng))]:
        cs, ls = [], []
        for _ in range(n_amostras):
            adj_r = _maior_componente(M.lista_adj(n, gerar()))
            c, l = _medidas(adj_r, rng, k_fontes)
            cs.append(c); ls.append(l)
        res[f'C_{nome}'] = float(np.mean(cs)); res[f'C_{nome}_dp'] = float(np.std(cs, ddof=1))
        res[f'L_{nome}'] = float(np.mean(ls)); res[f'L_{nome}_dp'] = float(np.std(ls, ddof=1))
        res[f'razao_C_{nome}'] = C / res[f'C_{nome}']
        res[f'razao_L_{nome}'] = L / res[f'L_{nome}']
        res[f'sigma_{nome}'] = res[f'razao_C_{nome}'] / res[f'razao_L_{nome}']
    res['n_amostras'] = n_amostras
    res['k_fontes_nulos'] = k_fontes
    return res
