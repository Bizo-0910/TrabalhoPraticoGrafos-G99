"""Robustez a falhas aleatórias e a ataques direcionados (Parte III, questão 3; Apêndice C).

Trabalhamos na maior componente conexa (política documentada). n = |V|, r = ceil(0.05 n).
Métricas após remover vértices (e suas arestas):
  A  tamanho da maior componente, em fração de n (S = |maior| / n); A_rel divide por n - r
  B  número de componentes (entre os vértices restantes)
  C  distância média na maior componente restante (L)
  D  fração de vértices isolados (grau 0) entre os restantes
"""
import math
import numpy as np
from . import estrutura as S
from .tratamento import componentes


def metricas(adj, removidos, rng=None, k_fontes=None):
    n = len(adj)
    fora = set(int(r) for r in removidos)
    vivos = [v for v in range(n) if v not in fora]
    pos = {v: i for i, v in enumerate(vivos)}
    red = [[pos[w] for w in adj[v] if w not in fora] for v in vivos]
    comps = componentes(red)
    maior = comps[0]
    isolados = sum(1 for a in red if not a)
    # distância média na maior componente
    p2 = {v: i for i, v in enumerate(maior)}
    sub = [[p2[w] for w in red[v]] for v in maior]
    if len(sub) < 2:
        L = 0.0
    elif k_fontes is None or k_fontes >= len(sub):
        L = S.distancias_exatas(sub)['caminho_medio']
    else:
        L = S.caminho_medio_amostrado(sub, rng, k_fontes)
    return {'A': len(maior) / n, 'A_rel': len(maior) / len(vivos), 'B': len(comps), 'C': float(L),
            'D': isolados / len(vivos)}


def tamanho_remocao(n, frac=0.05):
    return math.ceil(frac * n)


def aleatorio(adj, frac, T, rng, k_fontes=None):
    """T repetições removendo r = ceil(frac*n) vértices uniformemente ao acaso."""
    n = len(adj)
    r = tamanho_remocao(n, frac)
    return [metricas(adj, rng.choice(n, size=r, replace=False), rng, k_fontes) for _ in range(T)]


def ataque(adj, ordem, frac, rng=None, k_fontes=None):
    """Remove os r primeiros vértices do ranking fixo `ordem` (maior centralidade primeiro)."""
    r = tamanho_remocao(len(adj), frac)
    return metricas(adj, ordem[:r], rng, k_fontes)


def curva(adj, ordens, fracs, T, rng, k_fontes=300):
    """Curvas A, B, C, D em função da fração removida.
    ordens: dict nome -> ranking fixo; a estratégia 'aleatório' é sempre incluída (T repetições)."""
    n = len(adj)
    res = {'fracs': list(fracs)}
    res['aleatório'] = {k: [] for k in 'ABCD'}
    res['aleatório_dp'] = {k: [] for k in 'ABCD'}
    for f in fracs:
        reps = [metricas(adj, rng.choice(n, size=tamanho_remocao(n, f), replace=False), rng, k_fontes)
                for _ in range(T)] if f > 0 else [metricas(adj, [], rng, k_fontes)]
        for k in 'ABCD':
            vals = [x[k] for x in reps]
            res['aleatório'][k].append(float(np.mean(vals)))
            res['aleatório_dp'][k].append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    for nome, ordem in ordens.items():
        res[nome] = {k: [] for k in 'ABCD'}
        for f in fracs:
            x = metricas(adj, ordem[:tamanho_remocao(n, f)] if f > 0 else [], rng, k_fontes)
            for k in 'ABCD':
                res[nome][k].append(x[k])
    return res
