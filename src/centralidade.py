"""Centralidades usadas no ataque direcionado (Parte III): grau, intermediação e PageRank."""
import collections
import numpy as np


def betweenness(adj):
    """Intermediação exata (algoritmo de Brandes) em grafo não dirigido e não ponderado.
    O(V·E). Valores normalizados como no NetworkX: divididos por (n-1)(n-2)."""
    n = len(adj)
    cb = np.zeros(n)
    for s in range(n):
        pilha = []
        pred = [[] for _ in range(n)]
        sigma = [0] * n
        sigma[s] = 1
        dist = [-1] * n
        dist[s] = 0
        fila = collections.deque([s])
        while fila:
            v = fila.popleft()
            pilha.append(v)
            dv = dist[v] + 1
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dv
                    fila.append(w)
                if dist[w] == dv:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = [0.0] * n
        while pilha:
            w = pilha.pop()
            coef = (1.0 + delta[w]) / sigma[w]
            for v in pred[w]:
                delta[v] += sigma[v] * coef
            if w != s:
                cb[w] += delta[w]
    cb /= 2.0                                   # cada par foi contado nos dois sentidos
    if n > 2:
        cb /= (n - 1) * (n - 2) / 2.0
    return cb


def pagerank(out_adj, alfa=0.85, tol=1e-12, max_iter=200):
    """PageRank por iteração de potência no grafo dirigido (nós sem saída redistribuem uniformemente)."""
    n = len(out_adj)
    src = np.array([u for u, a in enumerate(out_adj) for _ in a])
    dst = np.array([v for a in out_adj for v in a])
    grau_out = np.array([len(a) for a in out_adj], dtype=float)
    pr = np.full(n, 1.0 / n)
    sem_saida = grau_out == 0
    for _ in range(max_iter):
        contrib = np.zeros(n)
        np.add.at(contrib, dst, pr[src] / grau_out[src])
        novo = alfa * (contrib + pr[sem_saida].sum() / n) + (1 - alfa) / n
        if np.abs(novo - pr).sum() < tol:
            pr = novo
            break
        pr = novo
    return pr


def ranking(valores):
    """Índices ordenados do maior para o menor valor (desempate pelo menor índice, reprodutível)."""
    v = np.asarray(valores)
    return list(np.lexsort((np.arange(len(v)), -v)))
