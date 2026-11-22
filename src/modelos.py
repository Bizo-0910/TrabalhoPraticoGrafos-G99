"""Geradores de grafos aleatórios (implementados à mão) usados como modelos nulos e clássicos.

Todos devolvem uma lista de arestas (u, v) com u < v, sem repetição e sem auto-loops.
"""
import numpy as np


def erdos_renyi_gnm(n, m, rng):
    """G(n, m): m arestas escolhidas uniformemente entre os pares de vértices."""
    arestas = set()
    while len(arestas) < m:
        faltam = m - len(arestas)
        u = rng.integers(0, n, size=2 * faltam + 10)
        v = rng.integers(0, n, size=2 * faltam + 10)
        for a, b in zip(u, v):
            if a != b:
                arestas.add((min(a, b), max(a, b)))
                if len(arestas) == m:
                    break
    return sorted((int(a), int(b)) for a, b in arestas)


def barabasi_albert(n, k, rng):
    """Barabási-Albert: cada novo vértice se liga a k vértices distintos com probabilidade
    proporcional ao grau (lista de 'estacas' repetidas). Começa com k vértices isolados."""
    arestas = []
    alvos = list(range(k))
    estacas = []
    for novo in range(k, n):
        arestas.extend((min(novo, t), max(novo, t)) for t in alvos)
        estacas.extend(alvos)
        estacas.extend([novo] * k)
        escolhidos = set()
        while len(escolhidos) < k:
            escolhidos.add(estacas[int(rng.integers(0, len(estacas)))])
        alvos = list(escolhidos)
    return sorted(set(arestas))


def watts_strogatz(n, k, p, rng):
    """Watts-Strogatz: anel com k vizinhos (k par); cada aresta é religada com prob. p."""
    k = k - k % 2
    adj = [set() for _ in range(n)]
    for u in range(n):
        for j in range(1, k // 2 + 1):
            v = (u + j) % n
            adj[u].add(v); adj[v].add(u)
    for u in range(n):
        for j in range(1, k // 2 + 1):
            v = (u + j) % n
            if rng.random() < p and v in adj[u]:
                w = int(rng.integers(0, n))
                tentativas = 0
                while (w == u or w in adj[u]) and tentativas < 50:
                    w = int(rng.integers(0, n)); tentativas += 1
                if tentativas < 50:
                    adj[u].discard(v); adj[v].discard(u)
                    adj[u].add(w); adj[w].add(u)
    return sorted({(min(u, v), max(u, v)) for u in range(n) for v in adj[u]})


def reconexao_preserva_grau(arestas, rng, trocas_por_aresta=10):
    """Modelo nulo com a MESMA sequência de graus (Maslov-Sneppen): repete trocas de arestas
    (a,b),(c,d) -> (a,d),(c,b) evitando loops e duplicatas."""
    arestas = [tuple(e) for e in arestas]
    conj = set(arestas)
    m = len(arestas)
    feitas = tentativas = 0
    alvo = trocas_por_aresta * m
    while feitas < alvo and tentativas < 20 * alvo:
        tentativas += 1
        i, j = int(rng.integers(0, m)), int(rng.integers(0, m))
        if i == j:
            continue
        a, b = arestas[i]
        c, d = arestas[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        n1 = (min(a, d), max(a, d)); n2 = (min(c, b), max(c, b))
        if n1 in conj or n2 in conj:
            continue
        conj.discard(arestas[i]); conj.discard(arestas[j])
        conj.add(n1); conj.add(n2)
        arestas[i], arestas[j] = n1, n2
        feitas += 1
    return sorted(conj)


def lista_adj(n, arestas):
    adj = [[] for _ in range(n)]
    for u, v in arestas:
        adj[u].append(v); adj[v].append(u)
    return adj
