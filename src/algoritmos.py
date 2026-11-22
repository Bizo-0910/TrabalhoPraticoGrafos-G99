"""Algoritmos da disciplina (Parte II), implementados à mão. Os testes comparam com o NetworkX."""
import collections
import heapq
import numpy as np

INF = float('inf')


def bfs(adj, s):
    """Busca em largura. O(V + E). Devolve a lista de distâncias (-1 = inalcançável)."""
    dist = [-1] * len(adj)
    dist[s] = 0
    fila = collections.deque([s])
    while fila:
        u = fila.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                fila.append(v)
    return dist


def dfs(adj, s):
    """Busca em profundidade iterativa. O(V + E). Devolve (ordem, descoberta, término)."""
    n = len(adj)
    desc = [-1] * n
    term = [-1] * n
    ordem = []
    t = 0
    desc[s] = t; t += 1
    ordem.append(s)
    pilha = [(s, iter(adj[s]))]
    while pilha:
        u, it = pilha[-1]
        for v in it:
            if desc[v] < 0:
                desc[v] = t; t += 1
                ordem.append(v)
                pilha.append((v, iter(adj[v])))
                break
        else:
            term[u] = t; t += 1
            pilha.pop()
    return ordem, desc, term


def eulerianidade_nao_dirigido(adj):
    """Euleriano (circuito) se conexo e todos os graus pares; semi-euleriano (caminho) se
    conexo e exatamente 2 graus ímpares. Considera só vértices com grau > 0. O(V + E)."""
    impares = sum(1 for a in adj if len(a) % 2 == 1)
    com_aresta = [i for i, a in enumerate(adj) if a]
    if not com_aresta:
        return 'euleriano (circuito)', 0, True
    dist = bfs(adj, com_aresta[0])
    conexo = all(dist[i] >= 0 for i in com_aresta)
    if conexo and impares == 0:
        tipo = 'euleriano (circuito)'
    elif conexo and impares == 2:
        tipo = 'semi-euleriano (caminho)'
    else:
        tipo = 'não euleriano'
    return tipo, impares, conexo


def eulerianidade_dirigido(out_adj):
    """Circuito euleriano dirigido: entrada = saída em todo vértice e conexo (fracamente).
    Devolve (tipo, vértices desbalanceados, conexo). O(V + E)."""
    n = len(out_adj)
    entrada = [0] * n
    und = [[] for _ in range(n)]
    for u in range(n):
        for v in out_adj[u]:
            entrada[v] += 1
            und[u].append(v); und[v].append(u)
    desbal = sum(1 for u in range(n) if len(out_adj[u]) != entrada[u])
    com_aresta = [i for i, a in enumerate(und) if a]
    dist = bfs(und, com_aresta[0])
    conexo = all(dist[i] >= 0 for i in com_aresta)
    return ('euleriano (circuito)' if conexo and desbal == 0 else 'não euleriano'), desbal, conexo


def dijkstra(adjw, s):
    """Fonte única, pesos >= 0, heap binário. O((V + E) log V)."""
    dist = [INF] * len(adjw)
    dist[s] = 0.0
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adjw[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def bellman_ford(n, arcos, s):
    """Fonte única, aceita pesos negativos. O(V·E) no pior caso (com parada antecipada).
    Devolve (dist, passadas, ciclo_negativo)."""
    dist = [INF] * n
    dist[s] = 0.0
    passadas = 0
    for _ in range(n - 1):
        mudou = False
        for u, v, w in arcos:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                mudou = True
        passadas += 1
        if not mudou:
            break
    ciclo_neg = any(dist[u] + w < dist[v] for u, v, w in arcos)
    return dist, passadas, ciclo_neg


def arcos_de(arestas):
    """Cada aresta não dirigida vira dois arcos."""
    return [(u, v, w) for u, v, w in arestas] + [(v, u, w) for u, v, w in arestas]


def matriz_pesos(n, arestas):
    W = np.full((n, n), INF)
    np.fill_diagonal(W, 0.0)
    for u, v, w in arestas:
        W[u, v] = W[v, u] = w
    return W


def floyd_warshall_py(W):
    """Floyd-Warshall em Python puro (lista de listas). O(V^3). Só para n pequeno."""
    n = len(W)
    D = [list(l) for l in W]
    for k in range(n):
        Dk = D[k]
        for i in range(n):
            dik = D[i][k]
            if dik == INF:
                continue
            Di = D[i]
            for j in range(n):
                nd = dik + Dk[j]
                if nd < Di[j]:
                    Di[j] = nd
    return D


def floyd_warshall_np(W):
    """Floyd-Warshall com os dois laços internos vetorizados em numpy. O(V^3) operações."""
    D = np.array(W, dtype=float)
    for k in range(D.shape[0]):
        np.minimum(D, D[:, k, None] + D[None, k, :], out=D)
    return D


def tarjan_scc(out_adj):
    """Componentes fortemente conexas (Tarjan), iterativo. O(V + E)."""
    n = len(out_adj)
    indice = [-1] * n
    low = [0] * n
    na_pilha = [False] * n
    pilha = []
    sccs = []
    c = 0
    for raiz in range(n):
        if indice[raiz] != -1:
            continue
        indice[raiz] = low[raiz] = c; c += 1
        pilha.append(raiz); na_pilha[raiz] = True
        trabalho = [(raiz, iter(out_adj[raiz]))]
        while trabalho:
            u, it = trabalho[-1]
            avancou = False
            for v in it:
                if indice[v] == -1:
                    indice[v] = low[v] = c; c += 1
                    pilha.append(v); na_pilha[v] = True
                    trabalho.append((v, iter(out_adj[v])))
                    avancou = True
                    break
                elif na_pilha[v]:
                    low[u] = min(low[u], indice[v])
            if avancou:
                continue
            if low[u] == indice[u]:
                comp = []
                while True:
                    w = pilha.pop(); na_pilha[w] = False
                    comp.append(w)
                    if w == u:
                        break
                sccs.append(comp)
            trabalho.pop()
            if trabalho:
                p = trabalho[-1][0]
                low[p] = min(low[p], low[u])
    return sccs


def pontes_e_articulacoes(adj):
    """Pontes e pontos de articulação (low-link de Tarjan) num grafo não dirigido simples. O(V + E)."""
    n = len(adj)
    desc = [-1] * n
    low = [0] * n
    pontes = []
    artic = set()
    t = 0
    for raiz in range(n):
        if desc[raiz] != -1:
            continue
        desc[raiz] = low[raiz] = t; t += 1
        filhos_raiz = 0
        pilha = [(raiz, -1, iter(adj[raiz]))]
        while pilha:
            u, pai, it = pilha[-1]
            avancou = False
            for v in it:
                if v == pai:
                    continue
                if desc[v] == -1:
                    desc[v] = low[v] = t; t += 1
                    pilha.append((v, u, iter(adj[v])))
                    avancou = True
                    break
                low[u] = min(low[u], desc[v])
            if avancou:
                continue
            pilha.pop()
            if pilha:
                p = pilha[-1][0]
                low[p] = min(low[p], low[u])
                if low[u] > desc[p]:
                    pontes.append((p, u))
                if p == raiz:
                    filhos_raiz += 1
                elif low[u] >= desc[p]:
                    artic.add(p)
        if filhos_raiz > 1:
            artic.add(raiz)
    return pontes, artic


def kruskal(n, arestas):
    """Árvore geradora mínima (floresta se desconexo): ordena + union-find. O(E log E)."""
    pai = list(range(n))
    rank = [0] * n

    def achar(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x
    peso, arv = 0.0, []
    for u, v, w in sorted(arestas, key=lambda e: e[2]):
        ru, rv = achar(u), achar(v)
        if ru != rv:
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            pai[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            peso += w
            arv.append((u, v, w))
            if len(arv) == n - 1:
                break
    return peso, arv


def prim(adjw, s=0):
    """Árvore geradora mínima da componente de s, com heap. O(E log V)."""
    visto = [False] * len(adjw)
    heap = [(0.0, s, -1)]
    peso, arv = 0.0, []
    while heap:
        w, u, p = heapq.heappop(heap)
        if visto[u]:
            continue
        visto[u] = True
        peso += w
        if p >= 0:
            arv.append((p, u, w))
        for v, wv in adjw[u]:
            if not visto[v]:
                heapq.heappush(heap, (wv, v, u))
    return peso, arv
