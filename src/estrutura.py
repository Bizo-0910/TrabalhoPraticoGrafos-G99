"""Medidas estruturais da Parte I, implementadas à mão (NetworkX só valida nos testes)."""
import collections
import numpy as np


def bfs(adj, s):
    """Distâncias (em arestas) de s a todos; -1 se inalcançável. O(V + E)."""
    dist = [-1] * len(adj)
    dist[s] = 0
    fila = collections.deque([s])
    while fila:
        u = fila.popleft()
        du = dist[u] + 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = du
                fila.append(v)
    return dist


def resumo_graus(graus):
    v = np.asarray(graus)
    return {'min': int(v.min()), 'max': int(v.max()), 'media': float(v.mean()),
            'mediana': float(np.median(v)), 'desvio': float(v.std(ddof=0))}


def densidade(n, m, dirigido=False):
    return m / (n * (n - 1)) if dirigido else 2 * m / (n * (n - 1))


def reciprocidade(out_adj):
    """Fração de arcos u->v cujo arco v->u também existe."""
    conj = [set(a) for a in out_adj]
    total = sum(len(a) for a in out_adj)
    rec = sum(1 for u, a in enumerate(out_adj) for v in a if u in conj[v])
    return rec / total if total else 0.0


def distancias_exatas(adj, nos=None):
    """BFS a partir de cada vértice de `nos` (grafo conexo). Devolve diâmetro, raio,
    caminho médio, histograma de distâncias e excentricidades."""
    nos = range(len(adj)) if nos is None else nos
    hist = collections.Counter()
    ecc = {}
    for s in nos:
        dist = bfs(adj, s)
        ecc[s] = max(dist)
        hist.update(d for d in dist if d > 0)
    total = sum(hist.values())
    return {'diametro': max(ecc.values()), 'raio': min(ecc.values()),
            'caminho_medio': sum(d * c for d, c in hist.items()) / total,
            'hist': dict(sorted(hist.items())), 'ecc': ecc}


def caminho_medio_amostrado(adj, rng, k=500):
    """Estimativa do caminho médio com BFS a partir de k fontes aleatórias (grafo conexo)."""
    n = len(adj)
    fontes = rng.choice(n, size=min(k, n), replace=False)
    soma = cont = 0
    for s in fontes:
        dist = bfs(adj, int(s))
        soma += sum(dist)
        cont += n - 1
    return soma / cont


def triangulos_e_clusterizacao(adj):
    """Triângulos por vértice, total, clusterização local, média e transitividade."""
    n = len(adj)
    viz = [set(a) for a in adj]
    tri = [0] * n
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c = len(viz[u] & viz[v])
                tri[u] += c
                tri[v] += c
    # para o vértice u, cada triângulo (u,v,w) é contado nas arestas (u,v) e (u,w): duas vezes
    tri = [t // 2 for t in tri]
    total = sum(tri) // 3
    local = []
    for u in range(n):
        k = len(adj[u])
        local.append(2 * tri[u] / (k * (k - 1)) if k > 1 else 0.0)
    triplas = sum(len(a) * (len(a) - 1) // 2 for a in adj)
    return {'triangulos_por_vertice': tri, 'triangulos': total, 'clust_local': local,
            'clust_medio': float(np.mean(local)),
            'transitividade': 3 * total / triplas if triplas else 0.0}


def kcore(adj):
    """Número de core de cada vértice (algoritmo de remoção sucessiva, O(V + E))."""
    n = len(adj)
    grau = [len(a) for a in adj]
    maxg = max(grau) if n else 0
    bins = [[] for _ in range(maxg + 1)]
    for v, g in enumerate(grau):
        bins[g].append(v)
    core = [0] * n
    removido = [False] * n
    atual = 0
    for _ in range(n):
        while atual <= maxg and not bins[atual]:
            atual += 1
        # pode haver entradas obsoletas; descarta
        while True:
            v = bins[atual].pop()
            if not removido[v] and grau[v] == atual:
                break
            while atual <= maxg and not bins[atual]:
                atual += 1
        core[v] = atual
        removido[v] = True
        for u in adj[v]:
            if not removido[u] and grau[u] > atual:
                grau[u] -= 1
                bins[grau[u]].append(u)
        # o grau de u pode cair abaixo de `atual`: o core nunca é menor que o atual
        atual = max(0, atual - 1) if atual > 0 else 0
    return core


def assortatividade_grau(adj):
    """Correlação de Pearson entre os graus nas pontas das arestas."""
    x, y = [], []
    for u, a in enumerate(adj):
        for v in a:
            x.append(len(adj[u])); y.append(len(adj[v]))
    return float(np.corrcoef(x, y)[0, 1])


def ccdf(valores):
    """Distribuição acumulada complementar P(K >= k): devolve (k_distintos, P)."""
    v = np.sort(np.asarray(valores))
    ks, first = np.unique(v, return_index=True)
    return ks, 1 - first / len(v)
