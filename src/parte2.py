"""Parte II: algoritmos da disciplina, com complexidade teórica e observada (tempo real)."""
import collections
import os
import numpy as np
import pandas as pd
from . import algoritmos as A
from . import benchmark as B
from . import figuras as F
from .util import salvar_json

REPS = 30        # repetições (n >= 30 -> IC pela Normal)
REPS_FW = 5      # Floyd-Warshall no grafo completo (n < 30 -> IC pela t-Student)

TEORIA = {'BFS': 'V + E', 'DFS': 'V + E', 'Dijkstra': '(V+E) log V', 'Bellman-Ford': 'V·E',
          'Kruskal': 'E log E', 'Prim': 'E log V', 'Tarjan (pontes)': 'V + E', 'Floyd-Warshall': 'V³'}


def f_teo(nome, V, E):
    return {'BFS': V + E, 'DFS': V + E, 'Dijkstra': (V + E) * np.log2(V), 'Bellman-Ford': V * E,
            'Kruskal': E * np.log2(E), 'Prim': E * np.log2(V), 'Tarjan (pontes)': V + E,
            'Floyd-Warshall': V ** 3}[nome]


def _ordem_bfs(adj):
    raiz = max(range(len(adj)), key=lambda v: len(adj[v]))
    vis = {raiz}
    ordem = [raiz]
    i = 0
    while i < len(ordem):
        for v in adj[ordem[i]]:
            if v not in vis:
                vis.add(v); ordem.append(v)
        i += 1
    return ordem


def executar(d, res_dir, fig_dir, rng, reps=REPS, reps_fw=REPS_FW, verbose=True):
    gc = d.gc
    n, m = gc.n, gc.m
    arcos = A.arcos_de(gc.arestas)
    r = {'n': n, 'm': m, 'reps': reps, 'reps_fw': reps_fw}

    # ---------- resultados dos algoritmos ----------
    s0 = int(rng.integers(0, n))
    dist = A.bfs(gc.adj, s0)
    ordem, desc, term = A.dfs(gc.adj, s0)
    r['bfs'] = {'fonte': d.ids[d.gc_nos[s0]], 'alcancados': sum(x >= 0 for x in dist), 'excentricidade': max(dist)}
    r['dfs'] = {'visitados': len(ordem)}
    tipo, impares, conexo = A.eulerianidade_nao_dirigido(gc.adj)
    r['euler_nd'] = {'tipo': tipo, 'graus_impares': impares, 'conexo': conexo}
    tipo_d, desbal, conexo_d = A.eulerianidade_dirigido(d.out_adj)
    r['euler_dir'] = {'tipo': tipo_d, 'desbalanceados': desbal, 'conexo': conexo_d}
    dj = A.dijkstra(gc.adjw, s0)
    bf, passadas, neg = A.bellman_ford(n, arcos, s0)
    assert max(abs(a - b) for a, b in zip(dj, bf)) < 1e-9
    r['dijkstra'] = {'dist_max': max(dj), 'dist_media': float(np.mean(dj))}
    r['bellman_ford'] = {'passadas': passadas, 'ciclo_negativo': neg}
    sccs = A.tarjan_scc(d.out_adj)
    tam = sorted((len(c) for c in sccs), reverse=True)
    r['tarjan_scc'] = {'n_scc': len(sccs), 'maior': tam[0], 'cinco_maiores': tam[:5], 'tamanho_1': sum(t == 1 for t in tam)}
    pontes, artic = A.pontes_e_articulacoes(gc.adj)
    r['pontes'] = {'n_pontes': len(pontes), 'n_articulacoes': len(artic)}
    pk, ak = A.kruskal(n, gc.arestas)
    pp, ap = A.prim(gc.adjw, 0)
    assert abs(pk - pp) < 1e-9 and len(ak) == n - 1
    r['mst'] = {'peso': pk, 'arestas': len(ak)}

    # ---------- tempo real no grafo completo ----------
    srcs = [int(s) for s in rng.choice(n, size=reps, replace=False)]
    W = A.matriz_pesos(n, gc.arestas)
    sub150, _ = gc.induzido(_ordem_bfs(gc.adj)[:150])
    W150 = A.matriz_pesos(150, sub150.arestas)
    W150l = W150.tolist()
    exper = [
        ('BFS', 'O(V + E)', B.medir(A.bfs, [(gc.adj, s) for s in srcs])),
        ('DFS', 'O(V + E)', B.medir(A.dfs, [(gc.adj, s) for s in srcs])),
        ('Eulerianidade', 'O(V + E)', B.medir(A.eulerianidade_nao_dirigido, [(gc.adj,)] * reps)),
        ('Dijkstra', 'O((V + E) log V)', B.medir(A.dijkstra, [(gc.adjw, s) for s in srcs])),
        ('Bellman-Ford', 'O(V·E)', B.medir(A.bellman_ford, [(n, arcos, s) for s in srcs])),
        ('Tarjan (SCC)', 'O(V + E)', B.medir(A.tarjan_scc, [(d.out_adj,)] * reps)),
        ('Tarjan (pontes)', 'O(V + E)', B.medir(A.pontes_e_articulacoes, [(gc.adj,)] * reps)),
        ('Kruskal', 'O(E log E)', B.medir(A.kruskal, [(n, gc.arestas)] * reps)),
        ('Prim', 'O(E log V)', B.medir(A.prim, [(gc.adjw, s) for s in srcs])),
        ('Floyd-Warshall', 'O(V³)', B.medir(A.floyd_warshall_np, [(W,)] * reps_fw, aquecimento=0)),
        ('Floyd-Warshall (n=150, python)', 'O(V³)', B.medir(A.floyd_warshall_py, [(W150l,)] * 10)),
        ('Floyd-Warshall (n=150, numpy)', 'O(V³)', B.medir(A.floyd_warshall_np, [(W150,)] * 10)),
    ]
    linhas = []
    for nome, teo, ts in exper:
        e = B.ic(ts)
        linhas.append({'algoritmo': nome, 'complexidade': teo, 'repeticoes': e['n'], 'media': e['media'],
                       'dp': e['dp'], 'ic_lo': e['lo'], 'ic_hi': e['hi'], 'distribuicao_ic': e['dist']})
    tab = pd.DataFrame(linhas)
    F.tempos(tab[~tab['algoritmo'].str.contains('n=150')], f'Tempo real dos algoritmos (V = {n}, E = {m})', fig_dir)

    # ---------- complexidade observada: subgrafos crescentes ----------
    ordem_b = _ordem_bfs(gc.adj)
    tamanhos = [200, 400, 800, 1200, 1600, n]
    tamanhos_fw = [100, 200, 300, 400, 600, 800]
    obs = collections.defaultdict(list)
    for k in sorted(set(tamanhos + tamanhos_fw)):
        sg, _ = gc.induzido(ordem_b[:k])
        arc = A.arcos_de(sg.arestas)
        reps_k = 10 if k > 800 else 30
        s_ = [int(x) for x in rng.choice(k, size=min(reps_k, k), replace=False)]
        runs = {
            'BFS': B.medir(A.bfs, [(sg.adj, s) for s in s_]),
            'DFS': B.medir(A.dfs, [(sg.adj, s) for s in s_]),
            'Dijkstra': B.medir(A.dijkstra, [(sg.adjw, s) for s in s_]),
            'Bellman-Ford': B.medir(A.bellman_ford, [(k, arc, s) for s in s_[:10]]),
            'Kruskal': B.medir(A.kruskal, [(k, sg.arestas)] * 10),
            'Prim': B.medir(A.prim, [(sg.adjw, s) for s in s_[:10]]),
            'Tarjan (pontes)': B.medir(A.pontes_e_articulacoes, [(sg.adj,)] * 10),
        }
        if k in tamanhos_fw:
            Wk = A.matriz_pesos(k, sg.arestas)
            runs['Floyd-Warshall'] = B.medir(A.floyd_warshall_np, [(Wk,)] * (10 if k <= 400 else 5))
        for nome, ts in runs.items():
            e = B.ic(ts)
            obs[nome].append((k, sg.m, e['media'], e['lo'], e['hi']))
        if verbose:
            print('  subgrafo', k, 'ok', flush=True)
    incl = {}
    for nome, dd in obs.items():
        dd = np.array(dd)
        x = np.array([f_teo(nome, V, E) for V, E, *_ in dd])
        incl[nome] = B.inclinacao_loglog(x, dd[:, 2])
    F.complexidade(dict(obs), TEORIA, f_teo, incl, fig_dir)
    r['inclinacoes'] = {k: v[0] for k, v in incl.items()}
    tab_incl = pd.DataFrame({'algoritmo': list(incl), 'complexidade_teorica': [TEORIA[k] for k in incl],
                             'inclinacao_loglog': [v[0] for v in incl.values()]})

    resumo = pd.DataFrame([
        ('BFS', 'Sim', f"maior componente; alcança {r['bfs']['alcancados']} vértices"),
        ('DFS', 'Sim', f"maior componente; visita {r['dfs']['visitados']} vértices"),
        ('Eulerianidade', 'Sim', f"{tipo}; {impares} vértices de grau ímpar (não dirigido); dirigido: {desbal} com entrada ≠ saída"),
        ('Dijkstra', 'Sim (pesos 1/m > 0)', f"distância máxima {r['dijkstra']['dist_max']:.3f} a partir da fonte"),
        ('Bellman-Ford', 'Sim (pesos 1/m)', f"sem ciclo negativo; {passadas} passadas até convergir (pior caso {n - 1})"),
        ('Floyd-Warshall', 'Sim', f"grafo completo ({n} vértices), versão numpy; python puro só com n = 150"),
        ('Tarjan (SCC)', 'Sim', f"{len(sccs)} componentes fortemente conexas; maior com {tam[0]}"),
        ('Tarjan (pontes/articulações)', 'Sim', f"{len(pontes)} pontes e {len(artic)} pontos de articulação"),
        ('Kruskal / Prim', 'Sim', f"árvore geradora mínima, peso {pk:.3f}"),
    ], columns=['Algoritmo', 'Aplicado?', 'Resultado'])

    os.makedirs(res_dir, exist_ok=True)
    tab.to_csv(os.path.join(res_dir, 'parte2_tempos.csv'), index=False)
    tab_incl.to_csv(os.path.join(res_dir, 'parte2_inclinacoes.csv'), index=False)
    resumo.to_csv(os.path.join(res_dir, 'parte2_resumo.csv'), index=False)
    r['obs'] = {k: [list(map(float, x)) for x in v] for k, v in obs.items()}
    salvar_json(r, os.path.join(res_dir, 'parte2.json'))
    return r, tab, tab_incl, resumo
