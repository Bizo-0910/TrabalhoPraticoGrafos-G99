"""Leitura e tratamento do dataset CollegeMsg (grafo G99).

Tratamentos aplicados (todos informados no relatório):
  1. grafo temporal -> estático (união de todas as arestas ao longo dos 193 dias);
  2. remoção de auto-loops (o arquivo original não possui nenhum);
  3. multigrafo -> grafo simples, guardando quantas mensagens cada par trocou;
  4. versão não dirigida: existe aresta {u,v} se u->v ou v->u;
  5. peso da aresta não dirigida w = 1/m, com m = total de mensagens do par
     (laço forte = distância curta; permite Dijkstra, Bellman-Ford, MST);
  6. a maior componente conexa é usada nas métricas de distância e na robustez.
"""
import collections
import csv
import os


class Grafo:
    """Grafo simples não dirigido com ids 0..n-1, lista de adjacência e pesos."""

    def __init__(self, n, arestas):
        """arestas: lista de (u, v, w) com u != v, sem repetição."""
        self.n = n
        self.arestas = list(arestas)
        self.adj = [[] for _ in range(n)]
        self.adjw = [[] for _ in range(n)]
        for u, v, w in self.arestas:
            self.adj[u].append(v); self.adj[v].append(u)
            self.adjw[u].append((v, w)); self.adjw[v].append((u, w))

    @property
    def m(self):
        return len(self.arestas)

    def induzido(self, nos):
        """Subgrafo induzido pelos vértices `nos`, reindexado 0..k-1. Devolve (Grafo, lista_original)."""
        nos = list(nos)
        pos = {v: i for i, v in enumerate(nos)}
        ar = [(pos[u], pos[v], w) for u, v, w in self.arestas if u in pos and v in pos]
        return Grafo(len(nos), ar), nos


class Dados:
    """Resultado do tratamento: grafo dirigido e não dirigido, mensagens e tempos."""
    pass


def ler_mensagens(caminho):
    """Lê o arquivo SRC DST UNIXTS. Devolve lista de (u, v, t)."""
    msgs = []
    with open(caminho) as f:
        for linha in f:
            p = linha.split()
            if len(p) >= 3:
                msgs.append((int(p[0]), int(p[1]), int(p[2])))
    return msgs


def componentes(adj):
    """Componentes conexas por BFS. Devolve lista de listas, da maior para a menor."""
    n = len(adj)
    visto = [False] * n
    comps = []
    for s in range(n):
        if visto[s]:
            continue
        visto[s] = True
        comp = [s]
        i = 0
        while i < len(comp):
            u = comp[i]; i += 1
            for v in adj[u]:
                if not visto[v]:
                    visto[v] = True
                    comp.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def construir(msgs):
    """Aplica o tratamento e devolve um objeto Dados."""
    d = Dados()
    d.n_mensagens = len(msgs)
    d.n_autoloops = sum(1 for u, v, _ in msgs if u == v)
    d.tempos = sorted(t for _, _, t in msgs)
    ids = sorted({x for u, v, _ in msgs for x in (u, v)})
    d.ids = ids
    idx = {v: i for i, v in enumerate(ids)}
    d.n = len(ids)

    cont = collections.Counter()
    primeira, ultima = {}, {}
    for u, v, t in msgs:
        if u == v:
            continue
        par = (idx[u], idx[v])
        cont[par] += 1
        primeira[par] = min(primeira.get(par, t), t)
        ultima[par] = max(ultima.get(par, t), t)
    d.cont_dir = cont                      # (u->v) -> nº de mensagens
    d.primeira, d.ultima = primeira, ultima
    d.out_adj = [[] for _ in range(d.n)]
    d.in_adj = [[] for _ in range(d.n)]
    for (u, v) in cont:
        d.out_adj[u].append(v)
        d.in_adj[v].append(u)

    msgs_nd = collections.Counter()
    for (u, v), c in cont.items():
        msgs_nd[(min(u, v), max(u, v))] += c
    d.msgs_nd = msgs_nd
    d.g = Grafo(d.n, [(u, v, 1.0 / c) for (u, v), c in msgs_nd.items()])
    comps = componentes(d.g.adj)
    d.componentes = comps
    d.gc, d.gc_nos = d.g.induzido(sorted(comps[0]))   # maior componente
    return d


def salvar_processado(d, pasta):
    """Grava as arestas tratadas (para o repositório) e um log do tratamento."""
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, 'arestas_dirigidas.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['origem', 'destino', 'n_mensagens', 'primeira_msg_unixts', 'ultima_msg_unixts'])
        for (u, v), c in sorted(d.cont_dir.items()):
            w.writerow([d.ids[u], d.ids[v], c, d.primeira[(u, v)], d.ultima[(u, v)]])
    with open(os.path.join(pasta, 'arestas_nao_dirigidas.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['u', 'v', 'n_mensagens_total', 'peso_1_sobre_m'])
        for (u, v), c in sorted(d.msgs_nd.items()):
            w.writerow([d.ids[u], d.ids[v], c, 1.0 / c])
