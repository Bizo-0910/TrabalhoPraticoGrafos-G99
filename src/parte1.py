"""Parte I: análise estrutural obrigatória."""
import numpy as np
import pandas as pd
from . import estrutura as S
from . import figuras as F
from .util import salvar_json
import os


def executar(d, res_dir, fig_dir, rng):
    g, gc = d.g, d.gc
    r = {}
    r['n_mensagens'] = d.n_mensagens
    r['n_autoloops'] = d.n_autoloops
    r['n'] = d.n
    r['m_dirigido'] = sum(len(a) for a in d.out_adj)
    r['m'] = g.m
    r['dias'] = (d.tempos[-1] - d.tempos[0]) / 86400

    gr = [len(a) for a in g.adj]
    gi = [len(a) for a in d.in_adj]
    go = [len(a) for a in d.out_adj]
    r['graus'] = {'nao_dirigido': S.resumo_graus(gr), 'entrada': S.resumo_graus(gi), 'saida': S.resumo_graus(go)}
    r['densidade'] = S.densidade(g.n, g.m)
    r['densidade_dirigida'] = S.densidade(g.n, r['m_dirigido'], dirigido=True)
    r['reciprocidade'] = S.reciprocidade(d.out_adj)
    r['assortatividade'] = S.assortatividade_grau(g.adj)
    r['n_isolados'] = sum(1 for k in gr if k == 0)

    tam = [len(c) for c in d.componentes]
    r['n_componentes'] = len(tam)
    r['tamanho_componentes'] = tam
    r['gc_n'], r['gc_m'] = gc.n, gc.m
    r['gc_fracao'] = gc.n / g.n

    dist = S.distancias_exatas(gc.adj)
    r['diametro'], r['raio'], r['caminho_medio'] = dist['diametro'], dist['raio'], dist['caminho_medio']
    r['hist_distancias'] = {str(k): v for k, v in dist['hist'].items()}

    tri = S.triangulos_e_clusterizacao(g.adj)
    r['triangulos'] = tri['triangulos']
    r['clust_medio'] = tri['clust_medio']
    r['transitividade'] = tri['transitividade']
    core = S.kcore(g.adj)
    r['core_max'] = max(core)
    r['n_core_max'] = sum(1 for c in core if c == max(core))

    # figuras
    F.atividade(d.tempos, fig_dir)
    F.graus(gr, gi, go, fig_dir)
    F.distancias(dist['hist'], dist['ecc'], dist['caminho_medio'], fig_dir)
    F.clusterizacao(gr, tri['clust_local'], fig_dir)
    F.kcore(core, fig_dir)
    gc_core = [core[v] for v in d.gc_nos]
    r['visual_nucleo'] = F.grafo_nucleo(gc, gc_core, fig_dir)
    r['visual_ego'] = F.grafo_ego(d, fig_dir)

    # tabela computado / não computado
    g_ = r['graus']['nao_dirigido']
    tabela = pd.DataFrame([
        ('Número de vértices', 'Sim', f"{g.n:,}"),
        ('Número de arestas', 'Sim', f"{r['m_dirigido']:,} (dirigido) / {g.m:,} (não dirigido)"),
        ('Grau mínimo, máximo e médio', 'Sim', f"{g_['min']}, {g_['max']}, {g_['media']:.2f}"),
        ('Distribuição de graus', 'Sim', 'figura distribuicao_graus'),
        ('Densidade', 'Sim', f"{r['densidade']:.5f}"),
        ('Número de componentes conexas', 'Sim', str(len(tam))),
        ('Tamanho de cada componente', 'Sim', str(tam)),
        ('Diâmetro', 'Sim (maior componente, exato)', str(r['diametro'])),
        ('Raio', 'Sim (maior componente, exato)', str(r['raio'])),
        ('Comprimento médio dos caminhos', 'Sim (maior componente, exato)', f"{r['caminho_medio']:.4f}"),
        ('Coeficiente de clusterização médio', 'Sim', f"{r['clust_medio']:.4f}"),
        ('Número de triângulos', 'Sim (viável)', f"{r['triangulos']:,}"),
        ('Visualização do grafo', 'Sim (versão reduzida)', 'núcleo (core ≥ %d) e rede ego do maior hub' % r['visual_nucleo']['limiar_core']),
    ], columns=['Medida', 'Computada?', 'Valor'])
    os.makedirs(res_dir, exist_ok=True)
    tabela.to_csv(os.path.join(res_dir, 'parte1_tabela.csv'), index=False)
    pd.DataFrame(r['graus']).T.to_csv(os.path.join(res_dir, 'parte1_graus.csv'))
    salvar_json(r, os.path.join(res_dir, 'parte1.json'))
    return r, tabela
