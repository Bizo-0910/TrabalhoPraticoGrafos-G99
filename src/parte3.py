"""Parte III: small-world, lei de potência, robustez, modelos clássicos (extra) e descobertas."""
import os
import numpy as np
import pandas as pd
from . import estrutura as S
from . import modelos as M
from . import smallworld as SW
from . import powerlaw as P
from . import robustez as R
from . import centralidade as CE
from . import figuras as F
from .util import salvar_json


# ------------------------------------------------------------------ 1. small-world
def questao1(d, rng, n_amostras=20):
    r = SW.avaliar(d.gc.adj, rng, n_amostras=n_amostras)
    return r


# ------------------------------------------------------------------ 2. lei de potência
def questao2(d, rng, B=200):
    graus = np.array([len(a) for a in d.g.adj])
    graus = graus[graus >= 1]
    aj = P.ajustar(graus)
    p, ks_sint = P.gof_bootstrap(graus, aj, B, rng)
    comp = P.comparar(graus, aj)
    ols = P.ajuste_ols_loglog(graus)
    # sensibilidade ao kmin: gamma e KS para alguns kmin
    sens = []
    for k0 in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        a = P.ajustar(graus, kmin=k0)
        sens.append({'kmin': k0, 'gamma': a['gamma'], 'ks': a['ks'], 'n_cauda': a['n_cauda']})
    # graus de entrada e de saída (ajuste sem bootstrap)
    extra = {}
    for nome, v in [('entrada', [len(a) for a in d.in_adj]), ('saida', [len(a) for a in d.out_adj])]:
        v = np.array([x for x in v if x >= 1])
        a = P.ajustar(v)
        extra[nome] = a
    r = {'ajuste': aj, 'p_valor': p, 'comparacao': comp, 'ols': ols, 'sensibilidade_kmin': sens,
         'bootstrap_B': B, 'n': int(len(graus)), 'entrada_saida': extra}
    return r, graus, ks_sint


# ------------------------------------------------------------------ 3. robustez
def questao3(d, rng, T=30, fracs=None, T_curva=10):
    gc = d.gc
    n = gc.n
    frac = 0.05
    r_ = R.tamanho_remocao(n, frac)
    base = R.metricas(gc.adj, [], rng)
    # centralidades (ranking fixo, na maior componente)
    c_grau = [len(a) for a in gc.adj]
    c_bet = CE.betweenness(gc.adj)
    pr_full = CE.pagerank(d.out_adj)
    c_pr = [pr_full[v] for v in d.gc_nos]
    ordens = {'grau': CE.ranking(c_grau), 'intermediação': CE.ranking(c_bet), 'PageRank': CE.ranking(c_pr)}
    reps = R.aleatorio(gc.adj, frac, T, rng)
    ataques = {nome: R.ataque(gc.adj, o, frac, rng) for nome, o in ordens.items()}
    # sobreposição entre rankings
    top = {k: set(o[:r_]) for k, o in ordens.items()}
    sobre = {f'{a}∩{b}': len(top[a] & top[b]) / r_ for a in top for b in top if a < b}
    if fracs is None:
        fracs = [0, .01, .02, .03, .04, .05, .075, .10, .15, .20, .25, .30]
    curvas = R.curva(gc.adj, ordens, fracs, T_curva, rng, k_fontes=200)
    resumo = {k: {'media': float(np.mean([x[k] for x in reps])), 'dp': float(np.std([x[k] for x in reps], ddof=1)),
                  'min': float(np.min([x[k] for x in reps])), 'max': float(np.max([x[k] for x in reps]))}
              for k in ['A', 'A_rel', 'B', 'C', 'D']}
    return {'n': n, 'r': r_, 'T': T, 'base': base, 'reps': reps, 'ataques': ataques, 'resumo_aleatorio': resumo,
            'sobreposicao_top': sobre, 'curvas': curvas,
            'centralidade_top5': {k: [int(d.ids[d.gc_nos[v]]) for v in o[:5]] for k, o in ordens.items()}}


# ------------------------------------------------------------------ extra: modelos clássicos
def _ks2(a, b):
    """Distância de Kolmogorov-Smirnov entre duas amostras de graus."""
    va = np.sort(a); vb = np.sort(b)
    todos = np.unique(np.concatenate([va, vb]))
    fa = np.searchsorted(va, todos, side='right') / len(va)
    fb = np.searchsorted(vb, todos, side='right') / len(vb)
    return float(np.abs(fa - fb).max())


def extra_modelos(d, rng, n_amostras=10, k_fontes=300):
    gc = d.gc
    n, m = gc.n, gc.m
    k_med = 2 * m / n
    arestas = [(min(u, v), max(u, v)) for u, v, _ in gc.arestas]
    kba = int(round(m / n))
    kws = int(2 * round(k_med / 2))
    gerar = {
        'ER': lambda: M.erdos_renyi_gnm(n, m, rng),
        'BA': lambda: M.barabasi_albert(n, kba, rng),
        'WS': lambda: M.watts_strogatz(n, kws, 0.1, rng),
        'mesmo grau': lambda: M.reconexao_preserva_grau(arestas, rng),
    }
    graus_obs = np.array([len(a) for a in gc.adj])
    C0, L0 = S.triangulos_e_clusterizacao(gc.adj)['clust_medio'], S.distancias_exatas(gc.adj)['caminho_medio']
    linhas = [{'modelo': 'observado', 'C': C0, 'L': L0, 'grau_max': int(graus_obs.max()), 'grau_dp': float(graus_obs.std()),
               'KS_grau': 0.0, 'm': m}]
    graus_mod = {}
    for nome, f in gerar.items():
        cs, ls, kmx, dps, ks, ms = [], [], [], [], [], []
        for i in range(n_amostras):
            ar = f()
            adj = SW._maior_componente(M.lista_adj(n, ar))
            gr = np.array([len(a) for a in M.lista_adj(n, ar)])
            cs.append(S.triangulos_e_clusterizacao(adj)['clust_medio'])
            ls.append(S.caminho_medio_amostrado(adj, rng, k_fontes))
            kmx.append(gr.max()); dps.append(gr.std()); ks.append(_ks2(gr, graus_obs)); ms.append(len(ar))
            if i == 0:
                graus_mod[nome] = gr
        linhas.append({'modelo': nome, 'C': float(np.mean(cs)), 'L': float(np.mean(ls)), 'grau_max': float(np.mean(kmx)),
                       'grau_dp': float(np.mean(dps)), 'KS_grau': float(np.mean(ks)), 'm': float(np.mean(ms))})
    tab = pd.DataFrame(linhas)
    return tab, graus_obs, graus_mod, {'k_BA': kba, 'k_WS': kws, 'p_WS': 0.1, 'n_amostras': n_amostras}


# ------------------------------------------------------------------ 4. descobertas
def descobertas(d):
    n = d.n
    gi = np.array([len(a) for a in d.in_adj]); go = np.array([len(a) for a in d.out_adj])
    conj = [set(a) for a in d.out_adj]
    # emissores em massa
    msgs_env = np.zeros(n); msgs_rec = np.zeros(n)
    for (u, v), c in d.cont_dir.items():
        msgs_env[u] += c; msgs_rec[v] += c
    top = np.argsort(-go)[:20]
    resp = []
    for u in top:
        resp.append(np.mean([u in conj[v] for v in d.out_adj[u]]))
    desc = {}
    desc['top20_saida_media'] = float(go[top].mean())
    desc['top20_entrada_media'] = float(gi[top].mean())
    desc['top20_taxa_resposta'] = float(np.mean(resp))
    desc['taxa_resposta_geral'] = float(np.mean([(u in conj[v]) for u in range(n) for v in d.out_adj[u]]))
    ordem = np.argsort(-msgs_env)
    tot = msgs_env.sum()
    desc['top1pct_msgs'] = float(msgs_env[ordem[:int(0.01 * n)]].sum() / tot)
    desc['top5pct_msgs'] = float(msgs_env[ordem[:int(0.05 * n)]].sum() / tot)
    desc['usuarios_que_enviaram'] = float((go > 0).mean())
    desc['usuarios_que_receberam'] = float((gi > 0).mean())
    desc['apenas_enviaram'] = float(((go > 0) & (gi == 0)).mean())
    desc['apenas_receberam'] = float(((go == 0) & (gi > 0)).mean())
    # pares com 1 mensagem
    cont_nd = np.array(list(d.msgs_nd.values()))
    desc['pares_1_msg'] = float((cont_nd == 1).mean())
    desc['pares_10mais_msgs'] = float((cont_nd >= 10).mean())
    desc['msgs_em_pares_10mais'] = float(cont_nd[cont_nd >= 10].sum() / cont_nd.sum())
    # tempo até a primeira resposta (pares recíprocos)
    delta = []
    for (u, v), t0 in d.primeira.items():
        if u < v and (v, u) in d.primeira:
            t1 = d.primeira[(v, u)]
            delta.append(abs(t1 - t0) / 3600)
    desc['pares_reciprocos'] = len(delta)
    desc['mediana_horas_entre_primeiras'] = float(np.median(delta))
    # concentração temporal
    t = np.array(d.tempos)
    t0 = t.min()
    desc['msgs_primeiras_8_semanas'] = float((t < t0 + 8 * 7 * 86400).mean())
    desc['msgs_primeiras_4_semanas'] = float((t < t0 + 4 * 7 * 86400).mean())
    # reciprocidade por faixa de grau (grau não dirigido)
    gr = np.array([len(a) for a in d.g.adj])
    faixas = [1, 2, 4, 8, 16, 32, 64, 128, 300]
    pts = []
    for lo, hi in zip(faixas[:-1], faixas[1:]):
        vs = [u for u in range(n) if lo <= gr[u] < hi and go[u] > 0]
        if len(vs) >= 5:
            pts.append((np.sqrt(lo * hi), float(np.mean([np.mean([u in conj[v] for v in d.out_adj[u]]) for u in vs]))))
    dados_fig = {'assimetria': {'entrada': gi, 'saida': go, 'top': {'entrada': gi[top], 'saida': go[top]}},
                 'reciprocidade_por_grau': {'k': [p[0] for p in pts], 'rec': [p[1] for p in pts]}}
    desc['reciprocidade_por_faixa'] = {f'{lo}-{hi - 1}': round(r, 3) for (lo, hi), (_, r) in zip(zip(faixas[:-1], faixas[1:]), pts)}
    return desc, dados_fig


# ------------------------------------------------------------------ orquestração
def executar(d, res_dir, fig_dir, rng, n_amostras_sw=20, B=200, T=30, verbose=True):
    os.makedirs(res_dir, exist_ok=True)
    out = {}
    if verbose: print('  [1/5] small-world', flush=True)
    out['q1'] = q1 = questao1(d, rng, n_amostras_sw)
    F.smallworld(q1, fig_dir)
    if verbose: print('  [2/5] lei de potência', flush=True)
    out['q2'], graus, ks_sint = questao2(d, rng, B)
    F.powerlaw(graus, out['q2']['ajuste'], out['q2']['comparacao'], out['q2']['ols'], ks_sint, out['q2']['p_valor'], fig_dir)
    if verbose: print('  [3/5] robustez', flush=True)
    q3 = questao3(d, rng, T)
    F.robustez_box(q3['reps'], q3['ataques'], q3['base'], fig_dir)
    F.robustez_curvas(q3['curvas'], fig_dir)
    out['q3'] = q3
    if verbose: print('  [4/5] descobertas', flush=True)
    desc, dados_fig = descobertas(d)
    F.descobertas(dados_fig, fig_dir)
    out['descobertas'] = desc
    if verbose: print('  [5/5] modelos clássicos', flush=True)
    tab_m, g_obs, g_mod, par = extra_modelos(d, rng)
    F.modelos(g_obs, g_mod, tab_m, fig_dir)
    out['extra'] = {'tabela': tab_m.to_dict(orient='records'), 'parametros': par}
    tab_m.to_csv(os.path.join(res_dir, 'parte3_modelos.csv'), index=False)
    # tabelas CSV
    pd.DataFrame([q1]).T.rename(columns={0: 'valor'}).to_csv(os.path.join(res_dir, 'parte3_smallworld.csv'))
    rr = pd.DataFrame([{'metrica': k, **v} for k, v in q3['resumo_aleatorio'].items()])
    rr.to_csv(os.path.join(res_dir, 'parte3_robustez_aleatoria.csv'), index=False)
    ta = pd.DataFrame([{'estrategia': 'íntegro', **q3['base']}] +
                      [{'estrategia': f'ataque {k}', **v} for k, v in q3['ataques'].items()])
    ta.to_csv(os.path.join(res_dir, 'parte3_robustez_ataques.csv'), index=False)
    salvar_json({k: v for k, v in out.items()}, os.path.join(res_dir, 'parte3.json'))
    return out
