"""Todas as figuras do trabalho (matplotlib). NetworkX é usado só para layout e coloração
das visualizações; nenhuma métrica é calculada com ele."""
import collections
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

AZUL, LARANJA, VERDE, CINZA, ROXO = '#2a6f97', '#e07a5f', '#5a9a6b', '#9aa5b1', '#81689d'
PALETA = ['#2a6f97', '#e07a5f', '#5a9a6b', '#f2cc8f', '#81689d', '#3d405b']

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .25, 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
})


def _salvar(fig, pasta, nome):
    os.makedirs(pasta, exist_ok=True)
    fig.savefig(os.path.join(pasta, nome + '.png'))
    fig.savefig(os.path.join(pasta, nome + '.pdf'))
    plt.close(fig)


def _ccdf(v):
    v = np.sort(np.asarray(v))
    ks, first = np.unique(v, return_index=True)
    return ks, 1 - first / len(v)


# ------------------------------------------------------------------ Parte I
def atividade(tempos, pasta):
    datas = pd.to_datetime(pd.Series(tempos), unit='s')
    sem = datas.dt.to_period('W').value_counts().sort_index()
    sem = sem.reindex(pd.period_range(sem.index.min(), sem.index.max(), freq='W'), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(len(sem)), sem.values, color=AZUL, width=.8)
    passo = max(1, len(sem) // 8)
    ax.set_xticks(range(0, len(sem), passo))
    ax.set_xticklabels([str(p.start_time.date()) for p in sem.index[::passo]], rotation=30, ha='right')
    ax.set_ylabel('mensagens por semana')
    ax.set_title(f'Atividade da rede ({datas.min().date()} a {datas.max().date()})')
    _salvar(fig, pasta, 'atividade_semanal')


def graus(g_nd, g_in, g_out, pasta):
    gv = np.asarray(g_nd)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    bins = np.unique(np.logspace(0, np.log10(gv.max() + 1), 25).astype(int))
    ax[0, 0].hist(gv, bins=bins, color=AZUL, edgecolor='white')
    ax[0, 0].set_xscale('log'); ax[0, 0].set_yscale('log')
    ax[0, 0].axvline(gv.mean(), color=LARANJA, ls='--', label=f'média = {gv.mean():.1f}')
    ax[0, 0].axvline(np.median(gv), color=VERDE, ls='--', label=f'mediana = {np.median(gv):.0f}')
    ax[0, 0].set(xlabel='grau', ylabel='nº de vértices', title='(a) Histograma do grau (bins logarítmicos)')
    ax[0, 0].legend()
    x, y = _ccdf(gv)
    ax[0, 1].loglog(x, y, '.', color=AZUL, ms=4)
    ax[0, 1].set(xlabel='grau k', ylabel='P(K ≥ k)', title='(b) CCDF do grau (log-log)')
    ei, eo = np.asarray(g_in), np.asarray(g_out)
    ax[1, 0].scatter(ei + 1, eo + 1, s=10, alpha=.35, color=LARANJA, edgecolor='none')
    lim = max(ei.max(), eo.max()) + 1
    ax[1, 0].plot([1, lim], [1, lim], color=CINZA, ls='--', lw=1)
    ax[1, 0].set_xscale('log'); ax[1, 0].set_yscale('log')
    ax[1, 0].set(xlabel='grau de entrada + 1', ylabel='grau de saída + 1', title='(c) Contatos que enviaram × contatos para quem enviou')
    for v, nome, cor in [(ei, 'entrada', AZUL), (eo, 'saída', LARANJA)]:
        x, y = _ccdf(v[v > 0])
        ax[1, 1].loglog(x, y, '.', ms=4, color=cor, label=nome)
    ax[1, 1].set(xlabel='grau k', ylabel='P(K ≥ k)', title='(d) CCDF dos graus de entrada e saída')
    ax[1, 1].legend()
    fig.tight_layout()
    _salvar(fig, pasta, 'distribuicao_graus')


def distancias(hist, ecc, L, pasta):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ds = sorted(hist)
    tot = sum(hist.values())
    ax[0].bar(ds, [hist[d] / tot * 100 for d in ds], color=AZUL)
    ax[0].axvline(L, color=LARANJA, ls='--', label=f'média = {L:.2f}')
    ax[0].set(xlabel='distância (nº de arestas)', ylabel='% dos pares de vértices', title='(a) Distribuição das distâncias')
    ax[0].legend()
    ec = collections.Counter(ecc.values())
    ks = sorted(ec)
    ax[1].bar(ks, [ec[k] for k in ks], color=VERDE)
    ax[1].set(xlabel='excentricidade', ylabel='nº de vértices',
              title=f'(b) Excentricidade (raio = {min(ks)}, diâmetro = {max(ks)})')
    fig.tight_layout()
    _salvar(fig, pasta, 'distancias')


def clusterizacao(grau, clust, pasta):
    ks, cs = np.asarray(grau), np.asarray(clust)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(cs, bins=30, color=AZUL, edgecolor='white'); ax[0].set_yscale('log')
    ax[0].set(xlabel='coeficiente de clusterização local', ylabel='nº de vértices (log)', title='(a) Clusterização local')
    mask = ks >= 2
    ax[1].scatter(ks[mask], cs[mask], s=8, alpha=.3, color=CINZA, edgecolor='none')
    df = pd.DataFrame({'k': ks[mask], 'c': cs[mask]}).groupby('k')['c'].mean()
    ax[1].plot(df.index, df.values, color=LARANJA, lw=1.8, label='média por grau')
    ax[1].set_xscale('log')
    ax[1].set(xlabel='grau', ylabel='clusterização local', title='(b) Clusterização × grau (grau ≥ 2)')
    ax[1].legend()
    fig.tight_layout()
    _salvar(fig, pasta, 'clusterizacao')


def kcore(core, pasta):
    cc = collections.Counter(core)
    ks = sorted(cc)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(ks, [cc[k] for k in ks], color=AZUL)
    ax.set_yscale('log')
    ax.set(xlabel='índice de core (k)', ylabel='nº de vértices (log)',
           title=f'Decomposição k-core (core máximo = {max(ks)}, {cc[max(ks)]} vértices)')
    _salvar(fig, pasta, 'kcore')


def _nx_grafo(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from((u, v) for u, v, _ in g.arestas)
    return G


def grafo_nucleo(gc, core, pasta, alvo=350):
    """Subgrafo dos vértices de maior core (~alvo vértices), cor = comunidade (Louvain), tamanho = grau."""
    G = _nx_grafo(gc)
    kmax = max(core)
    lim = kmax
    while sum(1 for c in core if c >= lim - 1) < alvo and lim > 1:
        lim -= 1
    nucleo = [v for v in range(gc.n) if core[v] >= lim]
    com = sorted(nx.community.louvain_communities(G, seed=1), key=len, reverse=True)
    com_de = {v: i for i, c in enumerate(com) for v in c}
    H = G.subgraph(nucleo)
    pos = nx.spring_layout(H, k=1.6 / np.sqrt(len(H)), iterations=300, seed=7)
    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=.10, width=.5, edge_color='#444')
    cores = [PALETA[com_de[v]] if com_de[v] < len(PALETA) else CINZA for v in H]
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[12 + 3.2 * H.degree(v) for v in H],
                           node_color=cores, alpha=.9, linewidths=.4, edgecolors='white')
    hubs = sorted(H, key=lambda v: -H.degree(v))[:6]
    nx.draw_networkx_labels(H, pos, {v: str(v) for v in hubs}, ax=ax, font_size=8, font_weight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.8))
    ax.set_title(f'Núcleo da rede: {len(H)} vértices com core ≥ {lim} (cor = comunidade, tamanho = grau)')
    ax.axis('off'); ax.grid(False)
    _salvar(fig, pasta, 'grafo_nucleo')
    return {'limiar_core': lim, 'n_nucleo': len(H), 'n_comunidades': len(com),
            'modularidade': nx.community.modularity(G, com)}


def grafo_ego(d, pasta):
    """Rede ego do vértice de maior grau, com o sentido das mensagens."""
    g = d.g
    hub = max(range(g.n), key=lambda v: len(g.adj[v]))
    viz = set(d.out_adj[hub]) | set(d.in_adj[hub])
    G = _nx_grafo(g)
    E = G.subgraph(viz | {hub}).copy()
    pos = nx.spring_layout(E, pos={hub: (0, 0)}, fixed=[hub], k=1.2 / np.sqrt(len(E)), iterations=300, seed=3)
    saida = set(d.out_adj[hub]); entra = set(d.in_adj[hub])

    def tipo(v):
        a, b = v in saida, v in entra
        return 'ambos' if a and b else ('hub enviou' if a else 'hub recebeu')
    cor = {'ambos': VERDE, 'hub enviou': AZUL, 'hub recebeu': LARANJA}
    cont = collections.Counter(tipo(v) for v in viz)
    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(E, pos, ax=ax, alpha=.08, width=.5, edge_color='#444')
    nx.draw_networkx_edges(E, pos, edgelist=[(hub, v) for v in viz], ax=ax, alpha=.35, width=.8, edge_color='#444')
    nx.draw_networkx_nodes(E, pos, nodelist=list(viz), ax=ax, node_size=[14 + 2 * E.degree(v) for v in viz],
                           node_color=[cor[tipo(v)] for v in viz], linewidths=.4, edgecolors='white')
    nx.draw_networkx_nodes(E, pos, nodelist=[hub], ax=ax, node_size=420, node_color='#111', edgecolors='white')
    for t, c in cor.items():
        ax.scatter([], [], color=c, label=f'{t} ({cont[t]})')
    ax.legend(loc='lower left', frameon=False)
    ax.set_title(f'Rede ego do vértice {d.ids[hub]} (grau {len(g.adj[hub])}): vizinhos e laços entre eles')
    ax.axis('off'); ax.grid(False)
    _salvar(fig, pasta, 'grafo_ego_hub')
    return {'hub_id': d.ids[hub], 'hub_grau': len(g.adj[hub]), 'hub_enviou': cont['hub enviou'],
            'hub_recebeu': cont['hub recebeu'], 'hub_ambos': cont['ambos']}


# ------------------------------------------------------------------ Parte II
def tempos(tab, titulo, pasta):
    t = tab.iloc[::-1]
    erro = (t['ic_hi'] - t['ic_lo']) / 2
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(t['algoritmo'], t['media'], xerr=erro, color=AZUL, ecolor='#111', capsize=3)
    ax.set_xscale('log')
    ax.set(xlabel='tempo médio por execução (s), escala log; barra = IC 95%', title=titulo)
    _salvar(fig, pasta, 'tempos_algoritmos')


def complexidade(obs, teoria, f_teo, inclinacoes, pasta):
    nomes = list(obs)
    lin, col = 2, 4
    fig, axs = plt.subplots(lin, col, figsize=(15, 6.5))
    axs = axs.ravel()
    for ax, nome in zip(axs, nomes):
        d = np.array(obs[nome])
        x = np.array([f_teo(nome, V, E) for V, E, *_ in d]); y = d[:, 2]
        a, b = inclinacoes[nome]
        ax.errorbar(x, y, yerr=[y - d[:, 3], d[:, 4] - y], fmt='o', color=AZUL, capsize=2)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.exp(b) * xs ** a, '--', color=LARANJA, label=f'inclinação = {a:.2f}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set(xlabel=f'f(V, E) = {teoria[nome]}', ylabel='tempo (s)', title=nome)
        ax.legend(loc='upper left')
    for ax in axs[len(nomes):]:
        ax.axis('off')
    fig.tight_layout()
    _salvar(fig, pasta, 'complexidade_observada')


# ------------------------------------------------------------------ Parte III
def smallworld(r, pasta):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    nomes = ['ER G(n,m)', 'mesmo grau']
    chaves = ['er', 'grau']
    cores = [AZUL, VERDE]
    ax[0].bar(['observado'] + nomes, [r['C']] + [r[f'C_{k}'] for k in chaves],
              yerr=[0] + [r[f'C_{k}_dp'] for k in chaves], color=[LARANJA] + cores, capsize=3)
    ax[0].set(ylabel='clusterização média C', title='(a) C: observado × nulos')
    ax[1].bar(['observado'] + nomes, [r['L']] + [r[f'L_{k}'] for k in chaves],
              yerr=[0] + [r[f'L_{k}_dp'] for k in chaves], color=[LARANJA] + cores, capsize=3)
    ax[1].set_ylim(2.5, 3.3)
    ax[1].set(ylabel='caminho médio L', title='(b) L: observado × nulos')
    ax[2].bar(nomes, [r[f'sigma_{k}'] for k in chaves], color=cores)
    ax[2].axhline(1, color='#111', ls='--', lw=1)
    ax[2].set_yscale('log'); ax[2].set_ylim(0.8, max(r['sigma_er'], r['sigma_grau']) * 2.2)
    ax[2].set(ylabel='σ = (C/C_rand)/(L/L_rand)', title='(c) Índice small-world σ (log)')
    for i, k in enumerate(chaves):
        ax[2].text(i, r[f'sigma_{k}'] * 1.08, f"{r[f'sigma_{k}']:.2f}", ha='center')
    fig.tight_layout()
    _salvar(fig, pasta, 'smallworld')


def powerlaw(x, ajuste, comparacao, ols, ks_sint, p_valor, pasta, rotulo='grau'):
    from .powerlaw import pmf_pl
    from scipy.special import zeta
    x = np.asarray(x)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ks, P = _ccdf(x)
    ax[0].loglog(ks, P, 'o', ms=3.5, color=AZUL, label='dados')
    kmin, g = ajuste['kmin'], ajuste['gamma']
    kk = np.arange(kmin, x.max() + 1)
    S_pl = zeta(g, kk) / zeta(g, kmin) * (x >= kmin).mean()
    ax[0].loglog(kk, S_pl, '-', color=LARANJA, lw=2, label=f'lei de potência (γ = {g:.2f}, k_min = {kmin})')
    ln = comparacao['lognormal']['parametros']
    from scipy.stats import norm
    ks_all = np.arange(kmin, x.max() * 5 + 1, dtype=float)
    lg = -(np.log(ks_all) - ln['mu']) ** 2 / (2 * ln['sigma'] ** 2) - np.log(ks_all)
    pm = np.exp(lg - lg.max()); pm /= pm.sum()
    surv = 1 - np.concatenate(([0], np.cumsum(pm)[:-1]))
    ax[0].loglog(ks_all[:len(kk)], (surv[:len(kk)]) * (x >= kmin).mean(), '--', color=VERDE, lw=1.8, label='lognormal (mesma cauda)')
    lam = comparacao['exponencial']['parametros']['lambda']
    ax[0].loglog(kk, np.exp(-lam * (kk - kmin)) * (x >= kmin).mean(), ':', color=ROXO, lw=2, label='exponencial (mesma cauda)')
    ax[0].axvline(kmin, color=CINZA, ls=':', lw=1)
    ax[0].set(xlabel=f'{rotulo} k', ylabel='P(K ≥ k)', title='(a) CCDF e ajustes')
    ax[0].set_ylim(P.min() / 2, 1.5)
    ax[0].legend(fontsize=8, loc='lower left')
    # (b) PMF empírica e OLS
    k2, c2 = np.unique(x, return_counts=True)
    p2 = c2 / c2.sum()
    ax[1].loglog(k2, p2, 'o', ms=3.5, color=AZUL, label='P(k) empírica')
    xs = np.linspace(k2.min(), k2.max(), 100)
    ax[1].loglog(xs, np.exp(ols['log_C']) * xs ** (-ols['gamma_ols']), '--', color=LARANJA,
                 label=f"reta log-log por MQO (γ = {ols['gamma_ols']:.2f})")
    ax[1].loglog(kk, pmf_pl(kk, g, kmin) * (x >= kmin).mean(), '-', color=VERDE, label=f'MLE (γ = {g:.2f})')
    ax[1].set(xlabel=f'{rotulo} k', ylabel='P(k)', title='(b) Reta do Apêndice B (MQO) × MLE')
    ax[1].legend(fontsize=8, loc='lower left')
    # (c) bootstrap
    ax[2].hist(ks_sint, bins=20, color=CINZA, edgecolor='white')
    ax[2].axvline(ajuste['ks'], color=LARANJA, lw=2, label=f"KS observado = {ajuste['ks']:.3f}")
    ax[2].set(xlabel='KS das amostras sintéticas', ylabel='frequência',
              title=f'(c) Qualidade do ajuste (p = {p_valor:.2f})')
    ax[2].legend()
    fig.tight_layout()
    _salvar(fig, pasta, 'lei_de_potencia')


def robustez_box(reps, ataques, base, pasta):
    """reps: lista de dicts (aleatório); ataques: nome -> dict; base: dict do grafo íntegro."""
    nomes = {'A': 'A: maior componente (fração de n)', 'B': 'B: nº de componentes',
             'C': 'C: caminho médio (maior comp.)', 'D': 'D: fração de isolados'}
    cor_at = {'grau': LARANJA, 'intermediação': VERDE, 'PageRank': ROXO}
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.3))
    for a, k in zip(ax, 'ABCD'):
        v = np.array([r[k] for r in reps])
        a.boxplot(v, positions=[0], widths=.5, patch_artist=True, showmeans=False,
                  boxprops=dict(facecolor='#d7e6f1'), medianprops=dict(color='#111'))
        a.errorbar([0], [v.mean()], yerr=[v.std(ddof=1)], fmt='o', color=AZUL, capsize=6, label='média ± dp (aleatório)')
        a.axhline(base[k], color=CINZA, ls='--', lw=1, label='grafo íntegro')
        for nome, res in ataques.items():
            a.scatter([1], [res[k]], marker='D', s=55, color=cor_at.get(nome, '#111'), label=f'ataque: {nome}', zorder=3)
        a.set_xticks([0, 1]); a.set_xticklabels(['aleatório\n(T rep.)', 'ataque\ndirecionado'])
        a.set_xlim(-.6, 1.6)
        a.set_title(nomes[k], fontsize=10)
    ax[0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    _salvar(fig, pasta, 'robustez_boxplot')


def robustez_curvas(res, pasta):
    fr = np.array(res['fracs']) * 100
    nomes = {'A': 'A: maior componente (fração de n)', 'B': 'B: nº de componentes',
             'C': 'C: caminho médio (maior comp.)', 'D': 'D: fração de isolados'}
    cores = {'aleatório': AZUL, 'grau': LARANJA, 'intermediação': VERDE, 'PageRank': ROXO}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    for a, k in zip(ax, 'ABCD'):
        for nome, cor in cores.items():
            if nome not in res:
                continue
            y = np.array(res[nome][k])
            a.plot(fr, y, '-o', ms=3, color=cor, label=nome)
            if nome == 'aleatório':
                dp = np.array(res['aleatório_dp'][k])
                a.fill_between(fr, y - dp, y + dp, color=cor, alpha=.2)
        a.axvline(5, color=CINZA, ls=':', lw=1)
        a.set(xlabel='% de vértices removidos', title=nomes[k])
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    _salvar(fig, pasta, 'robustez_curvas')


def modelos(graus_obs, graus_mod, tab, pasta):
    """CCDF do grau observado × modelos clássicos; barras de C e L."""
    cores = {'observado': '#111', 'ER': AZUL, 'BA': LARANJA, 'WS': VERDE, 'mesmo grau': ROXO}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x, y = _ccdf(graus_obs)
    ax[0].loglog(x, y, 'o', ms=3.5, color='#111', label='observado')
    for nome, gs in graus_mod.items():
        if nome == 'mesmo grau':
            continue
        x, y = _ccdf(gs)
        ax[0].loglog(x, y, '-', lw=1.8, color=cores[nome], label=nome)
    ax[0].set(xlabel='grau k', ylabel='P(K ≥ k)', title='(a) CCDF do grau: observado × modelos')
    ax[0].legend()
    nm = list(tab['modelo'])
    ax[1].bar(nm, tab['C'], color=[cores[n] for n in nm])
    ax[1].set(ylabel='clusterização média', title='(b) Clusterização')
    ax[2].bar(nm, tab['L'], color=[cores[n] for n in nm])
    ax[2].set_ylim(2.5, max(tab['L']) * 1.05)
    ax[2].set(ylabel='caminho médio', title='(c) Caminho médio')
    for a in ax[1:]:
        a.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    _salvar(fig, pasta, 'modelos_classicos')


def descobertas(dados, pasta):
    """dados: dict com 'razao' (out/in dos usuários), 'rec_hubs', 'cum_tempo' ..."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = dados['assimetria']
    ax[0].scatter(a['entrada'] + 1, a['saida'] + 1, s=10, alpha=.35, color=CINZA, edgecolor='none')
    top = a['top']
    ax[0].scatter(top['entrada'] + 1, top['saida'] + 1, s=35, color=LARANJA, edgecolor='white', label='20 maiores emissores')
    lim = max(a['entrada'].max(), a['saida'].max()) + 1
    ax[0].plot([1, lim], [1, lim], '--', color='#111', lw=1)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set(xlabel='pessoas de quem recebeu + 1', ylabel='pessoas para quem enviou + 1',
              title='(a) Emissores em massa: enviam para muitos, recebem de poucos')
    ax[0].legend()
    r = dados['reciprocidade_por_grau']
    ax[1].plot(r['k'], r['rec'], '-o', ms=4, color=AZUL)
    ax[1].set_xscale('log')
    ax[1].set(xlabel='grau do vértice (faixas)', ylabel='fração de contatos que respondem',
              title='(b) Reciprocidade cai com o grau')
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    _salvar(fig, pasta, 'descobertas')
