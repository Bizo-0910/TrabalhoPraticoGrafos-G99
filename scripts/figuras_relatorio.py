"""Gera as figuras compactas do relatório (largura de coluna ou de página, fonte 6-7 pt).

Usa os resultados já salvos em resultados/ e recalcula só o que é barato (graus, k-core).
Saída: relatorio/figuras/*.pdf
Uso:   python scripts/figuras_relatorio.py
"""
import collections
import contextlib
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from src import tratamento as T, estrutura as S, figuras as F, parte2, parte3  # noqa: E402
from src import powerlaw as P  # noqa: E402

SAIDA = os.path.join(RAIZ, 'relatorio', 'figuras')
J = lambda f: json.load(open(os.path.join(RAIZ, 'resultados', f), encoding='utf-8'))
A, LA, V, C, R = F.AZUL, F.LARANJA, F.VERDE, F.CINZA, F.ROXO
COL, PAG = 3.4, 7.1      # larguras (pol.) de uma coluna e da página, no template WebMedia

ESTILO = {'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5, 'xtick.labelsize': 6,
          'ytick.labelsize': 6, 'legend.fontsize': 5.8, 'axes.titleweight': 'bold',
          'lines.linewidth': 1.0, 'savefig.dpi': 300}


@contextlib.contextmanager
def tamanho(w, h):
    """Força o figsize de F.* (que fixam tamanhos grandes) para o tamanho final no papel."""
    orig = plt.subplots
    def sub(*a, **k):
        k['figsize'] = (w, h)
        return orig(*a, **k)
    plt.subplots = sub
    try:
        yield
    finally:
        plt.subplots = orig


def salvar(nome):
    os.makedirs(SAIDA, exist_ok=True)


def main():
    d = T.construir(T.ler_mensagens(os.path.join(RAIZ, 'data', 'CollegeMsg.txt')))
    p1, p2, p3 = J('parte1.json'), J('parte2.json'), J('parte3.json')
    plt.rcParams.update(ESTILO)

    # 0. tratamento dos dados (largura da página): arestas por etapa, mensagens por par e pesos
    m_dir = sum(len(a) for a in d.out_adj)
    arestas = [d.n_mensagens, d.n_mensagens - d.n_autoloops, m_dir, d.g.m, d.gc.m]
    cont = np.array(list(d.msgs_nd.values()))
    pesos = 1.0 / cont
    fig, ax = plt.subplots(1, 3, figsize=(PAG, 2.0))
    nomes_et = ['original', 'sem\nloops', 'dirigido', 'não\ndirigido', 'maior\ncomp.']
    ax[0].bar(nomes_et, arestas, color=[C, C, A, A, V])
    ax[0].set_ylim(0, max(arestas) * 1.18)
    for i, vv in enumerate(arestas):
        ax[0].text(i, vv + max(arestas) * .02, f'{vv:,}'.replace(',', '.'), ha='center', fontsize=5.5)
    ax[0].set(ylabel='arestas', title='(a) Arestas após cada etapa')
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', '.')))
    ax[0].tick_params(axis='x', labelsize=5.5)
    ks_, cs_ = np.unique(cont, return_counts=True)
    ax[1].loglog(ks_, cs_ / cs_.sum(), 'o', ms=2.5, color=A)
    ax[1].set(xlabel='mensagens trocadas pelo par (m)', ylabel='fração de pares',
              title=f'(b) Mensagens por par ({100 * (cont == 1).mean():.0f}% trocam só 1)')
    ax[2].hist(pesos, bins=40, color=LA, edgecolor='white', linewidth=.3)
    ax[2].set_yscale('log')
    ax[2].set(xlabel='peso w = 1/m', ylabel='nº de arestas (log)', title='(c) Pesos das arestas')
    fig.tight_layout(pad=.5)
    F._salvar(fig, SAIDA, 'tratamento')

    # 1. atividade semanal (uma coluna)
    with tamanho(COL, 1.9):
        F.atividade(d.tempos, SAIDA)

    # 2. estrutura: CCDF do grau, distâncias e k-core (largura da página)
    gr = np.array([len(a) for a in d.g.adj])
    core = S.kcore(d.g.adj)
    hist = {int(k): v for k, v in p1['hist_distancias'].items()}
    tot = sum(hist.values())
    fig, ax = plt.subplots(1, 3, figsize=(PAG, 2.0))
    x, y = F._ccdf(gr[gr >= 1])
    ax[0].loglog(x, y, '.', ms=2.5, color=A)
    ax[0].set(xlabel='grau k', ylabel='P(K ≥ k)', title='(a) CCDF do grau')
    ds = sorted(hist)
    ax[1].bar(ds, [hist[k] / tot * 100 for k in ds], color=A)
    ax[1].axvline(p1['caminho_medio'], color=LA, ls='--', label=f"média = {p1['caminho_medio']:.2f}")
    ax[1].set(xlabel='distância (arestas)', ylabel='% dos pares', title='(b) Distâncias (maior componente)')
    ax[1].legend()
    cc = collections.Counter(core)
    ks = sorted(cc)
    ax[2].bar(ks, [cc[k] for k in ks], color=A)
    ax[2].set_yscale('log')
    ax[2].set(xlabel='índice k do núcleo', ylabel='nº de vértices',
              title=f'(c) Núcleos k (máx. = {max(ks)}, {cc[max(ks)]} vértices)')
    fig.tight_layout(pad=.5)
    F._salvar(fig, SAIDA, 'estrutura')

    # 3. complexidade observada (largura da página)
    obs = p2['obs']
    obs = {k: v for k, v in obs.items() if k in parte2.TEORIA}
    incl = {}
    for nome, dd in obs.items():
        dd = np.array(dd)
        x = np.array([parte2.f_teo(nome, Vv, Ee) for Vv, Ee, *_ in dd])
        incl[nome] = tuple(np.polyfit(np.log(x), np.log(dd[:, 2]), 1))
    with tamanho(PAG, 3.0):
        F.complexidade(obs, parte2.TEORIA, parte2.f_teo, incl, SAIDA)

    # 4. lei de potência (largura da página; bootstrap refeito para o histograma)
    q2 = p3['q2']
    graus = gr[gr >= 1]
    aj = P.ajustar(graus)
    _, ks_sint = P.gof_bootstrap(graus, aj, q2['bootstrap_B'], np.random.default_rng(42))
    F.LEG_FONTE = 5.5
    with tamanho(PAG, 2.3):
        F.powerlaw(graus, q2['ajuste'], q2['comparacao'], q2['ols'], ks_sint, q2['p_valor'], SAIDA)

    # 5. robustez: boxplots (linha de cima) e curvas (linha de baixo)
    q3 = p3['q3']
    nomes = {'A': 'A: maior componente', 'B': 'B: nº de componentes', 'C': 'C: caminho médio', 'D': 'D: fração de isolados'}
    cor_at = {'grau': LA, 'intermediação': V, 'PageRank': R}
    fig, ax = plt.subplots(2, 4, figsize=(PAG, 3.9))
    for a, k in zip(ax[0], 'ABCD'):
        v = np.array([r[k] for r in q3['reps']])
        a.boxplot(v, positions=[0], widths=.5, patch_artist=True, boxprops=dict(facecolor='#d7e6f1'),
                  medianprops=dict(color='#111'), flierprops=dict(markersize=2))
        a.axhline(q3['base'][k], color=C, ls='--', lw=.8, label='grafo íntegro')
        for nome, res in q3['ataques'].items():
            a.scatter([1.6], [res[k]], marker='D', s=14, color=cor_at.get(nome, '#111'), label=f'ataque: {nome}', zorder=3)
        a.set_xticks([0, 1.6]); a.set_xticklabels(['aleatório\n(30 rep.)', 'ataque'])
        a.set_xlim(-.6, 2.1); a.set_title(nomes[k])
    ax[0, 3].legend(*ax[0, 0].get_legend_handles_labels(), loc='upper left', fontsize=5.3)
    fr = np.array(q3['curvas']['fracs']) * 100
    cores = {'aleatório': A, 'grau': LA, 'intermediação': V, 'PageRank': R}
    for a, k in zip(ax[1], 'ABCD'):
        for nome, cor in cores.items():
            y = np.array(q3['curvas'][nome][k])
            a.plot(fr, y, '-o', ms=1.8, color=cor, label=nome)
            if nome == 'aleatório':
                dp = np.array(q3['curvas']['aleatório_dp'][k])
                a.fill_between(fr, y - dp, y + dp, color=cor, alpha=.2)
        a.axvline(5, color=C, ls=':', lw=.8)
        a.set(xlabel='% de vértices removidos', title=nomes[k])
    ax[1, 0].legend()
    fig.tight_layout(pad=.5)
    F._salvar(fig, SAIDA, 'robustez')

    # 6. descobertas (largura da página)
    _, dados_fig = parte3.descobertas(d)
    with tamanho(PAG, 2.4):
        F.descobertas(dados_fig, SAIDA)
    print('figuras em', SAIDA)


if __name__ == '__main__':
    main()
