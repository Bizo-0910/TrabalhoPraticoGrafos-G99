"""Resultado do tratamento dos dados: etapas, amostras e distribuições (antes da Parte I)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import figuras as F
from .util import salvar_json


def executar(d, res_dir, fig_dir):
    os.makedirs(res_dir, exist_ok=True)
    m_dir = sum(len(a) for a in d.out_adj)
    etapas = pd.DataFrame([
        ('1. Arquivo original (1 linha = 1 mensagem)', d.n, d.n_mensagens, 'grafo temporal, multigrafo dirigido'),
        ('2. Remoção de auto-loops', d.n, d.n_mensagens - d.n_autoloops, f'{d.n_autoloops} auto-loops removidos'),
        ('3. Grafo estático simples dirigido', d.n, m_dir, 'pares (origem→destino) únicos'),
        ('4. Grafo estático simples não dirigido', d.n, d.g.m, 'aresta {u,v} se u→v ou v→u'),
        ('5. Maior componente conexa', d.gc.n, d.gc.m, f'{d.g.n - d.gc.n} vértices fora (pares isolados)'),
    ], columns=['Etapa', 'Vértices', 'Arestas', 'Observação'])
    etapas.to_csv(os.path.join(res_dir, 'parte0_etapas.csv'), index=False)

    cont = np.array(list(d.msgs_nd.values()))
    pesos = 1.0 / cont
    est = pd.DataFrame([
        ('mensagens por par (não dirigido)', int(cont.min()), float(np.median(cont)), float(cont.mean()), int(cont.max())),
        ('peso w = 1/m', float(pesos.min()), float(np.median(pesos)), float(pesos.mean()), float(pesos.max())),
    ], columns=['variável', 'mín', 'mediana', 'média', 'máx'])
    est.to_csv(os.path.join(res_dir, 'parte0_estatisticas.csv'), index=False)
    r = {'pares_1_msg': float((cont == 1).mean()), 'pares': int(len(cont)),
         'bidirecionais': float(np.mean([(v, u) in d.cont_dir for (u, v) in d.cont_dir]))}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    nomes = ['mensagens\n(original)', 'sem\nauto-loops', 'dirigido\nsimples', 'não dirigido\nsimples', 'maior\ncomponente']
    ax[0].bar(nomes, etapas['Arestas'], color=[F.CINZA, F.CINZA, F.AZUL, F.AZUL, F.VERDE])
    ax[0].set_ylim(0, etapas['Arestas'].max() * 1.15)
    for i, v in enumerate(etapas['Arestas']):
        ax[0].text(i, v + etapas['Arestas'].max() * 0.015, f'{v:,}'.replace(',', '.'), ha='center', fontsize=8)
    ax[0].set(ylabel='arestas', title='(a) Arestas após cada etapa do tratamento')
    ax[0].tick_params(axis='x', labelsize=8)
    ks, c = np.unique(cont, return_counts=True)
    ax[1].loglog(ks, c / c.sum(), 'o', ms=4, color=F.AZUL)
    ax[1].set(xlabel='mensagens trocadas pelo par (m)', ylabel='fração de pares',
              title=f'(b) Mensagens por par ({100 * r["pares_1_msg"]:.0f}% trocam só 1)')
    ax[2].hist(pesos, bins=40, color=F.LARANJA, edgecolor='white')
    ax[2].set_yscale('log')
    ax[2].set(xlabel='peso w = 1/m', ylabel='nº de arestas (log)', title='(c) Pesos das arestas não dirigidas')
    fig.tight_layout()
    F._salvar(fig, fig_dir, 'tratamento')
    salvar_json(r, os.path.join(res_dir, 'parte0.json'))
    return r, etapas