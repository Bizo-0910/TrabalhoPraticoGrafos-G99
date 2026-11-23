"""Executa o pipeline completo (tratamento + Partes I, II, III) sem Jupyter.

Uso:   python scripts/run_all.py            (tudo, ~7 minutos)
       python scripts/run_all.py --rapido   (menos repetições, ~2 minutos)
Saídas: resultados/*.csv|json|tex, figuras/*.png|pdf, data/processed/*.csv
"""
import argparse
import os
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from src import tratamento as T, parte0, parte1, parte2, parte3, macros          # noqa: E402
from src.util import salvar_json                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--rapido', action='store_true', help='menos repetições (resultados aproximados)')
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

dados = os.path.join(RAIZ, 'data', 'CollegeMsg.txt')
res, fig = os.path.join(RAIZ, 'resultados'), os.path.join(RAIZ, 'figuras')
t0 = time.time()

print('Tratamento...')
d = T.construir(T.ler_mensagens(dados))
T.salvar_processado(d, os.path.join(RAIZ, 'data', 'processed'))
print(f'  {d.n} vértices | {d.g.m} arestas | maior componente: {d.gc.n}')

print('Tratamento: etapas e estatísticas...')
parte0.executar(d, res, fig)

print('Parte I...')
p1, _ = parte1.executar(d, res, fig, np.random.default_rng(a.seed))

print('Parte II (mede tempo: não use o computador para outra coisa)...')
if a.rapido:
    p2, *_ = parte2.executar(d, res, fig, np.random.default_rng(a.seed), reps=30, reps_fw=3)
else:
    p2, *_ = parte2.executar(d, res, fig, np.random.default_rng(a.seed))

print('Parte III...')
if a.rapido:
    p3 = parte3.executar(d, res, fig, np.random.default_rng(a.seed), n_amostras_sw=5, B=50, T=30)
else:
    p3 = parte3.executar(d, res, fig, np.random.default_rng(a.seed))

macros.gerar(p1, p2, p3, os.path.join(res, 'macros.tex'))
print(f'Pronto em {(time.time() - t0) / 60:.1f} min. Resultados em resultados/ e figuras/.')
