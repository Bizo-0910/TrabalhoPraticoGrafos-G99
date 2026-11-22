"""Medição de tempo real, estatística (média, desvio, IC) e ajuste de escalonamento."""
import gc
import time
import numpy as np
from scipy import stats


def medir(f, lista_args, aquecimento=1):
    """Tempo (s) de cada chamada f(*args). Faz `aquecimento` chamadas descartadas antes e
    desliga o coletor de lixo durante a medição."""
    for a in lista_args[:aquecimento]:
        f(*a)
    ts = []
    gc.collect()
    gc.disable()
    try:
        for a in lista_args:
            t0 = time.perf_counter()
            f(*a)
            ts.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    return np.array(ts)


def ic(x, conf=0.95):
    """Média, desvio padrão amostral e IC. n >= 30: Normal (Z); n < 30 (σ desconhecido): t-Student."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    media, dp = x.mean(), x.std(ddof=1)
    if n >= 30:
        crit, dist = stats.norm.ppf(1 - (1 - conf) / 2), 'Normal (Z)'
    else:
        crit, dist = stats.t.ppf(1 - (1 - conf) / 2, n - 1), 't-Student'
    h = crit * dp / np.sqrt(n)
    return {'media': float(media), 'dp': float(dp), 'lo': float(media - h), 'hi': float(media + h),
            'dist': dist, 'n': n, 'crit': float(crit)}


def inclinacao_loglog(x, y):
    """Inclinação da reta log y = a log x + b. Se o tempo segue f(V,E), com x = f(V,E), a ≈ 1."""
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    return float(a), float(b)
