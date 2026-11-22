"""Lei de potência para o grau (Parte III, questão 2), seguindo Clauset, Shalizi & Newman (2009).

Modelo discreto  P(k) = k^(-gamma) / zeta(gamma, kmin)  para k >= kmin.
 - gamma por máxima verossimilhança (não por regressão no log-log, que é viesada);
 - kmin escolhido minimizando a distância de Kolmogorov-Smirnov (KS);
 - qualidade do ajuste por bootstrap paramétrico (p-valor);
 - comparação com alternativas (exponencial e lognormal discretas) pelo teste de razão de
   verossimilhança normalizado de Vuong.
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import zeta
from scipy.stats import norm


# ---------- ajuste da lei de potência ----------
def _gamma_mle(soma_ln, n, kmin):
    """gamma que maximiza  -n ln zeta(g, kmin) - g * soma_ln."""
    f = lambda g: n * np.log(zeta(g, kmin)) + g * soma_ln
    r = minimize_scalar(f, bounds=(1.01, 8.0), method='bounded', options={'xatol': 1e-7})
    return float(r.x)


def _ks(cauda_ordenada, gamma, kmin):
    """Distância KS entre a CDF empírica da cauda e a do modelo."""
    ks, cont = np.unique(cauda_ordenada, return_counts=True)
    emp = np.cumsum(cont) / cont.sum()
    mod = 1.0 - zeta(gamma, ks + 1) / zeta(gamma, kmin)
    # CDF empírica logo antes do degrau
    emp0 = np.concatenate(([0.0], emp[:-1]))
    mod0 = 1.0 - zeta(gamma, ks) / zeta(gamma, kmin)
    return float(max(np.abs(emp - mod).max(), np.abs(emp0 - mod0).max()))


def ajustar(x, kmin=None, min_cauda=30):
    """Ajusta a lei de potência discreta. Se kmin é None, escolhe pelo menor KS.
    Devolve dict com gamma, kmin, ks, n_cauda."""
    x = np.sort(np.asarray(x, dtype=int))
    x = x[x >= 1]
    candidatos = [kmin] if kmin is not None else [int(k) for k in np.unique(x)]
    melhor = None
    for k0 in candidatos:
        cauda = x[x >= k0]
        if len(cauda) < min_cauda and kmin is None:
            break
        g = _gamma_mle(np.log(cauda).sum(), len(cauda), k0)
        ks = _ks(cauda, g, k0)
        if melhor is None or ks < melhor['ks']:
            melhor = {'gamma': g, 'kmin': int(k0), 'ks': ks, 'n_cauda': int(len(cauda))}
    melhor['erro_gamma'] = (melhor['gamma'] - 1) / np.sqrt(melhor['n_cauda'])   # erro-padrão aproximado
    return melhor


def pmf_pl(k, gamma, kmin):
    return np.asarray(k, dtype=float) ** (-gamma) / zeta(gamma, kmin)


def amostrar_pl(n, gamma, kmin, rng, kmax=10 ** 6):
    """Amostra discreta de P(k) ~ k^-gamma, k >= kmin (inversão da CDF truncada em kmax)."""
    ks = np.arange(kmin, kmax + 1)
    cdf = np.cumsum(pmf_pl(ks, gamma, kmin))
    cdf /= cdf[-1]
    return ks[np.searchsorted(cdf, rng.random(n))]


# ---------- bootstrap de qualidade do ajuste ----------
def gof_bootstrap(x, ajuste, B, rng):
    """p-valor: fração de amostras sintéticas (cauda ~ lei de potência ajustada, corpo ~ empírico)
    cujo KS (reajustado) é pelo menos o observado. p > 0,1 não rejeita a lei de potência."""
    x = np.asarray(x, dtype=int)
    n = len(x)
    corpo = x[x < ajuste['kmin']]
    p_cauda = ajuste['n_cauda'] / n
    ks_sint = []
    for _ in range(B):
        n_c = rng.binomial(n, p_cauda)
        parte_cauda = amostrar_pl(n_c, ajuste['gamma'], ajuste['kmin'], rng)
        parte_corpo = rng.choice(corpo, size=n - n_c) if len(corpo) else np.array([], dtype=int)
        sint = np.concatenate([parte_cauda, parte_corpo])
        ks_sint.append(ajustar(sint)['ks'])
    ks_sint = np.array(ks_sint)
    return float(np.mean(ks_sint >= ajuste['ks'])), ks_sint


# ---------- distribuições alternativas (mesma cauda k >= kmin) ----------
def _loglik_exp(cauda, kmin):
    lam = np.log(1 + len(cauda) / (cauda - kmin).sum())
    ll = np.log(1 - np.exp(-lam)) - lam * (cauda - kmin)
    return ll, {'lambda': float(lam)}


def _loglik_lognormal(cauda, kmin, kmax):
    ks = np.arange(kmin, kmax + 1, dtype=float)
    lnk = np.log(ks)
    lc = np.log(cauda)

    def nll(par):
        mu, ls = par
        s = np.exp(ls)
        lg = -(lnk - mu) ** 2 / (2 * s ** 2) - lnk
        logZ = np.log(np.exp(lg - lg.max()).sum()) + lg.max()
        return -(np.sum(-(lc - mu) ** 2 / (2 * s ** 2) - lc) - len(cauda) * logZ)
    r = minimize(nll, x0=[np.log(kmin + 1), 0.0], method='Nelder-Mead',
                 options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 2000})
    mu, s = r.x[0], np.exp(r.x[1])
    lg = -(lnk - mu) ** 2 / (2 * s ** 2) - lnk
    logZ = np.log(np.exp(lg - lg.max()).sum()) + lg.max()
    ll = -(lc - mu) ** 2 / (2 * s ** 2) - lc - logZ
    return ll, {'mu': float(mu), 'sigma': float(s)}


def comparar(x, ajuste):
    """Razão de verossimilhança (lei de potência vs alternativa) com teste de Vuong.
    R > 0 favorece a lei de potência. p: probabilidade de ver |R| tão grande se as duas fossem equivalentes."""
    x = np.asarray(x, dtype=int)
    kmin = ajuste['kmin']
    cauda = x[x >= kmin].astype(float)
    ll_pl = -ajuste['gamma'] * np.log(cauda) - np.log(zeta(ajuste['gamma'], kmin))
    out = {}
    kmax = int(cauda.max() * 5)
    for nome, (ll_alt, par) in {'exponencial': _loglik_exp(cauda, kmin),
                                'lognormal': _loglik_lognormal(cauda, kmin, kmax)}.items():
        dif = ll_pl - ll_alt
        R = float(dif.sum())
        dp = float(dif.std(ddof=0))
        z = R / (np.sqrt(len(dif)) * dp) if dp > 0 else 0.0
        out[nome] = {'R': R, 'z': float(z), 'p': float(2 * norm.sf(abs(z))), 'parametros': par,
                     'loglik_alt': float(ll_alt.sum()), 'loglik_pl': float(ll_pl.sum())}
    return out


def ajuste_ols_loglog(x):
    """Regressão linear de log P(k) em log k (o 'jeito ingênuo' do Apêndice B), só para comparação."""
    ks, cont = np.unique(np.asarray(x, dtype=int), return_counts=True)
    p = cont / cont.sum()
    a, b = np.polyfit(np.log(ks), np.log(p), 1)
    return {'gamma_ols': float(-a), 'log_C': float(b)}
