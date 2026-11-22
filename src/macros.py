"""Gera resultados/macros.tex: os números do relatório (LaTeX) vêm direto dos resultados."""
from .latex import macro


def _n(x, casas=2):
    return f'{x:.{casas}f}'.replace('.', ',')


def _i(x):
    return f'{int(x):,}'.replace(',', '.')


def gerar(p1, p2, p3, caminho):
    q1, q2, q3, desc = p3['q1'], p3['q2'], p3['q3'], p3['descobertas']
    aj = q2['ajuste']
    L = []
    add = lambda nome, v: L.append(macro(nome, v))
    add('nNos', _i(p1['n'])); add('nMensagens', _i(p1['n_mensagens']))
    add('nArestasDir', _i(p1['m_dirigido'])); add('nArestasND', _i(p1['m']))
    add('nDias', _n(p1['dias'], 1))
    g = p1['graus']['nao_dirigido']
    add('grauMin', str(g['min'])); add('grauMax', str(g['max'])); add('grauMedio', _n(g['media']))
    add('grauMediana', _n(g['mediana'], 0))
    add('densidade', _n(p1['densidade'], 5)); add('reciprocidade', _n(100 * p1['reciprocidade'], 1) + r'\%')
    add('nComponentes', str(p1['n_componentes'])); add('tamMaiorComp', _i(p1['gc_n']))
    add('diametro', str(p1['diametro'])); add('raio', str(p1['raio'])); add('caminhoMedio', _n(p1['caminho_medio'], 3))
    add('clustMedio', _n(p1['clust_medio'], 3)); add('transitividade', _n(p1['transitividade'], 3))
    add('nTriangulos', _i(p1['triangulos'])); add('coreMax', str(p1['core_max']))
    add('assortatividade', _n(p1['assortatividade'], 3))
    add('sigmaER', _n(q1['sigma_er'], 1)); add('sigmaGrau', _n(q1['sigma_grau'], 2))
    add('razaoCER', _n(q1['razao_C_er'], 1)); add('razaoLER', _n(q1['razao_L_er'], 3))
    add('razaoCGrau', _n(q1['razao_C_grau'], 2)); add('razaoLGrau', _n(q1['razao_L_grau'], 3))
    add('gammaPL', _n(aj['gamma'], 2)); add('kminPL', str(aj['kmin'])); add('nCaudaPL', str(aj['n_cauda']))
    add('pvalorPL', _n(q2['p_valor'], 2)); add('gammaOLS', _n(q2['ols']['gamma_ols'], 2))
    add('RLognormal', _n(q2['comparacao']['lognormal']['R'], 1)); add('pLognormal', _n(q2['comparacao']['lognormal']['p'], 2))
    add('RExponencial', _n(q2['comparacao']['exponencial']['R'], 1)); add('pExponencial', _n(q2['comparacao']['exponencial']['p'], 2))
    add('nRemovidos', str(q3['r']))
    ra = q3['resumo_aleatorio']
    add('AAleatorio', _n(ra['A']['media'], 3)); add('ARelAleatorio', _n(ra['A_rel']['media'], 3))
    add('BAleatorio', _n(ra['B']['media'], 1)); add('CAleatorio', _n(ra['C']['media'], 3)); add('DAleatorio', _n(100 * ra['D']['media'], 1) + r'\%')
    for k, nome in [('grau', 'Grau'), ('intermediação', 'Interm'), ('PageRank', 'PR')]:
        a = q3['ataques'][k]
        add('AAtaque' + nome, _n(a['A'], 3)); add('BAtaque' + nome, str(a['B']))
        add('CAtaque' + nome, _n(a['C'], 3)); add('DAtaque' + nome, _n(100 * a['D'], 1) + r'\%')
    add('nPontes', str(p2['pontes']['n_pontes'])); add('nArticulacoes', str(p2['pontes']['n_articulacoes']))
    add('nSCC', str(p2['tarjan_scc']['n_scc'])); add('maiorSCC', str(p2['tarjan_scc']['maior']))
    add('grausImpares', str(p2['euler_nd']['graus_impares'])); add('pesoMST', _n(p2['mst']['peso'], 2))
    add('topCincoMsgs', _n(100 * desc['top5pct_msgs'], 1) + r'\%')
    add('primeirasOitoSem', _n(100 * desc['msgs_primeiras_8_semanas'], 1) + r'\%')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('% gerado automaticamente por scripts/run_all.py\n' + '\n'.join(L) + '\n')
