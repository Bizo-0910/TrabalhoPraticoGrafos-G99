"""Gera as tabelas LaTeX do relatório (relatorio/tabelas/*.tex) a partir de resultados/."""
import csv, json, os
R = os.path.join(os.path.dirname(__file__), '..')
res = lambda f: os.path.join(R, 'resultados', f)
out = lambda f: os.path.join(R, 'relatorio', 'tabelas', f)

def num(x, d=3):
    s = f'{x:.{d}f}'
    return s.replace('.', ',')

def milhar(n):
    return f'{int(n):,}'.replace(',', '.')

def cx(c):
    return '$' + c.replace('·', '\\cdot ').replace(' log ', ' \\log ').replace('V³', 'V^3').replace('E log', 'E \\log') + '$'

def tempo(s):
    if s >= 1: return f'{num(s, 2)} s'
    return f'{num(s * 1000, 2)} ms'

def tabela(nome, legenda, rotulo, cols, linhas, largura='col', cab=None):
    env = 'table*' if largura == 'larga' else 'table'
    with open(out(nome), 'w', encoding='utf-8') as f:
        f.write(f'\\begin{{{env}}}\n  \\caption{{{legenda}}}\n  \\label{{{rotulo}}}\n  \\small\n')
        f.write(f'  \\begin{{tabular}}{{{cols}}}\n    \\toprule\n    {cab} \\\\\n    \\midrule\n')
        for l in linhas:
            f.write('    ' + ' & '.join(l) + ' \\\\\n')
        f.write(f'    \\bottomrule\n  \\end{{tabular}}\n\\end{{{env}}}\n')

# 1. tratamento
rows = list(csv.DictReader(open(res('parte0_etapas.csv'), encoding='utf-8')))
tabela('tratamento.tex', 'Etapas do tratamento: tamanho do grafo após cada etapa.', 'tab:tratamento',
       'p{4.2cm}rr', [[r['Etapa'].replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', ''),
                milhar(r['Vértices']), milhar(r['Arestas'])] for r in rows],
       cab='Etapa & Vértices & Arestas')

# 3. tempos
rows = list(csv.DictReader(open(res('parte2_tempos.csv'), encoding='utf-8')))
tabela('tempos.tex', 'Tempo de execução (média, desvio-padrão e IC de 95\\%) na maior componente '
       '($n=1893$, $m=13835$) e complexidade teórica. IC pela Normal quando $n\\ge30$ e pela t-Student quando $n<30$.',
       'tab:tempos', 'llrrrl',
       [[r['algoritmo'].replace('n=150','$n=150$'), cx(r['complexidade']),
         r['repeticoes'], tempo(float(r['media'])), tempo(float(r['dp'])),
         f"[{tempo(float(r['ic_lo']))}; {tempo(float(r['ic_hi']))}]"] for r in rows],
       largura='larga', cab='Algoritmo & Complexidade & Rep. & Média & DP & IC 95\\%')

# 4. inclinações
rows = list(csv.DictReader(open(res('parte2_inclinacoes.csv'), encoding='utf-8')))
tabela('inclinacoes.tex', 'Inclinação do ajuste log-log do tempo observado em função da expressão teórica (por exemplo, $V+E$). Valor 1 indica que o tempo cresce como a fórmula prevê.',
       'tab:inclinacoes', 'llr',
       [[r['algoritmo'].replace('n=150','$n=150$'), cx(r['complexidade_teorica']),
         num(float(r['inclinacao_loglog']), 2)] for r in rows],
       cab='Algoritmo & Teórica & Inclinação')

# 5. small-world
q = json.load(open(res('parte3.json')))['q1']
tabela('smallworld.tex', 'Teste de mundo pequeno na maior componente. Médias de 20 amostras de cada modelo nulo '
       '(distâncias estimadas com 500 fontes).', 'tab:smallworld', 'lrrrr',
       [['Observado', num(q['C'], 4), num(q['L'], 3), '--', '--'],
        ['Aleatório $G(n,m)$', num(q['C_er'], 4), num(q['L_er'], 3), num(q['razao_C_er'], 1), num(q['razao_L_er'], 3)],
        ['Mesmos graus', num(q['C_grau'], 4), num(q['L_grau'], 3), num(q['razao_C_grau'], 2), num(q['razao_L_grau'], 3)]],
       cab='Grafo & $C$ & $L$ & $C/C_{al}$ & $L/L_{al}$')

# 7. modelos
rows = list(csv.DictReader(open(res('parte3_modelos.csv'), encoding='utf-8')))
nome = {'observado': 'Observado', 'ER': "Erd\\H{o}s--R\\'enyi", 'BA': "Barab\\'asi--Albert",
        'WS': 'Watts--Strogatz', 'mesmo grau': 'Mesmos graus'}
tabela('modelos.tex', 'Grafo observado e modelos clássicos de mesma ordem e tamanho aproximado '
       '(médias de réplicas). KS: distância de Kolmogorov--Smirnov entre as distribuições de grau.',
       'tab:modelos', 'lrrrrr',
       [[nome[r['modelo']], num(float(r['C']), 3), num(float(r['L']), 2), num(float(r['grau_max']), 0),
         num(float(r['grau_dp']), 1), num(float(r['KS_grau']), 2)] for r in rows],
       cab='Modelo & $C$ & $L$ & $k_{\\max}$ & DP grau & KS')

# robustez
at = list(csv.DictReader(open(res('parte3_robustez_ataques.csv'), encoding='utf-8')))
al = {r['metrica']: r for r in csv.DictReader(open(res('parte3_robustez_aleatoria.csv'), encoding='utf-8'))}
lin = [['Íntegro', '1,000', '1', num(at[0]['C'] and float(at[0]['C']), 3), '0,0\\%']]
lin.append(['Aleatória ($\\pm$ DP)', f"{num(float(al['A']['media']),3)} $\\pm$ {num(float(al['A']['dp']),3)}",
            f"{num(float(al['B']['media']),1)} $\\pm$ {num(float(al['B']['dp']),1)}",
            f"{num(float(al['C']['media']),3)} $\\pm$ {num(float(al['C']['dp']),3)}",
            f"{num(100*float(al['D']['media']),1)}\\%"])
for r in at[1:]:
    lin.append([r['estrategia'].replace('ataque ', 'Ataque: ').replace('intermediação', 'intermediação'),
                num(float(r['A']), 3), r['B'], num(float(r['C']), 3), num(100 * float(r['D']), 1) + '\\%'])
tabela('robustez.tex', 'Robustez após remover $r=95$ vértices (5\\% de $n$). A: fração na maior componente; '
       'B: nº de componentes; C: caminho médio na maior componente; D: fração de vértices isolados.',
       'tab:robustez', 'lrrrr', lin, largura='larga', cab='Remoção & A & B & C & D')
print('ok')
