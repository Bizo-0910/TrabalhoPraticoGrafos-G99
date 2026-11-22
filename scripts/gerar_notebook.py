"""Monta notebooks/analise.ipynb a partir de resultados/ e figuras/ (com as saídas já embutidas)."""
import base64
import io
import json
import os
import sys

import pandas as pd
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(RAIZ, 'resultados'), os.path.join(RAIZ, 'figuras')
J = lambda n: json.load(open(os.path.join(RES, n + '.json'), encoding='utf-8'))
p1, p2, p3 = J('parte1'), J('parte2'), J('parte3')
q1, q2, q3, ds = p3['q1'], p3['q2'], p3['q3'], p3['descobertas']
v = lambda x, c=2: f'{x:.{c}f}'.replace('.', ',')
pc = lambda x, c=1: v(100 * x, c) + '%'

cells = []
cont = [0]


def md(t):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n').splitlines(True)})


def code(src, outputs=()):
    cont[0] += 1
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': cont[0],
                  'outputs': [dict(o, **({'execution_count': cont[0]} if o['output_type'] == 'execute_result' else {})) for o in outputs],
                  'source': src.strip('\n').splitlines(True)})


def out_img(nome, largura=1100):
    im = Image.open(os.path.join(FIG, nome + '.png')).convert('RGB')
    if im.width > largura:
        im = im.resize((largura, int(im.height * largura / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'PNG', optimize=True)
    return {'output_type': 'display_data', 'metadata': {},
            'data': {'image/png': base64.b64encode(buf.getvalue()).decode(), 'text/plain': [f'<figura {nome}>']}}


def out_tab(df):
    return {'output_type': 'display_data', 'metadata': {},
            'data': {'text/html': df.to_html(index=False, border=0, float_format=lambda x: f'{x:.5g}').splitlines(True),
                     'text/plain': df.to_string(index=False).splitlines(True)}}


def figs(*nomes):
    src = 'mostrar(' + ', '.join(repr(n) for n in nomes) + ')'
    code(src, [out_img(n) for n in nomes])


def tabela(nome):
    df = pd.read_csv(os.path.join(RES, nome + '.csv'))
    code(f"tabela({nome!r})", [out_tab(df)])


# ----------------------------------------------------------------------------------------------
md("""
# TP – Analisando Grafos do SNAP
## Grafo G99 · CollegeMsg (matrícula 224215941)

Este notebook **exibe os resultados já calculados** por `scripts/run_all.py` (código em `src/`, testado contra o NetworkX).
Os valores abaixo são da execução com semente 42; os tempos da Parte II dependem do computador.
Para recalcular tudo, rode a última seção.
""")
code("""
import os, json
import pandas as pd
from IPython.display import Image, display, Markdown

RAIZ = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
RES, FIG = os.path.join(RAIZ, 'resultados'), os.path.join(RAIZ, 'figuras')
pd.set_option('display.width', 200)

def mostrar(*nomes):
    for n in nomes:
        display(Image(filename=os.path.join(FIG, n + '.png'), width=900))

def tabela(nome):
    display(pd.read_csv(os.path.join(RES, nome + '.csv')))

def resultados(parte):
    return json.load(open(os.path.join(RES, parte + '.json'), encoding='utf-8'))
""")
md("""
## 0. Determinação do grafo
Matrícula M = 224215941: soma dos dígitos = 30; últimos dois dígitos = 41.

f(M) = (30 × (41 + 1)) mod 129 = 1260 mod 129 = **99** → **G99 = CollegeMsg**.
""")
code("""
M = 224215941
soma = sum(int(c) for c in str(M)); ult = M % 100
f = (soma * (ult + 1)) % 129 or 129
print(f'soma dos dígitos = {soma} | últimos dois = {ult} | f(M) = {f}  ->  G{f}')
""", [{'output_type': 'stream', 'name': 'stdout', 'text': ['soma dos dígitos = 30 | últimos dois = 41 | f(M) = 99  ->  G99\n']}])

md(f"""
## 1. Descrição do grafo original e tratamento
CollegeMsg: mensagens privadas numa rede social da UC-Irvine. Cada linha `SRC DST UNIXTS` é uma mensagem.
O arquivo tem **{p1['n_mensagens']:,} mensagens** entre **{p1['n']:,} usuários** em {v(p1['dias'], 0)} dias.

> A tabela-índice do SNAP lista as colunas *Temporal Edges* e *Static Edges* trocadas em relação à página do dataset.
> Contamos no arquivo: {p1['n_mensagens']:,} arestas temporais e {p1['m_dirigido']:,} arestas estáticas dirigidas.

**Tratamentos realizados**
1. Grafo temporal → estático (união das arestas de todo o período).
2. Auto-loops: {p1['n_autoloops']} encontrados (nenhum a remover).
3. Multigrafo → grafo simples, guardando o número de mensagens de cada par.
4. Versão **dirigida** ({p1['m_dirigido']:,} arestas) e **não dirigida** ({p1['m']:,} arestas; existe aresta {{u,v}} se u→v ou v→u).
5. Peso da aresta não dirigida **w = 1/m** (m = mensagens do par): laço forte = distância curta. Usado só nos algoritmos que precisam de pesos.
6. Distâncias, small-world e robustez usam a **maior componente conexa** ({p1['gc_n']:,} vértices, {p1['gc_m']:,} arestas = {pc(p1['gc_fracao'])} dos vértices).
""")

md("""
## 2. Parte I – Análise estrutural
### 2.1 Tabela de medidas (computadas × não computadas)
""")
tabela('parte1_tabela')
g = p1['graus']
md(f"""
**Comentários.**
- Quase toda a rede está numa só componente: {p1['gc_n']:,} de {p1['n']:,} vértices ({pc(p1['gc_fracao'])}); as outras três componentes são pares isolados.
- Distribuição de graus muito assimétrica: grau médio {v(g['nao_dirigido']['media'])}, mediana {v(g['nao_dirigido']['mediana'], 0)}, máximo {g['nao_dirigido']['max']}. Poucos vértices concentram muitas conexões.
- Densidade {v(p1['densidade'], 5)}: grafo esparso. Reciprocidade {pc(p1['reciprocidade'])}: quase dois terços dos contatos são correspondidos.
- Assortatividade de grau {v(p1['assortatividade'], 3)} (negativa): vértices de grau alto se ligam a vértices de grau baixo, típico de rede com "emissores".
- Distâncias exatas na maior componente: diâmetro {p1['diametro']}, raio {p1['raio']}, caminho médio {v(p1['caminho_medio'], 3)}. Mais da metade dos pares está a 3 passos.
- Clusterização média {v(p1['clust_medio'], 3)}, transitividade {v(p1['transitividade'], 3)} e {p1['triangulos']:,} triângulos.
- k-core máximo {p1['core_max']} ({p1['n_core_max']} vértices): existe um núcleo coeso e uma periferia grande.
""")
figs('distribuicao_graus')
md("""
**Distribuição de graus.** (a) histograma em bins logarítmicos; (b) CCDF em log-log; (c) quem recebe de muitos
nem sempre envia para muitos; (d) a cauda dos graus de saída é mais pesada que a de entrada.
""")
figs('distancias', 'clusterizacao', 'kcore', 'atividade_semanal')
md(f"""
**Comentários.** As distâncias concentram-se em 3 passos (barra mais alta), com cauda curta até 8. A clusterização local
**decresce com o grau**: vértices com poucos contatos têm vizinhos que se conhecem, e os hubs conectam grupos que não se conhecem.
A atividade é muito concentrada no tempo: **{pc(ds['msgs_primeiras_8_semanas'])} das mensagens** ocorrem nas primeiras 8 semanas.
""")
figs('grafo_nucleo', 'grafo_ego_hub')
vn, ve = p1['visual_nucleo'], p1['visual_ego']
md(f"""
**Visualizações (versões reduzidas).** O grafo completo é ilegível, então mostramos (i) o **núcleo** com {vn['n_nucleo']} vértices de core ≥ {vn['limiar_core']},
colorido por comunidade (Louvain, {vn['n_comunidades']} comunidades no núcleo; modularidade global da maior componente {v(vn['modularidade'], 2)}), e
(ii) a **rede ego do maior hub** (vértice {ve['hub_id']}, grau {ve['hub_grau']}): enviou mensagens a {ve['hub_enviou']} contatos que não responderam,
recebeu de {ve['hub_recebeu']} sem responder e trocou mensagens nos dois sentidos com {ve['hub_ambos']}.
""")

# ---------------------------------------------------------------- Parte II
md(f"""
## 3. Parte II – Algoritmos da disciplina
Todos implementados à mão e validados contra o NetworkX (`pytest`). Grafo: maior componente ({p2['n']:,} vértices, {p2['m']:,} arestas).

### 3.1 O que cada algoritmo encontrou
""")
tabela('parte2_resumo')
md(f"""
- **Eulerianidade:** a rede **não é euleriana**: {p2['euler_nd']['graus_impares']:,} vértices têm grau ímpar (um circuito exige 0; um caminho, 2).
  No grafo dirigido, {p2['euler_dir']['desbalanceados']:,} vértices têm grau de entrada ≠ saída.
- **Tarjan:** {p2['tarjan_scc']['n_scc']} componentes fortemente conexas; uma gigante com {p2['tarjan_scc']['maior']:,} vértices e {p2['tarjan_scc']['tamanho_1']} vértices isolados (SCC de tamanho 1).
  Também encontramos {p2['pontes']['n_pontes']} pontes e {p2['pontes']['n_articulacoes']} pontos de articulação (vértices cuja remoção desconecta a rede).
- **Pesos:** w = 1/m é positivo, logo Dijkstra se aplica. Bellman-Ford confirmou a ausência de ciclo negativo e convergiu em {p2['bellman_ford']['passadas']} passadas (pior caso {p2['n'] - 1:,}).
- **Árvore geradora mínima:** Kruskal e Prim deram o mesmo peso ({v(p2['mst']['peso'], 3)}) com {p2['mst']['arestas']:,} arestas.

### 3.2 Tempo real (média, desvio padrão e IC 95%)
Algoritmos de fonte única foram repetidos a partir de 30 fontes aleatórias; os demais, 30 vezes (n ≥ 30 → IC pela **Normal**).
O Floyd-Warshall completo foi repetido {p2['reps_fw']} vezes (n < 30, σ desconhecido → IC pela **t-Student**).
Cada medição descarta uma execução de aquecimento e desliga o coletor de lixo.
""")
tabela('parte2_tempos')
figs('tempos_algoritmos')
tt = pd.read_csv(os.path.join(RES, 'parte2_tempos.csv')).set_index('algoritmo')
razao = tt.loc['Floyd-Warshall (n=150, python)', 'media'] / tt.loc['Floyd-Warshall (n=150, numpy)', 'media']
md(f"""
**Comentários.** Os algoritmos lineares (BFS, DFS, Tarjan, Eulerianidade) levam alguns milissegundos; Dijkstra, Kruskal, Prim e Bellman-Ford,
algumas dezenas de milissegundos; o **Floyd-Warshall é de outra ordem de grandeza** (≈ {v(tt.loc['Floyd-Warshall', 'media'], 0)} s para {p2['n']:,} vértices), como o V³ prevê.
Com n = 150, a versão vetorizada em numpy é ≈ {v(razao, 0)} vezes mais rápida que o Python puro, mantendo o mesmo número de operações.

### 3.3 Complexidade teórica × observada
Medimos cada algoritmo em subgrafos conexos de tamanhos crescentes e ajustamos a reta log(tempo) × log f(V,E).
Se o tempo seguir a teoria, a inclinação é ≈ 1.
""")
tabela('parte2_inclinacoes')
figs('complexidade_observada')
md("""
**Comentários.** Tarjan (pontes), Prim e Floyd-Warshall ficam muito perto de 1. BFS, DFS, Dijkstra e Kruskal ficam um pouco abaixo (≈ 0,8 a 0,95):
em Python há custos fixos por chamada que pesam mais nos grafos pequenos. **Bellman-Ford (≈ 0,65)** fica bem abaixo porque V·E é o *pior caso*:
com parada antecipada ele converge em poucas passadas (próximo ao diâmetro), então o tempo cresce quase como E.
O pior caso só apareceria em grafos com caminhos mínimos muito longos.
""")

# ---------------------------------------------------------------- Parte III
a = q2['ajuste']; cmp_ = q2['comparacao']; ra = q3['resumo_aleatorio']; at = q3['ataques']; cv = q3['curvas']
idx = lambda f: cv['fracs'].index(f)
md(f"""
## 4. Parte III – Análise estrutural
### 4.1 O grafo apresenta indícios de small-world?
Comparamos o grafo (maior componente) com dois grafos aleatórios de mesmo n e m, **implementados à mão**:
**Erdős-Rényi G(n,m)** e um **modelo nulo com a mesma sequência de graus** (reconexão de arestas). Em cada caso, {q1['n_amostras']} amostras;
o caminho médio dos nulos foi estimado com BFS de {q1['k_fontes_nulos']} fontes. Índice σ = (C/C_rand)/(L/L_rand).
""")
tabela('parte3_smallworld')
figs('smallworld')
md(f"""
**Resposta.**
- **Contra G(n,m): sim, com folga.** L = {v(q1['L'], 3)} é praticamente igual a L_ER = {v(q1['L_er'], 3)} (razão {v(q1['razao_L_er'], 3)}), enquanto C = {v(q1['C'], 3)} é **{v(q1['razao_C_er'], 1)} vezes** maior que C_ER = {v(q1['C_er'], 4)}. Assim, σ ≈ {v(q1['sigma_er'], 1)} ≫ 1 e a definição do Apêndice A é satisfeita.
- **Contra a mesma sequência de graus: apenas ligeiramente.** C é só {v(q1['razao_C_grau'], 2)} vez o do nulo (σ ≈ {v(q1['sigma_grau'], 2)}). **Quase toda a clusterização vem da heterogeneidade dos graus** (hubs), e não de comunidades fechadas.
- **Faz sentido?** Sim, para uma rede de mensagens: poucas pessoas muito ativas encurtam as distâncias e criam grupos em torno delas.
- **Implicação prática:** qualquer usuário alcança outro em ≈ 3 passos (diâmetro {p1['diametro']}); informação, convites ou spam se espalham muito rápido, e os hubs são decisivos nessa difusão.
""")

md(f"""
### 4.2 A distribuição de graus sugere uma lei de potência?
Ajuste discreto por **máxima verossimilhança** (não por regressão no log-log, que é viesada), k_min escolhido pelo menor KS,
p-valor por bootstrap ({p3['q2']['bootstrap_B']} amostras sintéticas) e razão de verossimilhança de Vuong contra exponencial e lognormal.
""")
figs('lei_de_potencia')
sens = pd.DataFrame(q2['sensibilidade_kmin'])
code("pd.DataFrame(resultados('parte3')['q2']['sensibilidade_kmin'])", [out_tab(sens)])
md(f"""
**Resposta: não há evidência de lei de potência; a cauda é pesada, mas não livre de escala.**
- Melhor ajuste: γ = {v(a['gamma'], 2)} ± {v(a['erro_gamma'], 2)} com k_min = {a['kmin']}, ou seja, só {a['n_cauda']} vértices ({pc(a['n_cauda'] / q2['n'])}) estão na cauda.
- **Qualidade do ajuste: p = {v(q2['p_valor'], 2)} < 0,10** → a lei de potência é rejeitada.
- Contra a **lognormal**: R = {v(cmp_['lognormal']['R'], 2)} (p = {v(cmp_['lognormal']['p'], 2)}): a lognormal ajusta melhor, com significância apenas marginal. Contra a **exponencial**: R = {v(cmp_['exponencial']['R'], 2)} (p = {v(cmp_['exponencial']['p'], 2)}): indistinguíveis na cauda curta.
- **Instabilidade:** γ cresce de {v(sens['gamma'].iloc[0], 2)} a {v(sens['gamma'].iloc[-1], 2)} quando k_min vai de 1 a 30. Não há uma região do gráfico com declive constante.
- A reta log-log por mínimos quadrados dá γ = {v(q2['ols']['gamma_ols'], 2)}, valor enganoso: a CCDF não é linear e o MQO é dominado pelos pontos ruidosos da cauda.
- Existem **hubs** (grau máximo {p1['graus']['nao_dirigido']['max']} contra média {v(p1['graus']['nao_dirigido']['media'], 1)}), mas a distribuição se curva para baixo (cauda "encolhida").
""")

md(f"""
### 4.3 Robustez
Política: trabalhar na **maior componente conexa** (n = {q3['n']:,}); r = ⌈0,05 n⌉ = **{q3['r']}** vértices removidos.
Métricas: **A** tamanho da maior componente (fração de n; A_rel divide por n − r), **B** número de componentes, **C** caminho médio na maior componente (exato), **D** fração de isolados.
Remoção aleatória: T = {q3['T']} repetições. Ataque: ranking fixo por **grau**, **intermediação** (Brandes) e **PageRank**.
""")
figs('robustez_boxplot')
tabela('parte3_robustez_aleatoria')
tabela('parte3_robustez_ataques')
figs('robustez_curvas')
md(f"""
**Resposta.**
- **Falhas aleatórias de 5%: a rede é robusta.** A = {v(ra['A']['media'], 3)} ± {v(ra['A']['dp'], 3)} (o máximo possível seria 0,95); entre os vértices que restam, {pc(ra['A_rel']['media'])} continuam na maior componente.
  B = {v(ra['B']['media'], 1)} ± {v(ra['B']['dp'], 1)} componentes, C = {v(ra['C']['media'], 3)} (era {v(q3['base']['C'], 3)}) e apenas {pc(ra['D']['media'])} de isolados.
- **Ataque aos 5% mais centrais: a rede é vulnerável.** Por intermediação, A cai para {v(at['intermediação']['A'], 3)} (B = {at['intermediação']['B']}, {pc(at['intermediação']['D'])} isolados); por grau, A = {v(at['grau']['A'], 3)} (B = {at['grau']['B']}); por PageRank, A = {v(at['PageRank']['A'], 3)} (B = {at['PageRank']['B']}).
  O caminho médio sobe para {v(at['grau']['C'], 2)}. Como S_rand ≫ S_cent e c_cent ≫ c_rand, o padrão é o de redes com hubs.
- **Curvas:** removendo 20% dos vértices, a maior componente retém {pc(cv['aleatório']['A'][idx(0.2)], 0)} (aleatório) contra {pc(cv['grau']['A'][idx(0.2)], 0)} (grau); com 30%, {pc(cv['aleatório']['A'][idx(0.3)], 0)} contra {pc(cv['grau']['A'][idx(0.3)], 0)} (grau) e {pc(cv['intermediação']['A'][idx(0.3)], 0)} (intermediação).
- Os rankings concordam bastante: {pc(q3['sobreposicao_top']['grau∩intermediação'], 0)} dos 5% de maior grau também estão entre os 5% de maior intermediação.
- Nas curvas, C é estimado com 200 fontes e fica instável quando a rede se fragmenta (> 20%).
""")

md(f"""
### 4.4 A descoberta mais interessante: os "hubs" não são conversadores, são emissores
Os vértices de grau alto não se comportam como centros sociais, e sim como emissores em massa:
- Os 20 usuários que mais enviaram escreveram a ≈ {v(ds['top20_saida_media'], 0)} pessoas em média, mas só {pc(ds['top20_taxa_resposta'], 0)} dos destinatários responderam (contra {pc(ds['taxa_resposta_geral'], 0)} na rede toda).
- A reciprocidade cai com o grau: ≈ 74–79% nas faixas de grau 2 a 31, e **{pc(ds['reciprocidade_por_faixa']['128-299'], 0)} nos vértices de grau ≥ 128**.
- Os 5% que mais enviam produzem **{pc(ds['top5pct_msgs'], 0)} das mensagens**; {pc(ds['pares_1_msg'], 0)} dos pares trocam uma única mensagem, e os {pc(ds['pares_10mais_msgs'], 0)} de pares com ≥ 10 mensagens concentram {pc(ds['msgs_em_pares_10mais'], 0)} do tráfego.
- Quando há resposta, ela é rápida: a mediana entre a 1ª mensagem de cada sentido é de {v(ds['mediana_horas_entre_primeiras'], 1)} h ({str(ds['pares_reciprocos'])[:-3]}.{str(ds['pares_reciprocos'])[-3:]} pares recíprocos).
- Temporalmente, {pc(ds['msgs_primeiras_8_semanas'], 0)} das mensagens ocorrem nas 8 primeiras semanas (pico e depois cauda).

**Ligação com os resultados anteriores:** são justamente esses emissores que conectam a rede. Removê-los fragmenta o grafo (seção 4.3), e eles explicam a clusterização "herdada" dos graus (seção 4.1).
""")
figs('descobertas')

et = pd.DataFrame(p3['extra']['tabela'])
md("""
### 4.5 (Extra) O grafo se aproxima de algum modelo clássico?
Comparamos com ER, Barabási-Albert (k = {k_ba}), Watts-Strogatz (k = {k_ws}, p = 0,1) e o nulo de mesmo grau. Cada modelo: {na} amostras, mesmo n e m (≈) da maior componente.
""".format(k_ba=p3['extra']['parametros']['k_BA'], k_ws=p3['extra']['parametros']['k_WS'], na=p3['extra']['parametros']['n_amostras']))
tabela('parte3_modelos')
figs('modelos_classicos')
r_ = et.set_index('modelo')
md(f"""
**Resposta: nenhum modelo clássico reproduz a rede.**
- **ER:** acerta L, mas C ({v(r_.loc['ER', 'C'], 4)}) é ≈ {v(r_.loc['observado', 'C'] / r_.loc['ER', 'C'], 0)}× menor e não há hubs (grau máx. ≈ {v(r_.loc['ER', 'grau_max'], 0)} contra {v(r_.loc['observado', 'grau_max'], 0)}).
- **Barabási-Albert:** reproduz hubs (≈ {v(r_.loc['BA', 'grau_max'], 0)}) e caminhos curtos ({v(r_.loc['BA', 'L'], 2)}), mas C = {v(r_.loc['BA', 'C'], 3)} é ≈ {v(r_.loc['observado', 'C'] / r_.loc['BA', 'C'], 1)}× menor e o grau mínimo é k = {p3['extra']['parametros']['k_BA']} (a rede real tem muitos vértices de grau 1 a 5).
- **Watts-Strogatz:** C = {v(r_.loc['WS', 'C'], 2)} é alto demais, L = {v(r_.loc['WS', 'L'], 2)} é longo demais e não há hubs.
- **Mesma sequência de graus:** é o que mais se aproxima (C = {v(r_.loc['mesmo grau', 'C'], 3)} vs {v(r_.loc['observado', 'C'], 3)}; L = {v(r_.loc['mesmo grau', 'L'], 2)} vs {v(r_.loc['observado', 'L'], 2)}), mostrando que **a heterogeneidade dos graus explica quase toda a estrutura**; falta só um pouco de clusterização local.
""")

md("""
## 5. Discussão crítica e limitações
- **Tempo ignorado.** O grafo estático une 193 dias; dois usuários podem estar ligados por um caminho que nunca existiu em ordem temporal. Distâncias e difusão reais seriam maiores. Além disso, ≈ 80% das mensagens ocorrem em 8 semanas, então a rede muda muito.
- **Direção.** Distâncias, clusterização e robustez usam a versão não dirigida. Com direção (Tarjan), só 1.294 vértices formam uma componente fortemente conexa.
- **Pesos.** w = 1/m é uma escolha nossa; outra definição (por exemplo, intervalo entre mensagens) mudaria Dijkstra, Floyd-Warshall e a árvore geradora, mas não os resultados sem pesos.
- **Estimativas.** O caminho médio dos nulos (small-world) e das curvas de robustez usa amostras de fontes; os valores do grafo observado são exatos.
- **Robustez com ranking fixo.** O ataque usa a centralidade calculada uma vez, como pede o enunciado. Um ataque **adaptativo** (recalculando a cada remoção) seria ainda mais destrutivo.
- **Lei de potência.** Com só 214 vértices na cauda, testes de cauda têm baixa potência; por isso reportamos também o ajuste por faixa de k_min.

## 6. Conclusão
O G99 é uma rede esparsa e quase toda conexa, com caminhos curtos (L ≈ 3, diâmetro 8) e alguns hubs muito ativos.
É small-world em relação a um grafo aleatório, mas essa propriedade vem sobretudo da heterogeneidade dos graus. A distribuição de graus tem cauda pesada **sem ser uma lei de potência**.
A rede é robusta a falhas aleatórias e **frágil a ataques aos hubs**, que são emissores em massa (baixa reciprocidade). Nenhum modelo clássico a descreve; o modelo de mesmo grau é o mais próximo.
""")

md("""
## 7. Recalcular tudo (opcional)
Roda o pipeline completo (~7 min). Execute em uma máquina sem outras tarefas pesadas, porque a Parte II mede tempo.
""")
code("""
# remova o '#' da linha abaixo para recalcular (grava em resultados/ e figuras/)
# %run ../scripts/run_all.py
""")

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
os.makedirs(os.path.join(RAIZ, 'notebooks'), exist_ok=True)
with open(os.path.join(RAIZ, 'notebooks', 'analise.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('notebook gerado:', len(cells), 'células')
