# TP – Analisando Grafos do SNAP · Grafo G99 (CollegeMsg)

Trabalho prático de Teoria dos Grafos (UFBA / IC). Grafo atribuído pela matrícula **224215941**:

```
soma dos dígitos = 2+2+4+2+1+5+9+4+1 = 30        últimos dois dígitos = 41
f(M) = (30 × (41 + 1)) mod 129 = 1260 mod 129 = 99   →   G99 = CollegeMsg
```

**CollegeMsg**: mensagens privadas numa rede social da UC-Irvine (Panzarasa, Opsahl e Carley, 2009).
Arquivo `SRC DST UNIXTS`, uma linha por mensagem: 1.899 usuários, 59.835 mensagens, 193 dias.
Fonte: <https://snap.stanford.edu/data/CollegeMsg.html>.

> Atenção: na tabela-índice do SNAP as colunas "Temporal Edges" e "Static Edges" desse dataset estão
> **invertidas** em relação à página do próprio dataset. Contamos direto no arquivo:
> **59.835 arestas temporais** (linhas) e **20.296 arestas estáticas dirigidas** (pares únicos).

## Como executar

```bash
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py            
jupyter notebook notebooks/analise.ipynb
pytest -q                            # 28 testes: cada algoritmo é comparado com o NetworkX
```

O `data/CollegeMsg.txt` já está no repositório (original do SNAP, sem alterações).
O notebook **exibe** os resultados já calculados e tem uma célula opcional para recalcular tudo.
A semente aleatória é fixa (42), exceto os tempos da Parte II, que dependem do computador.

## Estrutura

```
data/CollegeMsg.txt            dataset original
data/processed/                arestas tratadas (dirigidas e não dirigidas, com nº de mensagens e peso)
src/tratamento.py              leitura, grafo temporal → estático, pesos, maior componente
src/parte0.py                  tratamento: contagens e etapas (Tabela 1 do relatório)
src/estrutura.py               Parte I: graus, distâncias exatas, clusterização, triângulos, k-core
src/algoritmos.py              Parte II: BFS, DFS, Euler, Dijkstra, Bellman-Ford, Floyd-Warshall (python e numpy),
                               Tarjan (SCC, pontes, articulações), Kruskal, Prim
src/centralidade.py            Brandes (intermediação) e PageRank
src/modelos.py                 ER G(n,m), Barabási-Albert, Watts-Strogatz, reconexão que preserva graus
src/smallworld.py              Parte III-1: C, L e índice σ contra dois modelos nulos
src/powerlaw.py                Parte III-2: MLE discreto, k_min por KS, bootstrap, razão de verossimilhança
src/robustez.py                Parte III-3: remoção aleatória e ataques; métricas A, B, C, D
src/benchmark.py               tempo real, média, desvio padrão, IC (Z se n ≥ 30; t-Student se n < 30)
src/parte1.py parte2.py parte3.py   orquestram cada parte e gravam tabelas/figuras
src/figuras.py latex.py macros.py   figuras, tabelas LaTeX e macros para o relatório
scripts/run_all.py             executa tudo
scripts/gerar_notebook.py      recria notebooks/analise.ipynb
scripts/gerar_tabelas_tex.py   gera as tabelas LaTeX a partir de resultados/
scripts/figuras_relatorio.py   gera as figuras compactas do relatório
notebooks/analise.ipynb        notebook com os resultados
tests/                         pytest (cada algoritmo comparado com o NetworkX)
resultados/                    CSV, JSON e macros.tex (números do relatório)
figuras/                       PNG e PDF (versões completas, do pipeline)
relatorio/                     main.tex, sample-base.bib e 224215941.pdf (relatório final)
relatorio/figuras/             figuras em PDF usadas no relatório
```

Todos os algoritmos e métricas foram **implementados à mão**. O NetworkX é usado apenas nos testes
(validação) e para o layout/coloração das visualizações.

## Tratamento dos dados

1. Grafo temporal → estático: união de todas as arestas dos 193 dias.
2. Auto-loops: o arquivo não tem nenhum (0 removidos).
3. Multigrafo → grafo simples, guardando quantas mensagens cada par trocou.
4. Versão dirigida (20.296 arestas) e não dirigida (13.838 arestas; 63,6 % dos arcos são recíprocos).
5. Peso das arestas não dirigidas **w = 1/m** (m = mensagens do par): laço forte = distância curta.
6. Distâncias, small-world e robustez usam a **maior componente conexa** (1.893 vértices, 13.835 arestas).

## Decisões de método (resumo)

| Pergunta | Método |
|---|---|
| Small-world | C e L exatos; nulos: G(n,m) e mesma sequência de graus (reconexão); σ = (C/C_rand)/(L/L_rand) |
| Lei de potência | MLE discreto + k_min por KS + bootstrap (p-valor) + razão de verossimilhança (Vuong) contra exponencial e lognormal |
| Robustez | r = ⌈0,05 n⌉; 30 repetições aleatórias; ataque por grau, intermediação e PageRank (ranking fixo); curvas até 30 % |
| Extra | ER, BA e WS comparados com o grafo observado |

## Resultados principais (semente 42)

- **Estrutura:** 1.899 vértices, 20.296 arestas dirigidas (13.838 não dirigidas), grau médio 14,57 (mediana 5, máx. 255),
  maior componente com 1.893 vértices, diâmetro 8, raio 4, caminho médio 3,055, clusterização média 0,109, 14.319 triângulos.
- **Algoritmos:** a rede não é euleriana (1.062 vértices de grau ímpar); Tarjan: 601 SCCs (maior com 1.294), 395 pontes, 220 articulações;
  a inclinação log-log tempo × complexidade teórica fica entre ≈ 0,65 (Bellman-Ford, parada antecipada) e ≈ 1,0.
- **Small-world:** σ ≈ 14,5 contra G(n,m), mas σ ≈ 1,23 contra a mesma sequência de graus (a clusterização vem dos hubs).
- **Lei de potência:** rejeitada (γ̂ = 2,83, k_min = 35, p = 0,02); a lognormal ajusta melhor.
- **Robustez:** remover 5 % ao acaso mantém a rede (maior componente ≈ 99 % dos vértices restantes);
  remover os 5 % de maior intermediação reduz a maior componente a 76 % de n e cria 339 componentes.
- **Descoberta:** os hubs são emissores em massa (reciprocidade cai de ≈ 78 % para 44 % nos graus mais altos).
- **Modelos clássicos:** nenhum se ajusta; o nulo de mesmo grau é o mais próximo.

## Relatório (LaTeX)

`resultados/macros.tex` traz os números do trabalho como macros (`\diametro`, `\sigmaER`, `\gammaPL`, ...) e `figuras/*.pdf`
as figuras em vetor. Basta `\input{resultados/macros.tex}` no `.tex` do relatório (template WebMedia) e usar as macros.
`scripts/gerar_notebook.py` recria `notebooks/analise.ipynb` a partir de `resultados/` e `figuras/`.

## Referências

- Panzarasa, Opsahl, Carley (2009). *Patterns and dynamics of users' behavior and interaction.* JASIST 60(5).
- Clauset, Shalizi, Newman (2009). *Power-law distributions in empirical data.* SIAM Review 51(4).
- Humphries, Gurney (2008). *Network "small-world-ness".* PLoS ONE 3(4).
- Maslov, Sneppen (2002). *Specificity and stability in topology of protein networks.* Science 296.
- Brandes (2001). *A faster algorithm for betweenness centrality.* J. Math. Sociology 25(2).
