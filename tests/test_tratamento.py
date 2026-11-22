from src import tratamento as T


def test_tratamento_colapsa_e_pesa():
    msgs = [(1, 2, 10), (1, 2, 20), (2, 1, 30), (3, 3, 40), (2, 4, 50), (7, 8, 60)]
    d = T.construir(msgs)
    assert d.n_mensagens == 6 and d.n_autoloops == 1
    assert d.g.m == 3                                   # {1,2}, {2,4}, {7,8}; loop removido
    i1, i2 = d.ids.index(1), d.ids.index(2)
    assert d.cont_dir[(i1, i2)] == 2 and d.cont_dir[(i2, i1)] == 1
    w = {frozenset((u, v)): w for u, v, w in d.g.arestas}
    assert abs(w[frozenset((i1, i2))] - 1 / 3) < 1e-12   # 3 mensagens no par -> w = 1/3
    assert len(d.componentes) == 3 and d.gc.n == 3        # maior componente: {1, 2, 4}


def test_componentes_ordenadas():
    comps = T.componentes([[1], [0], [3], [2, 4], [3]])
    assert [len(c) for c in comps] == [3, 2]
