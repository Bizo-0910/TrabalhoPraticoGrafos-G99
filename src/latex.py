"""Utilitários para gerar tabelas LaTeX e macros (os números do relatório vêm direto dos resultados)."""
import re


def _fmt(x, casas=4):
    if isinstance(x, float):
        return f'{x:.{casas}f}'.replace('.', ',') if abs(x) < 1e6 else f'{x:.3e}'
    return str(x)


def tabela_tex(df, legenda, rotulo, casas=4):
    cols = list(df.columns)
    linhas = [r'\begin{table}[ht]', r'\centering', rf'\caption{{{legenda}}}', rf'\label{{{rotulo}}}',
              r'\begin{tabular}{' + 'l' * len(cols) + '}', r'\hline',
              ' & '.join(_esc(c) for c in cols) + r' \\', r'\hline']
    for _, row in df.iterrows():
        linhas.append(' & '.join(_esc(_fmt(v, casas)) for v in row) + r' \\')
    linhas += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(linhas)


def _esc(s):
    s = str(s)
    for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'), ('→', r'$\to$'),
                 ('≥', r'$\geq$'), ('≠', r'$\neq$'), ('·', r'$\cdot$'), ('³', r'$^3$'), ('σ', r'$\sigma$')]:
        s = s.replace(a, b)
    return s


def macro(nome, valor):
    """\\newcommand com nome só de letras (LaTeX não aceita dígitos/underscore em macros)."""
    nome = re.sub(r'[^A-Za-z]', '', nome)
    return rf'\newcommand{{\{nome}}}{{{valor}}}'
