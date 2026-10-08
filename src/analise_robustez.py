"""Matriz de robustez do Plano 9: placar por oponente, pior caso e queda entre oponentes vistos e novos."""
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from src.analise_liga import NOMES
from src.estrategias.politicas import ESTRATEGIAS, EXTRAS


def metricas(celulas: dict[str, float], vistos: set[str]) -> dict:
    v = [x for o, x in celulas.items() if o in vistos]
    n = [x for o, x in celulas.items() if o not in vistos]
    pior = min(celulas.items(), key=lambda kv: kv[1])
    return {"vistos": st.mean(v), "novos": st.mean(n), "queda": st.mean(v) - st.mean(n), "pior": pior}


def familia(agente: str) -> str:
    achado = re.search(r"rede_clone_(g4|rl|re)", agente)
    return achado.group(1) if achado else agente


def celulas(registros) -> dict:
    """(família, oponente) -> lista dos placares por semente (média nos dois assentos), uma por treino e semente."""
    g = defaultdict(lambda: defaultdict(list))
    for r in registros:
        g[(familia(r["a"]), r["b"], r["a"])][r["semente"]].append(r["pontos_a"])
    saida = defaultdict(list)
    for (fam, op, agente), d in g.items():
        saida[(fam, op)] += [st.mean(v) for v in d.values()]
    return saida


def media_ic(valores):
    v = list(valores)
    return st.mean(v), 1.96 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def ler(arquivo: str) -> list[dict]:
    return [json.loads(l) for l in Path(arquivo).read_text().splitlines()]


def resumo(arquivo: str) -> dict:
    c = celulas(ler(arquivo))
    familias = sorted({f for f, _ in c})
    saida = {}
    for f in familias:
        por_op = {op: st.mean(v) for (ff, op), v in c.items() if ff == f}
        saida[f] = {"celulas": por_op, **metricas(por_op, set(ESTRATEGIAS))}
    return saida


def celula(valores) -> str:
    if not valores:
        return "--"
    m, ic = media_ic(valores)
    return f"${m:.3f}\\pm{ic:.3f}$"


def linhas_tabela(c: dict, familias: tuple[str, ...]) -> list[str]:
    ops = [*ESTRATEGIAS, *EXTRAS]
    linhas = [f"{NOMES[o]} & " + " & ".join(celula(c.get((f, o))) for f in familias) + " \\\\" for o in ops]
    resumos = [("Mean, seen (7)", lambda o: o in ESTRATEGIAS), ("Mean, unseen (5)", lambda o: o in EXTRAS)]
    for nome, filtro in resumos:
        celulas_ = []
        for f in familias:
            v = [st.mean(c[(f, o)]) for o in ops if filtro(o) and (f, o) in c]
            celulas_.append(f"${st.mean(v):.3f}$")
        linhas.append(f"{nome} & " + " & ".join(celulas_) + " \\\\")
    piores = []
    for f in familias:
        o, x = min(((o, st.mean(c[(f, o)])) for o in ops if (f, o) in c), key=lambda t: t[1])
        piores.append(f"${x:.3f}$ ({NOMES[o]})")
    linhas.append("Worst case & " + " & ".join(piores) + " \\\\")
    return linhas


def main() -> None:
    c = celulas(ler("resultados/robustez.jsonl") + ler("resultados/robustez_ref.jsonl"))
    familias = ("g4", "rl", "max_pecas", "poupar_coringa")
    cab = ["\\begin{tabular}{@{}lcccc@{}}\\toprule",
           "Opponent & Group $K{=}4$ & Moving avg. & max-tiles & joker-saver\\\\\\midrule"]
    linhas = linhas_tabela(c, familias)
    corpo = linhas[:7] + ["\\midrule"] + linhas[7:12] + ["\\midrule"] + linhas[12:]
    Path("artigo/tabela_robustez.tex").write_text("\n".join(cab + corpo + ["\\bottomrule\\end{tabular}"]) + "\n")


if __name__ == "__main__":
    main()
