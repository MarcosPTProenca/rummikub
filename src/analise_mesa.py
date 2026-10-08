"""Gera artigo/tabela_mesa.tex a partir de resultados/mesa_pontos.jsonl (Plano 6)."""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

NOMES = {"poupar_coringa": "joker-saver", "max_pecas": "max-tiles", "max_pontos": "max-points",
         "minimo": "min-play", "aleatorio": "random", "so_baixar": "no-rearrange", "cauteloso": "cautious"}


def campo(r: dict) -> str:
    return "iguais" if set(r["mesa"]) <= {r["foco"], "max_pecas"} and r["mesa"].count("max_pecas") >= r["n"] - 1 else "misto"


def por_semente(registros):
    """(n, campo, foco) -> {semente: média dos pontos nas rotações do assento}."""
    grupos = defaultdict(lambda: defaultdict(list))
    for r in registros:
        grupos[(r["n"], campo(r), r["foco"])][(r.get("replica"), r["semente"])].append(r["pontos_foco"])
    return {k: {s: st.mean(v) for s, v in d.items()} for k, d in grupos.items()}


def media_ic(valores):
    v = list(valores)
    return st.mean(v), 1.96 * st.stdev(v) / math.sqrt(len(v))


def linhas_aprendidas(grupos, colunas=((3, "iguais"), (3, "misto"), (4, "iguais"), (4, "misto"))):
    linhas = []
    for foco, nome in (("rede", "network"), ("rl_meta", "Q-learner")):
        celulas = []
        for c in colunas:
            m, ic = media_ic(grupos[(*c, foco)].values())
            celulas.append(f"${m:.3f}\\pm{ic:.3f}$")
        linhas.append(f"{nome} & " + " & ".join(celulas) + " \\\\")
    return linhas


def ler(arquivo: str) -> list[dict]:
    registros = [json.loads(l) for l in Path(arquivo).read_text().splitlines()]
    for r in registros:
        if r["foco"].startswith("rede:"):
            r["foco"] = "rede"
            r["mesa"] = ["rede" if x.startswith("rede:") else x for x in r["mesa"]]
    return registros


def main() -> None:
    registros = ler("resultados/mesa_pontos.jsonl")
    aprendidos = [r for n in (3, 4) for r in ler(f"resultados/mesa_rede_n{n}_pontos.jsonl")]
    aprendidos += [r for n in (3, 4) for l in Path(f"resultados/treino_pontos_n{n}.jsonl").read_text().splitlines()
                   for r in json.loads(l)["avaliacao"]]
    grupos = por_semente(registros)
    grupos_aprendidos = por_semente(aprendidos)
    colunas = [(3, "iguais"), (3, "misto"), (4, "iguais"), (4, "misto")]
    ordem = sorted(NOMES, key=lambda f: -media_ic(grupos[(3, "misto", f)].values())[0])
    linhas = ["\\begin{tabular}{@{}lcccc@{}}\\toprule",
              " & \\multicolumn{2}{c}{3 players ($1/N=0.333$)} & \\multicolumn{2}{c}{4 players ($1/N=0.250$)}\\\\",
              "Strategy & vs max-tiles & mixed field & vs max-tiles & mixed field\\\\\\midrule"]
    for f in ordem:
        celulas = []
        for c in colunas:
            m, ic = media_ic(grupos[(*c, f)].values())
            celulas.append(f"${m:.3f}\\pm{ic:.3f}$")
        linhas.append(f"{NOMES[f]} & " + " & ".join(celulas) + " \\\\")
    linhas.append("\\midrule")
    linhas += linhas_aprendidas(grupos_aprendidos)
    linhas.append("\\midrule")
    decididas = []
    for c in colunas:
        rs = [r for r in registros + aprendidos if r["n"] == c[0] and campo(r) == c[1]]
        decididas.append(f"{100 * sum(r['vencedor'] is None for r in rs) / len(rs):.1f}")
    linhas.append("Games with no winner (\\%) & " + " & ".join(decididas) + " \\\\")
    linhas.append("\\bottomrule\\end{tabular}")
    Path("artigo/tabela_mesa.tex").write_text("\n".join(linhas) + "\n")


if __name__ == "__main__":
    main()
