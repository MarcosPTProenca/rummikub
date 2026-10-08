"""Gera artigo/tabela_liga.tex e imprime pares/ciclos a partir de resultados/liga_pontos.jsonl (Plano 8)."""
import json
import math
import statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

NOMES = {"poupar_coringa": "joker-saver", "max_pecas": "max-tiles", "max_pontos": "max-points",
         "minimo": "min-play", "aleatorio": "random", "so_baixar": "no-rearrange", "cauteloso": "cautious",
         "adaptativo": "adaptive*", "reorganizador": "reorganiser*", "flexivel": "flexible*",
         "poupar_ate_2": "saver-until-2*", "poupar_ate_3": "saver-until-3*"}


def pares(registros):
    """(a, b) -> {semente: média de pontos de a nos dois assentos}."""
    g = defaultdict(lambda: defaultdict(list))
    for r in registros:
        g[(r["a"], r["b"])][r["semente"]].append(r["pontos_a"])
    return {k: {s: st.mean(v) for s, v in d.items()} for k, d in g.items()}


def p_valor(valores) -> float:
    v = list(valores)
    if st.stdev(v) == 0:
        return float(st.mean(v) == 0.5)
    z = (st.mean(v) - 0.5) / (st.stdev(v) / math.sqrt(len(v)))
    return math.erfc(abs(z) / math.sqrt(2))


def holm(ps: dict, alfa: float = 0.05) -> set:
    rejeitados = set()
    for i, (k, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        if p > alfa / (len(ps) - i):
            break
        rejeitados.add(k)
    return rejeitados


def bradley_terry(vitorias: dict, nomes: list, iteracoes: int = 500) -> dict:
    """vitorias[(i, j)] = pontos de i contra j (empate = 0,5); estimação MM; força normalizada para média geométrica 1."""
    forca = {n: 1.0 for n in nomes}
    for _ in range(iteracoes):
        novo = {}
        for i in nomes:
            ganhos = sum(vitorias.get((i, j), 0) for j in nomes if j != i)
            den = sum((vitorias.get((i, j), 0) + vitorias.get((j, i), 0)) / (forca[i] + forca[j])
                      for j in nomes if j != i)
            novo[i] = ganhos / den
        g = math.exp(st.mean(math.log(x) for x in novo.values()))
        forca = {n: x / g for n, x in novo.items()}
    return forca


def ciclos(vencedor_sobre: set, nomes: list) -> list:
    return [(a, b, c) for a, b, c in combinations(nomes, 3)
            for x, y, z in ((a, b, c), (a, c, b))
            if (x, y) in vencedor_sobre and (y, z) in vencedor_sobre and (z, x) in vencedor_sobre]


def main() -> None:
    registros = [json.loads(l) for l in Path("resultados/liga_pontos.jsonl").read_text().splitlines()]
    por_par = pares(registros)
    nomes = sorted({r["a"] for r in registros} | {r["b"] for r in registros})
    pontos, vit, ps = defaultdict(list), {}, {}
    for (a, b), d in por_par.items():
        m = st.mean(d.values())
        vit[(a, b)], vit[(b, a)] = m * len(d), (1 - m) * len(d)
        pontos[a].append(m), pontos[b].append(1 - m)
        ps[(a, b)] = p_valor(d.values())
    rejeitados = holm(ps)
    vencedor_sobre = {(a, b) if st.mean(por_par[(a, b)].values()) > 0.5 else (b, a) for a, b in rejeitados}
    forca = bradley_terry(vit, nomes)
    saida = defaultdict(list)
    for r in registros:
        if r["pontos_a"] == 1.0:
            saida[r["a"]].append(r["saida_com_coringa_a"])
    ordem = sorted(nomes, key=lambda n: -st.mean(pontos[n]))
    linhas = ["\\begin{tabular}{@{}lccc@{}}\\toprule",
              "Strategy & Mean score & Bradley--Terry & Wins ending with a joker (\\%)\\\\\\midrule"]
    for n in ordem:
        cor = f"{100 * st.mean(saida[n]):.0f}" if saida[n] else "--"
        linhas.append(f"{NOMES[n]} & ${st.mean(pontos[n]):.3f}$ & ${forca[n]:.2f}$ & {cor} \\\\")
    linhas.append("\\bottomrule\\end{tabular}")
    Path("artigo/tabela_liga.tex").write_text("\n".join(linhas) + "\n")
    print(f"partidas: {len(registros)}; pares significativos (Holm, 66 pares): {len(rejeitados)}; ciclos: {ciclos(vencedor_sobre, nomes)}")
    topo = ordem[:4]
    for a, b in combinations(topo, 2):
        k = (a, b) if (a, b) in por_par else (b, a)
        m = st.mean(por_par[k].values())
        print(f"{k[0]} vs {k[1]}: {m:.3f} p={ps[k]:.3g} holm={k in rejeitados}")


if __name__ == "__main__":
    main()
