"""Figuras dos Planos 6, 7 e 8 (limites de busca, mesas de N jogadores, liga ampliada)."""
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analise_liga import NOMES, bradley_terry, pares
from src.analise_limites import PERFIS, carregar, pontuacao, saver_vs_max
from src.analise_mesa import campo, media_ic, por_semente

SAIDA = Path("artigo/figuras")
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})


def ler(arquivo: str) -> list[dict]:
    return [json.loads(l) for l in Path(arquivo).read_text().splitlines()]


def limites() -> None:
    por, comuns = carregar(Path("resultados/limites.jsonl"))
    rks = {p: pontuacao(por, comuns, p) for p in PERFIS}
    caps = [128, 512, 2048, 4096]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for n in sorted(rks["padrao"], key=lambda n: -rks["padrao"][n]):
        a.plot(caps, [rks[p][n] for p in PERFIS], marker="o", ms=3, label=NOMES[n].rstrip("*"),
               lw=2 if n == "poupar_coringa" else 1)
    a.set_xscale("log", base=2)
    a.set_xticks(caps, [str(c) for c in caps])
    a.set_xlabel("plans per decision (cap)")
    a.set_ylabel("tournament score")
    a.legend(fontsize=6, ncol=2, frameon=False, loc="center left", bbox_to_anchor=(0, 0.45))
    rng = random.Random(0)
    dados = [saver_vs_max(por, comuns, p, rng) for p in PERFIS]
    m = np.array([d[0] for d in dados])
    b.errorbar(range(4), m, yerr=[m - [d[1] for d in dados], [d[2] for d in dados] - m], fmt="o-", capsize=3)
    b.axhline(0.5, color="gray", ls="--", lw=0.8)
    b.set_xticks(range(4), [str(c) for c in caps])
    b.set_xlabel("plans per decision (cap)")
    b.set_ylabel("joker-saver vs max-tiles")
    fig.tight_layout()
    fig.savefig(SAIDA / "limites.pdf")
    plt.close(fig)


def mesa() -> None:
    registros = ler("resultados/mesa_pontos.jsonl")
    g = por_semente(registros)
    colunas = [(3, "iguais", "3 players\nvs max-tiles"), (3, "misto", "3 players\nmixed"),
               (4, "iguais", "4 players\nvs max-tiles"), (4, "misto", "4 players\nmixed")]
    focos = ["poupar_coringa", "max_pecas", "max_pontos", "minimo", "aleatorio"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    largura = 0.16
    for i, f in enumerate(focos):
        valores = [media_ic(g[(n, c, f)].values()) for n, c, _ in colunas]
        a.bar(np.arange(4) + (i - 2) * largura, [v[0] for v in valores], largura, yerr=[v[1] for v in valores],
              capsize=1.5, label=NOMES[f])
    a.hlines([1 / 3, 1 / 3, 1 / 4, 1 / 4], np.arange(4) - 0.45, np.arange(4) + 0.45, color="gray", ls="--", lw=0.8)
    a.set_xticks(range(4), [c[2] for c in colunas])
    a.set_ylabel("score (chance dashed)")
    a.set_ylim(0, 0.8)
    a.legend(fontsize=6, ncol=5, frameon=False, loc="upper center")
    # joker-saver vs max-tiles por número de jogadores (2 jogadores: torneio com a regra oficial)
    dois = [r for r in ler("resultados/mecanismo_pontos.jsonl")
            if r["a"] == "poupar_coringa" and r["b"] == "max_pecas" and not r["congelar"]]
    por_sem = defaultdict(list)
    for r in dois:
        por_sem[r["semente"]].append(r["pontos_a"])
    v2 = [st.mean(v) for v in por_sem.values()]
    pts = [(2, st.mean(v2), 1.96 * st.stdev(v2) / len(v2) ** 0.5, 0.5)]
    for n in (3, 4):
        m, e = media_ic(g[(n, "iguais", "poupar_coringa")].values())
        pts.append((n, m, e, 1 / n))
    b.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], fmt="o-", capsize=3, label="joker-saver")
    b.plot([p[0] for p in pts], [p[3] for p in pts], "--", color="gray", label="chance 1/N")
    b.set_xticks([2, 3, 4])
    b.set_xlabel("players")
    b.set_ylabel("joker-saver vs max-tiles")
    b.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(SAIDA / "mesa.pdf")
    plt.close(fig)


def liga() -> None:
    registros = ler("resultados/liga_pontos.jsonl")
    pp = pares(registros)
    nomes = sorted({r["a"] for r in registros} | {r["b"] for r in registros})
    vit, pontos = {}, defaultdict(list)
    for (x, y), d in pp.items():
        m = st.mean(d.values())
        vit[(x, y)], vit[(y, x)] = m * len(d), (1 - m) * len(d)
        pontos[x].append(m), pontos[y].append(1 - m)
    forca = bradley_terry(vit, nomes)
    ordem = sorted(nomes, key=lambda n: -forca[n])
    mat = np.full((len(ordem),) * 2, np.nan)
    for (x, y), d in pp.items():
        mat[ordem.index(x), ordem.index(y)] = st.mean(d.values())
        mat[ordem.index(y), ordem.index(x)] = 1 - st.mean(d.values())
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    ys = np.arange(len(ordem))[::-1]
    cor = ["tab:orange" if NOMES[n].endswith("*") else "tab:blue" for n in ordem]
    a.barh(ys, [forca[n] for n in ordem], color=cor)
    a.set_yticks(ys, [NOMES[n] for n in ordem])
    a.set_xlabel("Bradley-Terry strength (orange: exploratory)")
    im = b.imshow(mat, cmap="RdBu", vmin=0, vmax=1)
    b.set_xticks(range(len(ordem)), [NOMES[n] for n in ordem], rotation=90, fontsize=6)
    b.set_yticks(range(len(ordem)), [NOMES[n] for n in ordem], fontsize=6)
    b.set_xlabel("opponent")
    b.set_title("row's score against column", fontsize=8)
    fig.colorbar(im, ax=b, fraction=0.046)
    fig.tight_layout()
    fig.savefig(SAIDA / "liga.pdf")
    plt.close(fig)


if __name__ == "__main__":
    limites(), mesa(), liga()
