"""Tabela e figura do plano 12: as duas métricas lado a lado."""
import collections
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.estrategias.politicas import ESTRATEGIAS, PONTOS

SAIDA = Path("artigo/figuras")
ARTIGO = Path("artigo")
NOMES = {"aleatorio": "random", "max_pecas": "max-tiles", "max_pontos": "max-points",
         "minimo": "min-play", "poupar_coringa": "joker-saver", "so_baixar": "no-rearrange",
         "cauteloso": "cautious", "poupar_descarta": "saver-then-dump$^*$",
         "poupar_descarta_perto": "saver-dump-near$^*$", "gastar_coringa": "joker-spender$^*$"}
TODAS = list(ESTRATEGIAS) + list(PONTOS)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})


def ic95(x):
    return 1.96 * statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def carregar(caminho):
    vitoria, pontos, coringas = (collections.defaultdict(list) for _ in range(3))
    for linha in open(caminho):
        r = json.loads(linha)
        a, b, s = r["a"], r["b"], r["semente"]
        vitoria[(a, b, s)].append(r["pontos_a"])
        vitoria[(b, a, s)].append(1.0 - r["pontos_a"])
        pontos[(a, b, s)].append(r["pontos_oficiais_a"])
        pontos[(b, a, s)].append(-r["pontos_oficiais_a"])
        coringas[a].append(r["coringas_mao_a"])
        coringas[b].append(r["coringas_mao_b"])
    return vitoria, pontos, coringas


def contra_campo(tabela, estrategia, campo, sementes):
    return [statistics.mean([v for o in campo for v in tabela[(estrategia, o, s)]])
            for s in sementes if any(tabela[(estrategia, o, s)] for o in campo)]


def resumo(caminho):
    vitoria, pontos, coringas = carregar(caminho)
    sementes = sorted({k[2] for k in vitoria})
    linhas = {}
    for nome in TODAS:
        campo = [o for o in TODAS if o != nome]
        v = contra_campo(vitoria, nome, campo, sementes)
        p = contra_campo(pontos, nome, campo, sementes)
        linhas[nome] = {"win": statistics.mean(v), "win_ic": ic95(v),
                        "pts": statistics.mean(p), "pts_ic": ic95(p),
                        "coringa": statistics.mean(coringas[nome])}
    return linhas


def tabela(linhas) -> None:
    ordem = sorted(TODAS, key=lambda n: -linhas[n]["win"])
    postos_pts = {n: i for i, n in enumerate(sorted(TODAS, key=lambda n: -linhas[n]["pts"]), 1)}
    corpo = []
    for n in ordem:
        l = linhas[n]
        corpo.append(f"{NOMES[n]} & ${l['win']:.3f}\\pm{l['win_ic']:.3f}$ & "
                     f"${l['pts']:+.1f}\\pm{l['pts_ic']:.1f}$ & {postos_pts[n]} & {l['coringa']:.2f} \\\\")
    cabecalho = ("\\begin{tabular}{@{}lcccc@{}}\\toprule\nStrategy & Win score & Official points & "
                 "Points rank & Jokers left\\\\\\midrule\n")
    (ARTIGO / "tabela_pontos.tex").write_text(cabecalho + "\n".join(corpo) + "\n\\bottomrule\n\\end{tabular}\n")


def figura(linhas) -> None:
    fig, (esq, dir) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ordem = sorted(TODAS, key=lambda n: linhas[n]["win"])
    ys = range(len(ordem))
    cores = ["#dd8452" if n in PONTOS else "#4c72b0" for n in ordem]
    esq.barh(list(ys), [linhas[n]["win"] for n in ordem],
             xerr=[linhas[n]["win_ic"] for n in ordem], color=cores, error_kw={"lw": 0.8})
    esq.axvline(0.5, color="gray", ls="--", lw=0.8)
    esq.set_yticks(list(ys), [NOMES[n].replace("$^*$", "*") for n in ordem], fontsize=7)
    esq.set_xlabel("win score vs.\u00a0the other nine")

    # Painel direito: só as seis do topo. As duas fracas achatariam a escala de pontos
    # (no-rearrange está em -102) e esconderiam a inversão, que é o ponto da figura.
    topo = sorted(TODAS, key=lambda n: -linhas[n]["win"])[:6]
    desvios = {"poupar_coringa": (6, -10), "max_pecas": (6, -4), "max_pontos": (6, 4),
               "poupar_descarta": (-10, 8), "poupar_descarta_perto": (-56, -4),
               "gastar_coringa": (6, -10)}
    for n in topo:
        cor = "#dd8452" if n in PONTOS else "#4c72b0"
        dir.errorbar(linhas[n]["win"], linhas[n]["pts"], xerr=linhas[n]["win_ic"],
                     yerr=linhas[n]["pts_ic"], fmt="o", color=cor, ms=5, capsize=2, lw=0.8, zorder=3)
        dir.annotate(NOMES[n].replace("$^*$", "*"), (linhas[n]["win"], linhas[n]["pts"]),
                     textcoords="offset points", xytext=desvios[n], fontsize=6.5, color=cor)
    dir.set_xlabel("win score vs.\u00a0the other nine")
    dir.set_ylabel("official points per game")
    dir.set_title("the six strongest by win score", fontsize=7.5)
    dir.margins(x=0.22, y=0.3)
    fig.tight_layout()
    fig.savefig(SAIDA / "pontos.pdf")
    plt.close(fig)


if __name__ == "__main__":
    linhas = resumo(Path("resultados/plano12/pontos_225000.jsonl"))
    tabela(linhas)
    figura(linhas)
    print("artigo/tabela_pontos.tex e artigo/figuras/pontos.pdf")
