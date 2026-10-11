"""Tabela e figura do plano 14: a sessão de várias mãos inverte a vantagem."""
import collections
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAIDA = Path("artigo/figuras")
ARTIGO = Path("artigo")
MENU = ["max_pecas", "max_pontos", "poupar_coringa", "poupar_descarta_perto",
        "poupar_descarta", "sessao_adaptativa"]
NOMES = {"max_pecas": "max-tiles", "max_pontos": "max-points", "poupar_coringa": "joker-saver",
         "poupar_descarta_perto": "saver-dump-near$^*$", "poupar_descarta": "saver-then-dump$^*$",
         "sessao_adaptativa": "session-adaptive$^*$"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})


def ic95(x):
    return 1.96 * statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def carregar(caminho):
    vit, pts, maos = (collections.defaultdict(list) for _ in range(3))
    for linha in open(caminho):
        r = json.loads(linha)
        a, b, s = r["a"], r["b"], r["semente"]
        vit[(a, b, s)].append(r["vitoria_a"])
        vit[(b, a, s)].append(1.0 - r["vitoria_a"])
        pts[(a, b, s)].append(r["pontos_a"])
        pts[(b, a, s)].append(-r["pontos_a"])
        maos[(a, b, s)].append(r["maos_ganhas_a"])
        maos[(b, a, s)].append(r["maos_ganhas_b"])
    return vit, pts, maos


def resumo(caminho):
    vit, pts, maos = carregar(caminho)
    sementes = sorted({k[2] for k in vit})
    linhas = {}
    for nome in MENU:
        campo = [o for o in MENU if o != nome]
        def serie(tabela):
            return [statistics.mean([v for o in campo for v in tabela[(nome, o, s)]])
                    for s in sementes if any(tabela[(nome, o, s)] for o in campo)]
        v, p, m = serie(vit), serie(pts), serie(maos)
        linhas[nome] = {"ses": statistics.mean(v), "ses_ic": ic95(v),
                        "pts": statistics.mean(p), "pts_ic": ic95(p),
                        "maos": statistics.mean(m), "maos_ic": ic95(m)}
    return linhas


def tabela(linhas) -> None:
    corpo = []
    for n in sorted(MENU, key=lambda n: -linhas[n]["ses"]):
        l = linhas[n]
        corpo.append(f"{NOMES[n]} & ${l['ses']:.3f}\\pm{l['ses_ic']:.3f}$ & "
                     f"${l['pts']:+.0f}\\pm{l['pts_ic']:.0f}$ & ${l['maos']:.2f}\\pm{l['maos_ic']:.2f}$ \\\\")
    cabecalho = ("\\begin{tabular}{@{}lccc@{}}\\toprule\nStrategy & Sessions won & Points per session & "
                 "Hands won of 10\\\\\\midrule\n")
    (ARTIGO / "tabela_sessao.tex").write_text(cabecalho + "\n".join(corpo) + "\n\\bottomrule\n\\end{tabular}\n")


def figura(linhas) -> None:
    fig, (esq, dir) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ordem = sorted(MENU, key=lambda n: linhas[n]["ses"])
    cores = ["#dd8452" if "$^*$" in NOMES[n] else "#4c72b0" for n in ordem]
    esq.barh(range(len(ordem)), [linhas[n]["ses"] for n in ordem],
             xerr=[linhas[n]["ses_ic"] for n in ordem], color=cores, error_kw={"lw": 0.8})
    esq.axvline(0.5, color="gray", ls="--", lw=0.8)
    esq.set_yticks(range(len(ordem)), [NOMES[n].replace("$^*$", "*") for n in ordem], fontsize=7)
    esq.set_xlabel("sessions won vs.\u00a0the other five")

    # O ponto da figura: ganhar mãos e ganhar sessões andam em direções opostas.
    desvios = {"poupar_coringa": (-12, -14), "poupar_descarta": (8, 0),
               "max_pecas": (8, -2), "max_pontos": (-18, -14),
               "poupar_descarta_perto": (8, 2), "sessao_adaptativa": (8, -2)}
    for n in MENU:
        cor = "#dd8452" if "$^*$" in NOMES[n] else "#4c72b0"
        dir.errorbar(linhas[n]["maos"], linhas[n]["ses"], xerr=linhas[n]["maos_ic"],
                     yerr=linhas[n]["ses_ic"], fmt="o", color=cor, ms=5, capsize=2, lw=0.8, zorder=3)
        dir.annotate(NOMES[n].replace("$^*$", "*"), (linhas[n]["maos"], linhas[n]["ses"]),
                     textcoords="offset points", xytext=desvios[n], fontsize=6.5, color=cor)
    dir.axhline(0.5, color="gray", ls="--", lw=0.8)
    dir.set_xlabel("hands won per session (of 10)")
    dir.set_ylabel("sessions won")
    dir.margins(x=0.25, y=0.18)
    fig.tight_layout()
    fig.savefig(SAIDA / "sessao.pdf")
    plt.close(fig)


if __name__ == "__main__":
    linhas = resumo(Path("resultados/plano14/sessao_245000_k10.jsonl"))
    tabela(linhas)
    figura(linhas)
    print("artigo/tabela_sessao.tex e artigo/figuras/sessao.pdf")
