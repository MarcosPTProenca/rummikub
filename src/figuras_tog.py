"""Figuras em largura de coluna para a submissão em duas colunas (artigo/tog.tex).

As figuras do artigo longo têm 7,2 polegadas; comprimidas numa coluna de 3,5 os
rótulos ficam ilegíveis. Aqui cada figura tem a largura final e fontes
dimensionadas para ela, sem escala no LaTeX. Fica também um painel só por
figura: o painel de barras que elas traziam repetia a tabela ao lado.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analise_pontos import NOMES as MENU_NOMES, resumo as resumo_pontos
from src.analise_sessao import MENU as MENU_SESSAO, NOMES as NOMES_SESSAO, resumo as resumo_sessao

SAIDA = Path("artigo/figuras")
LARGURA = 3.45  # coluna do IEEEtran, em polegadas
AZUL, LARANJA = "#4c72b0", "#dd8452"
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def limpo(rotulo):
    return rotulo.replace("$^*$", "*")


def mecanismo() -> None:
    """Forest plot das 11 variantes: o que o texto discute é a ordem e o zero-efeito."""
    linhas = [
        ("joker-saver (full)", 0.568, 0.027, False),
        ("jokers frozen on table", 0.581, 0.027, False),
        ("save only after opening", 0.537, 0.022, False),
        ("save only before opening", 0.495, 0.021, False),
        ("never play a joker", 0.189, 0.021, True),
        ("soft penalty $\\lambda=4$", 0.508, 0.014, False),
        ("soft penalty $\\lambda=2$", 0.506, 0.006, False),
        ("soft penalty $\\lambda=1$", 0.503, 0.003, False),
        ("soft penalty $\\lambda=0.5$", 0.500, 0.000, False),
        ("joker bonus $\\lambda=-1$", 0.500, 0.000, False),
        ("joker bonus $\\lambda=-2$", 0.500, 0.000, False),
    ]
    fig, ax = plt.subplots(figsize=(LARGURA, 2.3))
    ys = list(range(len(linhas)))[::-1]
    ax.errorbar([l[1] for l in linhas], ys, xerr=[l[2] for l in linhas], fmt="o",
                color=AZUL, ms=3, capsize=1.5, lw=0.7, zorder=3)
    for y, l in zip(ys, linhas):
        if l[3]:
            ax.plot(l[1], y, "o", color=LARANJA, ms=3, zorder=4)
    ax.axvline(0.5, color="gray", ls="--", lw=0.7)
    ax.set_yticks(ys, [l[0] for l in linhas], fontsize=6)
    ax.set_xlabel("score against max-tiles")
    ax.set_xlim(0.12, 0.65)
    fig.tight_layout(pad=0.3)
    fig.savefig(SAIDA / "tog_mecanismo.pdf")
    plt.close(fig)


def pontos() -> None:
    """As duas métricas, só o topo do ranking: é onde elas discordam."""
    linhas = resumo_pontos(Path("resultados/plano12/pontos_225000.jsonl"))
    # joker-spender cai exatamente sobre max-tiles (idênticos em todas as partidas
    # comparáveis), então fica fora da figura e entra na legenda.
    topo = [n for n in sorted(linhas, key=lambda n: -linhas[n]["win"])[:6] if n != "gastar_coringa"]
    fig, ax = plt.subplots(figsize=(LARGURA, 2.4))
    desvios = {"poupar_descarta_perto": (-32, 11), "max_pontos": (8, 6), "max_pecas": (8, -11),
               "poupar_descarta": (-24, 9), "poupar_coringa": (6, -10)}
    for n in topo:
        cor = LARANJA if "$^*$" in MENU_NOMES[n] else AZUL
        ax.errorbar(linhas[n]["win"], linhas[n]["pts"], xerr=linhas[n]["win_ic"],
                    yerr=linhas[n]["pts_ic"], fmt="o", color=cor, ms=4, capsize=1.5, lw=0.7, zorder=3)
        ax.annotate(limpo(MENU_NOMES[n]), (linhas[n]["win"], linhas[n]["pts"]),
                    textcoords="offset points", xytext=desvios[n], fontsize=5.8, color=cor)
    ax.set_xlabel("win score vs.\u00a0the other nine")
    ax.set_ylabel("official points per game")
    ax.margins(x=0.28, y=0.34)
    fig.tight_layout(pad=0.3)
    fig.savefig(SAIDA / "tog_pontos.pdf")
    plt.close(fig)


def sessao() -> None:
    """Mãos ganhas contra sessões ganhas: a dissociação é o resultado."""
    linhas = resumo_sessao(Path("resultados/plano14/sessao_245000_k10.jsonl"))
    fig, ax = plt.subplots(figsize=(LARGURA, 2.4))
    desvios = {"poupar_descarta_perto": (6, 6), "sessao_adaptativa": (7, -3),
               "max_pecas": (6, 4), "max_pontos": (-18, -16),
               "poupar_descarta": (5, 6), "poupar_coringa": (5, -10)}
    for n in MENU_SESSAO:
        cor = LARANJA if "$^*$" in NOMES_SESSAO[n] else AZUL
        ax.errorbar(linhas[n]["maos"], linhas[n]["ses"], xerr=linhas[n]["maos_ic"],
                    yerr=linhas[n]["ses_ic"], fmt="o", color=cor, ms=4, capsize=1.5, lw=0.7, zorder=3)
        ax.annotate(limpo(NOMES_SESSAO[n]), (linhas[n]["maos"], linhas[n]["ses"]),
                    textcoords="offset points", xytext=desvios[n], fontsize=5.8, color=cor)
    ax.axhline(0.5, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("hands won per session (of 10)")
    ax.set_ylabel("sessions won")
    ax.margins(x=0.30, y=0.22)
    fig.tight_layout(pad=0.3)
    fig.savefig(SAIDA / "tog_sessao.pdf")
    plt.close(fig)


if __name__ == "__main__":
    mecanismo()
    pontos()
    sessao()
    print("figuras tog_*.pdf em artigo/figuras/")
