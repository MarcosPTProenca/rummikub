"""Tabela e figura do plano 15: o RL escalado no campo comum.

A pergunta que a figura responde é "o agente aprendido é o melhor?", e a
resposta honesta exige a comparação pareada, não o lugar no ranking: a rede
aparece em primeiro por +1,36 pontos sobre max-tiles, mas a diferença nas
mesmas mãos é -1,69 [-3,70, +0,31], indistinguível de zero.
"""
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
CAMPO = Path("resultados/plano15/campo_270000.jsonl")
NOMES = {"aleatorio": "random", "max_pecas": "max-tiles", "max_pontos": "max-points",
         "minimo": "min-play", "poupar_coringa": "joker-saver", "so_baixar": "no-rearrange",
         "cauteloso": "cautious", "poupar_descarta": "saver-then-dump",
         "poupar_descarta_perto": "saver-dump-near", "gastar_coringa": "joker-spender",
         "rede": "points-net (RL)"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})


def ic95(x):
    return 1.96 * statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def carregar(caminho=CAMPO):
    vitoria, pontos = collections.defaultdict(list), collections.defaultdict(list)
    for linha in Path(caminho).read_text().splitlines():
        r = json.loads(linha)
        for x, y, sinal in ((r["a"], r["b"], +1), (r["b"], r["a"], -1)):
            vitoria[(x, y, r["semente"])].append(
                r["vitoria_a"] if sinal > 0 else 1 - r["vitoria_a"])
            pontos[(x, y, r["semente"])].append(sinal * r["pontos_a"])
    return vitoria, pontos


def resumo(vitoria, pontos):
    """Média por mão de cada competidor contra o campo; as três sementes da rede somam um agente."""
    nomes = sorted({k[0] for k in vitoria})
    redes = [n for n in nomes if n.startswith("rede_pontos")]
    grupos = {n: [n] for n in nomes if n not in redes} | {"rede": redes}
    sementes = sorted({k[2] for k in vitoria})
    linhas = {}
    for nome, membros in grupos.items():
        campo = [o for o in nomes if o not in membros]
        p = [statistics.mean([x for m in membros for o in campo for x in pontos[(m, o, s)]])
             for s in sementes]
        v = [statistics.mean([x for m in membros for o in campo for x in vitoria[(m, o, s)]])
             for s in sementes]
        linhas[nome] = {"pts": statistics.mean(p), "pts_ic": ic95(p),
                        "win": statistics.mean(v), "win_ic": ic95(v)}
    return linhas, nomes, sementes


def pareado(pontos, nomes, sementes):
    """Rede menos max-tiles nas mesmas mãos, contra o mesmo campo de oponentes comuns."""
    redes = [n for n in nomes if n.startswith("rede_pontos")]
    comum = [o for o in nomes if o not in redes and o != "max_pecas"]
    difs = []
    for s in sementes:
        r = statistics.mean([x for m in redes for o in comum for x in pontos[(m, o, s)]])
        m = statistics.mean([x for o in comum for x in pontos[("max_pecas", o, s)]])
        difs.append(r - m)
    return statistics.mean(difs), ic95(difs), len(difs)


def duelo(vitoria, pontos, oponente, sementes, nomes):
    redes = [n for n in nomes if n.startswith("rede_pontos")]
    p = [statistics.mean([x for m in redes for x in pontos[(m, oponente, s)]]) for s in sementes]
    v = [statistics.mean([x for m in redes for x in vitoria[(m, oponente, s)]]) for s in sementes]
    return statistics.mean(p), ic95(p), statistics.mean(v)


def figura(linhas, destino=SAIDA / "campo_rl.pdf"):
    ordem = sorted(linhas, key=lambda n: linhas[n]["pts"])
    ys = range(len(ordem))
    fig, eixo = plt.subplots(figsize=(5.0, 3.4))
    cores = ["#00897B" if n == "rede" else "#607D8B" for n in ordem]
    eixo.barh(list(ys), [linhas[n]["pts"] for n in ordem],
              xerr=[linhas[n]["pts_ic"] for n in ordem],
              color=cores, error_kw={"lw": 0.8, "ecolor": "#263238"})
    eixo.set_yticks(list(ys), [NOMES.get(n, n) for n in ordem], fontsize=8)
    eixo.axvline(0, color="#90A4AE", lw=0.8)
    eixo.set_xlabel("Official points per hand against the common field")
    eixo.set_title("Scaled RL in the common field (9,360 games)", fontsize=9)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino)
    plt.close(fig)
    return destino


def tabela(linhas, destino=ARTIGO / "tabela_campo_rl.tex"):
    ordem = sorted(linhas, key=lambda n: -linhas[n]["pts"])
    corpo = []
    for n in ordem:
        l = linhas[n]
        nome = NOMES.get(n, n)
        if n == "rede":
            nome = r"\textbf{points-net (RL)}"
        corpo.append(f"{nome} & ${l['pts']:+.2f}\\pm{l['pts_ic']:.2f}$ & "
                     f"${l['win']:.3f}\\pm{l['win_ic']:.3f}$\\\\")
    destino.write_text(
        "\\begin{tabular}{lcc}\\toprule\n"
        "Agent & Official points & Win rate\\\\\\midrule\n"
        + "\n".join(corpo) + "\n\\bottomrule\\end{tabular}\n")
    return destino


def main():
    vitoria, pontos = carregar()
    linhas, nomes, sementes = resumo(vitoria, pontos)
    print(f"{len(sementes)} mãos, {len(nomes)} competidores\n")
    for n in sorted(linhas, key=lambda x: -linhas[x]["pts"]):
        l = linhas[n]
        print(f"  {NOMES.get(n, n):22s} {l['pts']:+7.2f} ±{l['pts_ic']:4.2f}   "
              f"win {l['win']:.3f}")
    media, meia, n = pareado(pontos, nomes, sementes)
    print(f"\npareado rede − max-tiles: {media:+.2f} [{media - meia:+.2f}, {media + meia:+.2f}] "
          f"({n} mãos) → {'real' if media - meia > 0 else 'indistinguível de zero'}")
    for oponente in ("max_pecas", "poupar_coringa"):
        p, h, v = duelo(vitoria, pontos, oponente, sementes, nomes)
        print(f"duelo vs {NOMES[oponente]:12s}: {p:+.2f} [{p - h:+.2f}, {p + h:+.2f}] pts, "
              f"win {v:.3f}")
    print("\n" + str(figura(linhas)))
    print(tabela(linhas))


if __name__ == "__main__":
    main()
