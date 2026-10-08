"""Gera artigo/tabela_limites.tex a partir de resultados/limites.jsonl (Plano 7)."""
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

PERFIS = ["padrao", "x4", "x16", "xl"]
NOMES = {"poupar_coringa": "joker-saver", "max_pecas": "max-tiles", "max_pontos": "max-points",
         "minimo": "min-play", "aleatorio": "random", "so_baixar": "no-rearrange", "cauteloso": "cautious"}
EXCLUIDO = {"so_baixar", "cauteloso"}


def carregar(caminho: Path):
    por = defaultdict(dict)
    for linha in caminho.read_text().splitlines():
        r = json.loads(linha)
        por[r["perfil"]][(r["a"], r["b"], r["semente"], r["assento_a"])] = r
    comuns = set.intersection(*[set(por[p]) for p in PERFIS])
    return por, {k for k in comuns if {k[0], k[1]} != EXCLUIDO}


def pontuacao(por, comuns, perfil):
    pts = defaultdict(list)
    for k in comuns:
        v = por[perfil][k]["pontos_a"]
        pts[k[0]].append(v)
        pts[k[1]].append(1 - v)
    return {n: st.mean(v) for n, v in pts.items()}


def saver_vs_max(por, comuns, perfil, rng):
    por_semente = defaultdict(list)
    for k in comuns:
        if {k[0], k[1]} == {"poupar_coringa", "max_pecas"}:
            v = por[perfil][k]["pontos_a"]
            por_semente[k[2]].append(v if k[0] == "poupar_coringa" else 1 - v)
    medias = [st.mean(v) for v in por_semente.values()]
    boot = sorted(st.mean(rng.choices(medias, k=len(medias))) for _ in range(4000))
    return st.mean(medias), boot[100], boot[3899], len(medias)


def main() -> None:
    por, comuns = carregar(Path("resultados/limites.jsonl"))
    rks = {p: pontuacao(por, comuns, p) for p in PERFIS}
    ordem = sorted(rks["padrao"], key=lambda n: -rks["padrao"][n])
    linhas = ["\\begin{tabular}{@{}lcccc@{}}\\toprule",
              "Strategy & Default (128) & $\\times4$ (512) & $\\times16$ (2048) & Large (4096)\\\\\\midrule"]
    for n in ordem:
        linhas.append(f"{NOMES[n]} & " + " & ".join(f"${rks[p][n]:.3f}$" for p in PERFIS) + " \\\\")
    linhas.append("\\midrule")
    sat = [sum(r["saturacao"]["planos"] for r in por[p].values()) / sum(r["decisoes"] for r in por[p].values())
           for p in PERFIS]
    linhas.append("Decisions at the plan cap (\\%) & " + " & ".join(f"{100 * s:.1f}" for s in sat) + " \\\\")
    rng = random.Random(0)
    celulas = []
    for p in PERFIS:
        m, lo, hi, _ = saver_vs_max(por, comuns, p, rng)
        celulas.append(f"${m:.3f}$ [{lo:.2f}, {hi:.2f}]")
    linhas.append("Joker-saver vs.\\ max-tiles & " + " & ".join(celulas) + " \\\\")
    linhas.append("\\bottomrule\\end{tabular}")
    Path("artigo/tabela_limites.tex").write_text("\n".join(linhas) + "\n")
    print(f"jogos pareados: {len(comuns)}")


if __name__ == "__main__":
    main()
