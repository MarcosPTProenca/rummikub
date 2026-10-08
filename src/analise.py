"""Gráficos e números do artigo a partir de resultados/*.jsonl."""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("resultados")
SAIDA = Path("artigo/figuras")
NOMES = {"aleatorio": "random", "max_pecas": "max-tiles", "max_pontos": "max-points",
         "minimo": "min-play", "poupar_coringa": "joker-saver", "so_baixar": "no-rearrange",
         "cauteloso": "cautious", "rl_meta": "Q-meta (RL)", "jev_meta": "Jev-meta",
         "comprar": "draw"}
OPCOES = ["max_pecas", "max_pontos", "poupar_coringa", "so_baixar", "minimo", "cauteloso", "comprar"]
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})


def ler(arquivo: Path) -> list[dict]:
    return [json.loads(l) for l in arquivo.read_text().splitlines()]


def media_ic(valores) -> list[float]:
    v = np.asarray(valores, float)
    erro = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return [float(v.mean()), float(erro)]


def por_semente(registros: list[dict]) -> dict[tuple[str, str], list[float]]:
    """Pontuação de A contra B por semente: média dos dois assentos (desenho pareado)."""
    grupos = defaultdict(lambda: defaultdict(list))
    for r in registros:
        grupos[(r["a"], r["b"])][r["semente"]].append(r["pontos_a"])
    return {par: [float(np.mean(v)) for v in sementes.values()] for par, sementes in grupos.items()}


def matriz(pares, nomes):
    m = np.full((len(nomes), len(nomes)), np.nan)
    for (a, b), v in pares.items():
        m[nomes.index(a), nomes.index(b)] = np.mean(v)
        m[nomes.index(b), nomes.index(a)] = 1 - np.mean(v)
    return m


def barras(arquivo, nomes, valores, erros, cor, rotulo):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ys = np.arange(len(nomes))[::-1]
    ax.barh(ys, valores, xerr=erros, color=cor, capsize=2)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_yticks(ys, [NOMES[n] for n in nomes])
    ax.set_xlabel(rotulo)
    fig.tight_layout()
    fig.savefig(SAIDA / arquivo)
    plt.close(fig)


def pontuacao_por_estrategia(pares, ranking) -> dict:
    por_estrategia = {}
    for n in ranking:
        contra = [np.array(v) if a == n else 1 - np.array(v) for (a, b), v in pares.items() if n in (a, b)]
        por_estrategia[n] = media_ic(np.mean(contra, axis=0))
    return por_estrategia


def torneio(saidas: dict) -> list[str]:
    reg = ler(RES / "torneio.jsonl")
    pares = por_semente(reg)
    nomes = sorted({r["a"] for r in reg} | {r["b"] for r in reg})
    bruta = matriz(pares, nomes)
    ranking = sorted(nomes, key=lambda n: -np.nanmean(bruta[nomes.index(n)]))
    m = matriz(pares, ranking)
    por_estrategia = pontuacao_por_estrategia(pares, ranking)
    saidas["ranking"] = por_estrategia
    saidas["matriz"] = {"nomes": ranking, "valores": [[None if np.isnan(x) else round(float(x), 3) for x in linha] for linha in m]}
    saidas["pares"] = {f"{a}|{b}": media_ic(v) for (a, b), v in pares.items()}
    saidas["assento_0"] = media_ic([r["pontos_a"] if r["assento_a"] == 0 else 1 - r["pontos_a"] for r in reg])
    saidas["empates"] = float(np.mean([r["pontos_a"] == 0.5 for r in reg]))
    saidas["decisoes_medias"] = float(np.mean([r["decisoes"] for r in reg]))
    saidas["partidas_torneio"] = len(reg)
    barras("ranking.pdf", ranking, [por_estrategia[n][0] for n in ranking],
           [por_estrategia[n][1] for n in ranking], "#4c72b0",
           "Mean score vs. the other six")

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(m, cmap="RdBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(ranking)), [NOMES[n] for n in ranking], rotation=45, ha="right")
    ax.set_yticks(range(len(ranking)), [NOMES[n] for n in ranking])
    for i in range(len(ranking)):
        for j in range(len(ranking)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(m[i, j] - 0.5) > 0.3 else "black")
    fig.colorbar(im, label="Row's score against column")
    fig.tight_layout()
    fig.savefig(SAIDA / "matriz.pdf")
    plt.close(fig)
    return ranking


def treino(saidas: dict, ranking: list[str]) -> list[dict]:
    replicas = ler(RES / "treino.jsonl")
    janela = 100
    curvas = np.array([np.convolve((np.array(r["retornos"]) + 1) / 2, np.ones(janela) / janela, mode="valid")
                       for r in replicas])
    x = np.arange(janela, curvas.shape[1] + janela)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for c in curvas:
        ax.plot(x, c, color="#4c72b0", alpha=0.25, lw=0.8)
    ax.plot(x, curvas.mean(axis=0), color="#c44e52", lw=1.6, label="mean of replicas")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Training game")
    ax.set_ylabel(f"Score (moving avg. of {janela})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SAIDA / "aprendizado.pdf")
    plt.close(fig)
    saidas["aprendizado"] = {"inicio": float(curvas[:, 0].mean()), "fim": float(curvas[:, -1].mean()),
                             "replicas": len(replicas), "episodios": len(replicas[0]["retornos"])}
    por_oponente = {o: media_ic([np.mean([a["pontos_a"] for a in r["avaliacao"] if a["b"] == o])
                                 for r in replicas]) for o in ranking}
    saidas["rl_vs"] = por_oponente
    saidas["rl_vs_campo"] = media_ic([np.mean([a["pontos_a"] for a in r["avaliacao"]]) for r in replicas])
    saidas["partidas_treino"] = sum(len(r["retornos"]) for r in replicas)
    saidas["partidas_avaliacao_rl"] = sum(len(r["avaliacao"]) for r in replicas)
    barras("rl_vs_oponentes.pdf", ranking, [por_oponente[n][0] for n in ranking],
           [por_oponente[n][1] for n in ranking], "#55a868", "Q-meta score against opponent (greedy policy)")
    return replicas


def rl_desempate(saidas: dict, ranking: list[str]) -> None:
    """Q-meta treinado e avaliado com a regra oficial (treino_pontos.jsonl) ao lado da regra de empate."""
    arquivo = RES / "treino_pontos.jsonl"
    if not arquivo.exists() or not arquivo.stat().st_size:
        return
    saida = {}
    for regra, caminho in (("empate", RES / "treino.jsonl"), ("pontos", arquivo)):
        replicas = ler(caminho)
        saida[regra] = {
            "replicas": len(replicas),
            "campo": media_ic([np.mean([a["pontos_a"] for a in r["avaliacao"]]) for r in replicas]),
            "vs": {o: media_ic([np.mean([a["pontos_a"] for a in r["avaliacao"] if a["b"] == o]) for r in replicas])
                   for o in ranking},
        }
    saidas["rl_desempate"] = saida


def politica(saidas: dict, replicas: list[dict]) -> None:
    """Estratégia escolhida pela política greedy em cada estado aprendido, somando as réplicas."""
    fases = ("before opening", "after opening")
    contagem = {f: defaultdict(float) for f in fases}
    total = dict.fromkeys(fases, 0)
    for r in replicas:
        valores = defaultdict(dict)
        for estado, opcao, valor in r["tabela_q"]:
            valores[tuple(estado)][opcao] = valor
        for estado, qs in valores.items():
            if len(qs) < 3 or max(qs.values()) == min(qs.values()):
                continue
            fase = fases[1] if estado[0] == "True" else fases[0]
            contagem[fase][max(qs, key=qs.get)] += 1
            total[fase] += 1
    saidas["politica"] = {f: {o: contagem[f][o] / max(total[f], 1) for o in OPCOES} for f in fases}
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    x = np.arange(len(OPCOES))
    for i, fase in enumerate(fases):
        ax.bar(x + (i - 0.5) * 0.38, [saidas["politica"][fase][o] for o in OPCOES], 0.38, label=fase)
    ax.set_xticks(x, [NOMES.get(o, o) for o in OPCOES], rotation=40, ha="right")
    ax.set_ylabel("Share of learned states")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SAIDA / "politica.pdf")
    plt.close(fig)


def jev(saidas: dict) -> None:
    arquivos = [a for a in (RES / "jev.jsonl", RES / "jev_lider.jsonl") if a.exists()]
    if not arquivos:
        return
    reg = [r for a in arquivos for r in ler(a)]
    saidas["jev_vs"] = {b: media_ic(v) for (a, b), v in por_semente(reg).items()}
    escolhas = defaultdict(int)
    for r in reg:
        for o, n in r["jev"]["escolhas"].items():
            escolhas[o] += n
    tokens = sum(r["jev"]["tokens"] for r in reg)
    saidas["jev_custo"] = {"partidas": len(reg), "chamadas": sum(r["jev"]["chamadas"] for r in reg),
                           "tokens": tokens, "usd": tokens * 0.042 / 1e6,
                           "fallbacks": sum(r["jev"]["fallbacks"] for r in reg),
                           "falhas": sum(r["jev"]["falhas"] for r in reg)}
    saidas["jev_escolhas"] = dict(escolhas)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    total = sum(escolhas.values())
    ax.bar([NOMES.get(o, o) for o in OPCOES], [escolhas[o] / total for o in OPCOES], color="#8172b2")
    ax.set_ylabel("Share of Jev decisions")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(SAIDA / "jev_escolhas.pdf")
    plt.close(fig)


def teste(valores, nulo=0.5) -> dict:
    """Média por semente, IC 95% e p bilateral (aproximação normal, n grande) contra o valor nulo."""
    v = np.asarray(valores, float)
    media, erro = float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))
    z = (media - nulo) / erro if erro else 0.0
    return {"media": media, "ic95": 1.96 * erro, "n_sementes": len(v), "p": math.erfc(abs(z) / math.sqrt(2))}


def holm(ps: list[float]) -> list[float]:
    ordem = sorted(range(len(ps)), key=ps.__getitem__)
    ajustados, maximo = [0.0] * len(ps), 0.0
    for posicao, i in enumerate(ordem):
        maximo = max(maximo, min(1.0, (len(ps) - posicao) * ps[i]))
        ajustados[i] = maximo
    return ajustados


ARTIGO = Path("artigo")


def rotulos_tabela():
    return [("poupar_coringa|normal", "joker-saver (full rule)"),
            ("poupar_coringa|congelado", "joker-saver, jokers frozen on table"),
            ("poupar_depois|normal", "save only after the opening meld"),
            ("poupar_antes|normal", "save only before the opening meld"),
            ("poupar_sem_final|normal", "never play a joker, even to go out"),
            ("penal_4|normal", "soft penalty $\\lambda=4$"),
            ("penal_2|normal", "soft penalty $\\lambda=2$"),
            ("penal_1|normal", "soft penalty $\\lambda=1$"),
            ("penal_0.5|normal", "soft penalty $\\lambda=0.5$"),
            ("bonus_1|normal", "joker bonus $\\lambda=-1$"),
            ("bonus_2|normal", "joker bonus $\\lambda=-2$"),
            ("max_pecas|normal", "max-tiles mirror (control)")]


def mecanismo(saidas: dict) -> None:
    arquivo = RES / "mecanismo.jsonl"
    if not arquivo.exists():
        return
    grupos = defaultdict(list)
    for r in ler(arquivo):
        if r["semente"] < 70_500:
            grupos[(r["a"], r["congelar"])].append(r)
    linhas = {}
    for (nome, congelar), regs in grupos.items():
        por_semente = defaultdict(list)
        for r in regs:
            por_semente[r["semente"]].append(r["pontos_a"])
        linha = teste([np.mean(v) for v in por_semente.values()])
        n = len(regs)
        for lado in ("a", "b"):
            chave = f"tele_{lado}"
            linha[f"coringas_jogados_{lado}"] = sum(r[chave]["coringas_jogados"] for r in regs) / n
            linha[f"compras_{lado}"] = sum(r[chave]["compras"] for r in regs) / n
            linha[f"jogadas_{lado}"] = sum(r[chave]["jogadas"] for r in regs) / n
            linha[f"coringas_mao_fim_{lado}"] = sum(r[f"coringas_mao_{lado}"] for r in regs) / n
        vitorias = [r for r in regs if r["pontos_a"] == 1.0]
        linha["vitorias_saindo_com_coringa"] = (
            sum(r["saida_com_coringa_a"] for r in vitorias) / len(vitorias)) if vitorias else None
        linha["partidas"] = n
        linhas[f"{nome}|{'congelado' if congelar else 'normal'}"] = linha
    ordem = list(linhas)
    for k, p in zip(ordem, holm([linhas[k]["p"] for k in ordem])):
        linhas[k]["p_holm"] = p
    saidas["mecanismo"] = linhas
    nomes = dict(rotulos_tabela())
    corpo = []
    for k, rot in nomes.items():
        if k not in linhas:
            continue
        l = linhas[k]
        pv = "$<0.001$" if l["p_holm"] < 0.001 else f"{l['p_holm']:.3f}"
        saida = "--" if l["vitorias_saindo_com_coringa"] is None else f"{100 * l['vitorias_saindo_com_coringa']:.0f}\\%"
        corpo.append(f"{rot} & ${l['media']:.3f}\\pm{l['ic95']:.3f}$ & {pv} & {l['coringas_jogados_a']:.2f} & "
                     f"{l['coringas_mao_fim_a']:.2f} & {saida} \\\\")
    cabecalho = ("\\begin{tabular}{@{}lcrccc@{}}\\toprule\nVariant & Score & $p_{\\text{Holm}}$ & Jokers played & "
                 "Jokers left & Win with joker\\\\\\midrule\n")
    (ARTIGO / "tabela_mecanismo.tex").write_text(cabecalho + "\n".join(corpo) + "\n\\bottomrule\n\\end{tabular}\n")

    rotulos = [("poupar_coringa|normal", "joker-saver (full)"),
               ("poupar_coringa|congelado", "joker-saver, jokers frozen on table"),
               ("poupar_depois|normal", "save only after opening"),
               ("poupar_antes|normal", "save only before opening"),
               ("poupar_sem_final|normal", "never play jokers, even to go out"),
               ("penal_4|normal", "soft penalty, 4 tiles per joker"),
               ("penal_2|normal", "soft penalty, 2 tiles per joker"),
               ("penal_1|normal", "soft penalty, 1 tile per joker"),
               ("penal_0.5|normal", "soft penalty, 0.5 tile per joker"),
               ("bonus_1|normal", "joker bonus, 1 tile"),
               ("bonus_2|normal", "joker bonus, 2 tiles")]
    presentes = [(k, r) for k, r in rotulos if k in linhas]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ys = list(range(len(presentes)))[::-1]
    ax.errorbar([linhas[k]["media"] for k, _ in presentes], ys, xerr=[linhas[k]["ic95"] for k, _ in presentes],
                fmt="o", color="#4c72b0", capsize=2)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_yticks(ys, [r for _, r in presentes], fontsize=7)
    ax.set_xlabel("Score against max-tiles (500 seeds $\\times$ 2 seats)")
    fig.tight_layout()
    fig.savefig(SAIDA / "dose_resposta.pdf")
    plt.close(fig)


REGRAS = [("empate", RES / "torneio.jsonl", RES / "mecanismo.jsonl", "Draw (0.5)"),
          ("pontos", RES / "torneio_pontos.jsonl", RES / "mecanismo_pontos.jsonl", "Fewest points"),
          ("pecas", RES / "torneio_pecas.jsonl", RES / "mecanismo_pecas.jsonl", "Fewest tiles")]


def placar_mecanismo(arquivo: Path, nome: str) -> dict | None:
    if not arquivo.exists():
        return None
    por = defaultdict(list)
    for r in ler(arquivo):
        if r["a"] == nome and not r["congelar"] and r["semente"] < 70_500:
            por[r["semente"]].append(r["pontos_a"])
    return teste([np.mean(v) for v in por.values()]) if por else None


def desempate(saidas: dict) -> None:
    """Sensibilidade à regra de fim de partida com o monte esgotado."""
    por_regra = {}
    for regra, torneio_arq, mecanismo_arq, _ in REGRAS:
        if not torneio_arq.exists():
            continue
        reg = ler(torneio_arq)
        pares = por_semente(reg)
        nomes = sorted({r["a"] for r in reg} | {r["b"] for r in reg})
        pont = pontuacao_por_estrategia(pares, nomes)
        por_regra[regra] = {
            "ranking": pont, "partidas": len(reg),
            "empates": float(np.mean([r["pontos_a"] == 0.5 for r in reg])),
            "assento_0": media_ic([r["pontos_a"] if r["assento_a"] == 0 else 1 - r["pontos_a"] for r in reg]),
            "empates_com_no_rearrange": float(np.mean([r["pontos_a"] == 0.5 for r in reg if "so_baixar" in (r["a"], r["b"])])),
            "empates_sem_no_rearrange": float(np.mean([r["pontos_a"] == 0.5 for r in reg if "so_baixar" not in (r["a"], r["b"])])),
            "e4": {n: placar_mecanismo(mecanismo_arq, n) for n in ("poupar_coringa", "poupar_sem_final")},
        }
    if len(por_regra) < 2:
        return
    base = por_regra["empate"]["ranking"]
    ordem = sorted(base, key=lambda n: -base[n][0])
    for regra, d in por_regra.items():
        a = np.argsort(np.argsort([-base[n][0] for n in ordem]))
        b = np.argsort(np.argsort([-d["ranking"][n][0] for n in ordem]))
        d["spearman_com_empate"] = float(np.corrcoef(a, b)[0, 1])
    saidas["desempate"] = por_regra
    regras = [(r, rot) for r, _, _, rot in REGRAS if r in por_regra]
    corpo = [f"{NOMES[n]} & " + " & ".join(
        f"${por_regra[r]['ranking'][n][0]:.3f}\\pm{por_regra[r]['ranking'][n][1]:.3f}$" for r, _ in regras) + " \\\\"
        for n in sorted(ordem, key=lambda n: -por_regra.get("pontos", por_regra["empate"])["ranking"][n][0])]
    corpo.append("\\midrule")
    corpo.append("Draws (\\% of games) & " + " & ".join(f"{100 * por_regra[r]['empates']:.1f}" for r, _ in regras) + " \\\\")
    for nome, rot in (("poupar_coringa", "Joker-saver vs.\\ max-tiles"), ("poupar_sem_final", "Never-play-joker vs.\\ max-tiles")):
        celulas = []
        for r, _ in regras:
            e = por_regra[r]["e4"][nome]
            celulas.append("--" if e is None else f"${e['media']:.3f}\\pm{e['ic95']:.3f}$")
        corpo.append(f"{rot} & " + " & ".join(celulas) + " \\\\")
    cabecalho = ("\\begin{tabular}{@{}l" + "c" * len(regras) + "@{}}\\toprule\nStrategy & "
                 + " & ".join(rot for _, rot in regras) + "\\\\\\midrule\n")
    (ARTIGO / "tabela_desempate.tex").write_text(cabecalho + "\n".join(corpo) + "\n\\bottomrule\n\\end{tabular}\n")


def coringas_campo(saidas: dict) -> None:
    arquivo = RES / "coringas_campo.json"
    if not arquivo.exists():
        return
    d = json.loads(arquivo.read_text())
    if "por_coringas_iniciais" not in d:
        return
    saidas["coringas_campo"] = {k: v for k, v in d.items() if k != "por_estrategia_inicial"}
    linhas = []
    for titulo, chave in (("Initial jokers", "por_coringas_iniciais"), ("Bought jokers", "por_coringas_comprados")):
        for c, v in sorted(d[chave].items()):
            linhas.append(f"{titulo} = {c} & {v['pontuacao']:.3f}$\\pm${v['ic95']:.3f} & {v['maos']} \\\\")
        linhas.append("\\midrule" if titulo == "Initial jokers" else "")
    ef = lambda e: "--" if e["efeito"] is None else f"${e['efeito']:+.3f}$ [{e['ic95'][0]:+.3f}, {e['ic95'][1]:+.3f}]"
    linhas = [l for l in linhas if l]
    linhas += ["\\midrule",
               f"Initial joker, effect (strategy-stratified) & \\multicolumn{{2}}{{c}}{{{ef(d['inicial_estratificado'])}}} \\\\",
               f"Bought joker, naive difference & \\multicolumn{{2}}{{c}}{{{ef(d['comprado_ingenuo'])}}} \\\\",
               f"Bought joker, stratified by draws & \\multicolumn{{2}}{{c}}{{{ef(d['comprado_estratificado'])}}} \\\\"]
    cab = "\\begin{tabular}{@{}lcr@{}}\\toprule\nGroup & Score (pooled field) & Hands\\\\\\midrule\n"
    (ARTIGO / "tabela_coringas_campo.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n")
    por = d["comprado_por_estrategia"]
    ordem = [n for n in OPCOES if n in por and por[n]["efeito"] is not None]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ys = list(range(len(ordem) + 1))[::-1]
    pontos = [d["comprado_estratificado"]] + [por[n] for n in ordem]
    ax.errorbar([e["efeito"] for e in pontos], ys,
                xerr=[[e["efeito"] - e["ic95"][0] for e in pontos], [e["ic95"][1] - e["efeito"] for e in pontos]],
                fmt="o", color="#4c72b0", capsize=2)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_yticks(ys, ["all (pooled)"] + [NOMES[n] for n in ordem])
    ax.set_xlabel("Effect of buying $\\geq 1$ joker on score (stratified)")
    fig.tight_layout()
    fig.savefig(SAIDA / "coringa_comprado.pdf")
    plt.close(fig)


def coringas(saidas: dict) -> None:
    arquivo = RES / "coringas.json"
    if not arquivo.exists():
        return
    dados = json.loads(arquivo.read_text())
    saidas["coringas_iniciais"] = dados
    corpo = []
    for nome, por in dados.items():
        corpo.append(" & ".join([NOMES[nome]] + [
            f"{por[c]['pontuacao']:.3f}$\\pm${por[c]['ic95']:.3f} ({por[c]['maos']})" if c in por else "--"
            for c in ("0", "1", "2")]) + " \\\\")
    cabecalho = ("\\begin{tabular}{@{}lccc@{}}\\toprule\nStrategy (mirror) & 0 jokers & 1 joker & 2 jokers\\\\\\midrule\n")
    (ARTIGO / "tabela_coringas.tex").write_text(cabecalho + "\n".join(corpo) + "\n\\bottomrule\n\\end{tabular}\n")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for i, (nome, por) in enumerate(dados.items()):
        xs = sorted(por)
        ax.errorbar([int(x) + (i - 1) * 0.08 for x in xs], [por[x]["pontuacao"] for x in xs],
                    yerr=[por[x]["ic95"] for x in xs], marker="o", capsize=2, label=NOMES[nome])
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Jokers in the initial hand")
    ax.set_ylabel("Score (mirror matches)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SAIDA / "coringas_iniciais.pdf")
    plt.close(fig)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    saidas: dict = {}
    ranking = torneio(saidas)
    if (RES / "treino.jsonl").exists() and (RES / "treino.jsonl").stat().st_size:
        politica(saidas, treino(saidas, ranking))
        rl_desempate(saidas, ranking)
    jev(saidas)
    mecanismo(saidas)
    coringas(saidas)
    desempate(saidas)
    coringas_campo(saidas)
    (RES / "resumo.json").write_text(json.dumps(saidas, indent=2, ensure_ascii=False))
    print(json.dumps(saidas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
