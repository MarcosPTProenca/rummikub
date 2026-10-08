"""Chance de vitória por número de coringas na mão inicial: espelho por estratégia ou campo misto."""
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations_with_replacement
from multiprocessing import Pool
from pathlib import Path

from src.estrategias.arena import jogar
from src.estrategias.politicas import ESTRATEGIAS, criar
from src.jogo.jogo import Jogo


def partida(tarefa):
    nome, semente = tarefa
    inicial = [sum(p.cor == "coringa" for p in j.mao) for j in Jogo(semente=semente).jogadores]
    r = jogar([criar(nome, random.Random(semente * 10 + i)) for i in range(2)], semente)
    ponto = [0.5, 0.5] if r["vencedor"] is None else [float(r["vencedor"] == i) for i in range(2)]
    return nome, [(inicial[i], ponto[i]) for i in range(2)]


def resumir(observacoes) -> dict:
    """(semente, grupo, ponto) -> média e IC95 com erro-padrão agrupado por semente (mãos da mesma distribuição são correlacionadas)."""
    grupos = defaultdict(lambda: defaultdict(list))
    for semente, grupo, ponto in observacoes:
        grupos[grupo][semente].append(ponto)
    resumo = {}
    for grupo, por_semente in grupos.items():
        n = sum(len(v) for v in por_semente.values())
        m = sum(sum(v) for v in por_semente.values()) / n
        var = sum((sum(v) - m * len(v)) ** 2 for v in por_semente.values()) / n ** 2
        resumo[grupo] = {"maos": n, "sementes": len(por_semente), "pontuacao": round(m, 4),
                         "ic95": round(1.96 * math.sqrt(var), 4)}
    return resumo


def efeito_estratificado(observacoes, reps: int = 1000, rng_semente: int = 0) -> dict:
    """(semente, estrato, tratado, ponto) -> diferença média tratados-controles dentro de cada estrato,
    ponderada por n1*n0/(n1+n0); IC95 por bootstrap sobre sementes (os dois lados do mesmo jogo são correlacionados)."""
    por_semente = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for semente, estrato, tratado, ponto in observacoes:
        celula = por_semente[semente][(estrato, bool(tratado))]
        celula[0] += 1
        celula[1] += ponto

    def estimar(sementes) -> float:
        total = defaultdict(lambda: [0, 0.0])
        for s in sementes:
            for chave, (n, soma) in por_semente[s].items():
                total[chave][0] += n
                total[chave][1] += soma
        numerador = denominador = 0.0
        for estrato in {e for e, _ in total}:
            n1, s1 = total.get((estrato, True), (0, 0.0))
            n0, s0 = total.get((estrato, False), (0, 0.0))
            if n1 and n0:
                peso = n1 * n0 / (n1 + n0)
                numerador += peso * (s1 / n1 - s0 / n0)
                denominador += peso
        return numerador / denominador if denominador else float("nan")

    lista = list(por_semente)
    rng = random.Random(rng_semente)
    amostras = sorted(x for x in (estimar(rng.choices(lista, k=len(lista))) for _ in range(reps)) if x == x)
    tratados = sum(1 for o in observacoes if o[2])
    pontual = estimar(lista)
    if not amostras or pontual != pontual:
        return {"efeito": None, "ic95": None, "tratados": tratados, "controles": len(observacoes) - tratados}
    return {"efeito": round(pontual, 4),
            "ic95": [round(amostras[int(0.025 * len(amostras))], 4), round(amostras[int(0.975 * len(amostras)) - 1], 4)],
            "tratados": tratados, "controles": len(observacoes) - tratados}


def partida_campo(tarefa):
    a, b, semente, assento_a, desempate = tarefa
    inicial = [sum(p.cor == "coringa" for p in j.mao) for j in Jogo(semente=semente).jogadores]
    nomes = [None, None]
    nomes[assento_a], nomes[1 - assento_a] = a, b
    r = jogar([criar(nomes[i], random.Random(semente * 10 + i)) for i in range(2)], semente, desempate=desempate)
    ponto = [0.5, 0.5] if r["vencedor"] is None else [float(r["vencedor"] == i) for i in range(2)]
    t = r["telemetria"]
    return [{"semente": semente, "estrategia": nomes[i], "iniciais": inicial[i], "iniciais_oponente": inicial[1 - i],
             "comprados": t[i]["coringas_comprados"], "compras": t[i]["compras"], "ponto": ponto[i]} for i in range(2)]


def _cortes(valores, grupos: int = 5) -> list:
    ordenados = sorted(valores)
    return [ordenados[len(ordenados) * k // grupos] for k in range(1, grupos)]


def analisar_campo(maos: list[dict], reps: int = 1000) -> dict:
    """Coringa inicial (sorteio exógeno) e coringa comprado (depende de quantas compras o jogador fez: estratificado)."""
    from bisect import bisect_right
    cortes = _cortes([m["compras"] for m in maos])
    faixa = [bisect_right(cortes, m["compras"]) for m in maos]
    obs = lambda estrato, tratado, filtro=lambda m: True: [
        (m["semente"], estrato(m, f), tratado(m), m["ponto"]) for m, f in zip(maos, faixa) if filtro(m)]
    comprado = lambda m: m["comprados"] >= 1
    por_estrategia = {}
    for nome in sorted({m["estrategia"] for m in maos}):
        por_estrategia[nome] = efeito_estratificado(
            obs(lambda m, f: f, comprado, lambda m, n=nome: m["estrategia"] == n), reps)
    return {
        "maos": len(maos), "cortes_de_compras": cortes,
        "maos_com_coringa_comprado": sum(m["comprados"] >= 1 for m in maos),
        "compras_medias": sum(m["compras"] for m in maos) / len(maos),
        "comprados_por_mao": sum(m["comprados"] for m in maos) / len(maos),
        "por_coringas_iniciais": resumir([(m["semente"], m["iniciais"], m["ponto"]) for m in maos]),
        "inicial_estratificado": efeito_estratificado(
            obs(lambda m, f: m["estrategia"], lambda m: m["iniciais"] >= 1), reps),
        "por_coringas_comprados": resumir([(m["semente"], m["comprados"], m["ponto"]) for m in maos]),
        "comprado_ingenuo": efeito_estratificado(obs(lambda m, f: 0, comprado), reps),
        "comprado_estratificado": efeito_estratificado(obs(lambda m, f: (m["estrategia"], f), comprado), reps),
        "comprado_por_estrategia": por_estrategia,
        "por_estrategia_inicial": resumir([(m["semente"], f"{m['estrategia']}|{min(m['iniciais'], 1)}", m["ponto"])
                                           for m in maos]),
    }


def campo(n: int = 2000, desempate: str = "pontos", bruto=Path("resultados/coringas_campo.jsonl"),
          saida=Path("resultados/coringas_campo.json")) -> None:
    """Cada semente sorteia um confronto entre as 28 duplas (com repetição) das 7 estratégias, nos dois assentos."""
    duplas = list(combinations_with_replacement(ESTRATEGIAS, 2))
    tarefas = [(*duplas[s % len(duplas)], 90_000 + s, assento, desempate) for s in range(n) for assento in (0, 1)]
    with Pool(6) as pool, bruto.open("w") as arquivo:
        for lados in pool.imap_unordered(partida_campo, tarefas, chunksize=4):
            for m in lados:
                arquivo.write(json.dumps(m) + "\n")
    reanalisar(bruto, saida, desempate)


def reanalisar(bruto=Path("resultados/coringas_campo.jsonl"), saida=Path("resultados/coringas_campo.json"),
               desempate: str = "pontos") -> None:
    maos = [json.loads(l) for l in bruto.read_text().splitlines()]
    resumo = {"desempate": desempate, "jogos": len(maos) // 2, **analisar_campo(maos)}
    saida.write_text(json.dumps(resumo, indent=2))
    print(json.dumps({k: v for k, v in resumo.items() if k != "por_estrategia_inicial"}, indent=1)[:3000])


def main(estrategias=("max_pecas", "poupar_coringa", "aleatorio"), n=2000, saida=Path("resultados/coringas.json")):
    tarefas = [(nome, 50_000 + s) for nome in estrategias for s in range(n)]
    grupos = {}
    with Pool(6) as pool:
        for nome, lados in pool.imap_unordered(partida, tarefas, chunksize=10):
            for coringas, ponto in lados:
                grupos.setdefault(nome, {}).setdefault(coringas, []).append(ponto)
    resumo = {}
    for nome, por_c in grupos.items():
        resumo[nome] = {}
        for c, v in sorted(por_c.items()):
            m = sum(v) / len(v)
            dp = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0
            resumo[nome][c] = {"maos": len(v), "pontuacao": round(m, 4), "ic95": round(1.96 * dp / math.sqrt(len(v)), 4)}
    saida.write_text(json.dumps(resumo, indent=2))
    print(json.dumps(resumo, indent=2))


if __name__ == "__main__":
    if sys.argv[1:2] == ["campo"]:
        campo(n=int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    elif sys.argv[1:2] == ["reanalisar"]:
        reanalisar()
    else:
        main(n=int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
