"""Plano 19: treinar para ganhar a sessão de várias mãos, não cada mão isolada.

A recompensa de sessão é um potencial sobre a probabilidade de vencer:

    V(s, h) = Φ(s / (σ·√h))        s = meus pontos acumulados, h = mãos restantes
    R_mão   = V(s + p, h − 1) − V(s, h)

A soma de `R` sobre a sessão telescopa para `1[vitória] − 0,5`, então o objetivo final
é exatamente ganhar a sessão. Cada mão recebe só o seu `R`: `V` já embute o futuro, o que
faz de `R` uma vantagem TD com crítico fixo. Esse crítico assume mãos independentes e
simétricas (não é o verdadeiro), e o viés está declarado no plano.

Ao contrário do shaping do plano 17, isto MUDA a política ótima: `V` é côncava para quem
está à frente e convexa para quem está atrás, então o valor de um ponto de variância
depende de (s, h).
"""
import json
import math
import random
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .politicas import ESTRATEGIAS, criar
from .rede import ESCALA_PONTOS, PoliticaRede, RedeSessao, carregar_rede, vantagens_grupo
from .sessao import _jogar_mao, semente_da_mao

# Desvio de pontos por mão do modelo de referência. Fica entre os 19,6 (rede vs max-tiles),
# 33,1 (joker-saver vs max-tiles) e 36,3 (rede vs joker-saver) medidos; fixado antes de rodar
# e não ajustado depois.
SIGMA = 30.0
RECOMPENSAS = ("sessao", "pontos")
_CDF = NormalDist().cdf


def valor_sessao(s: float, h: int, sigma: float = SIGMA) -> float:
    """Probabilidade de vencer a sessão com `s` pontos acumulados e `h` mãos por jogar."""
    if h <= 0:
        return 1.0 if s > 0 else 0.5 if s == 0 else 0.0
    return _CDF(s / (sigma * math.sqrt(h)))


def recompensa_mao(modo: str, s: float, p: float, h_restantes: int, sigma: float = SIGMA) -> float:
    """Recompensa de uma mão: `p` pontos com `s` acumulados e `h_restantes` mãos (esta incluída)."""
    if modo == "pontos":
        return p / ESCALA_PONTOS
    if modo == "sessao":
        return valor_sessao(s + p, h_restantes - 1, sigma) - valor_sessao(s, h_restantes, sigma)
    raise ValueError(f"recompensa desconhecida: {modo!r} (use {' ou '.join(RECOMPENSAS)})")


def _jogar_sessao_treino(tarefa):
    """Uma sessão de treino do agente contra uma heurística; devolve o que cada mão rendeu.

    Retorna `(maos, total, vitoria)`, com `maos` = lista de (recompensa, decisões gravadas)."""
    (theta, ocultos, oponente, sessao_semente, assento, variante, maos, modo, sigma, rearranjar) = tarefa
    rede = RedeSessao(theta, ocultos)
    saldos = [0.0, 0.0]
    resultado = []
    for h in range(maos):
        agente = PoliticaRede.da_rede(rede, random.Random(f"{sessao_semente}|{variante}|{h}"), gravar=True)
        outro = 1 - assento
        adversario = criar(oponente, random.Random(sessao_semente * 10 + outro))
        politicas = [None, None]
        politicas[assento], politicas[outro] = agente, adversario
        mao = _jogar_mao(["_", "_"], semente_da_mao(sessao_semente, h), h % 2, saldos, None, rearranjar,
                         politicas=politicas, maos_restantes=maos - h, maos_total=maos)
        p = mao["pontos"][assento]
        resultado.append((recompensa_mao(modo, saldos[assento], p, maos - h, sigma),
                          list(agente.trajetoria)))
        for lado in (0, 1):
            saldos[lado] += mao["pontos"][lado]
    total = saldos[assento]
    return resultado, total, 1.0 if total > 0 else 0.5 if total == 0 else 0.0


def treinar_sessao(saida: Path, iteracoes: int, sessoes: int, processos: int, inicial: Path,
                   maos: int = 10, recompensa: str = "sessao", grupo: int = 4, lr: float = 3e-3,
                   beta: float = 0.01, semente: int = 0, base_semente: int = 8_000_000,
                   sigma: float = SIGMA, rearranjar: bool = False, checkpoint_a_cada: int = 50) -> RedeSessao:
    """REINFORCE com grupos de `grupo` sessões que compartilham as mesmas mãos.

    A vantagem de cada decisão é a recompensa da sua mão padronizada dentro do grupo, por
    índice de mão (mesmas cartas, escolhas diferentes). O oponente de cada grupo é sorteado
    entre as heurísticas originais. `base_semente` é parâmetro explícito e vai ao manifest:
    em `treinar_rede` a base é fixa, o que já causou uma afirmação errada num plano."""
    if recompensa not in RECOMPENSAS:
        raise ValueError(f"recompensa desconhecida: {recompensa!r} (use {' ou '.join(RECOMPENSAS)})")
    if sessoes % grupo:
        raise ValueError(f"sessoes ({sessoes}) deve ser múltiplo de grupo ({grupo})")
    rede = carregar_rede(inicial)
    if not isinstance(rede, RedeSessao):
        raise ValueError("o ponto de partida precisa ser uma RedeSessao (use RedeSessao.de_rede)")
    saida.mkdir(parents=True, exist_ok=True)
    rng = random.Random(semente)
    heuristicas = list(ESTRATEGIAS)
    contador = passos = 0
    registro_arquivo = (saida / "treino.jsonl").open("w")
    pool = Pool(processos) if processos > 1 else None
    try:
        for it in range(1, iteracoes + 1):
            tarefas = []
            for _ in range(sessoes // grupo):
                contador += 1
                assento, oponente = rng.randrange(2), rng.choice(heuristicas)
                tarefas += [(list(rede.theta), rede.ocultos, oponente, base_semente + contador, assento, v,
                             maos, recompensa, sigma, rearranjar) for v in range(grupo)]
            resultados = pool.map(_jogar_sessao_treino, tarefas) if pool else [
                _jogar_sessao_treino(t) for t in tarefas]
            grad = np.zeros(len(rede.theta))
            decisoes = 0
            for inicio in range(0, len(resultados), grupo):
                bloco = resultados[inicio:inicio + grupo]
                for h in range(maos):
                    vantagens = vantagens_grupo([sessao_[0][h][0] for sessao_ in bloco])
                    for vantagem, (por_mao, _, _) in zip(vantagens, bloco):
                        for X, indice in por_mao[h][1]:
                            grad += rede.gradiente(X, indice, vantagem, beta)
                            decisoes += 1
            if decisoes:
                passos += 1
                rede.passo(grad / decisoes, passos, lr)
            registro = {
                "iteracao": it, "recompensa": recompensa, "maos": maos, "sigma": sigma, "grupo": grupo,
                "sessoes": sessoes, "base_semente": base_semente, "rearranjar": rearranjar, "lr": lr,
                "beta": beta, "decisoes": decisoes,
                "vitorias_sessao": sum(v for _, _, v in resultados) / len(resultados),
                "pontos_sessao": sum(t for _, t, _ in resultados) / len(resultados),
                "retorno_medio": sum(r for res, _, _ in resultados for r, _ in res)
                / max(1, sum(len(res) for res, _, _ in resultados)),
            }
            registro_arquivo.write(json.dumps(registro) + "\n")
            registro_arquivo.flush()
            if it % checkpoint_a_cada == 0:
                rede.salvar(saida / f"iter_{it:06d}.json", iteracao=it)
    finally:
        registro_arquivo.close()
        if pool:
            pool.close()
            pool.join()
    rede.salvar(saida / "final.json", iteracao=iteracoes)
    return rede
