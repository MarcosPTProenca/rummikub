"""Clonagem e treino da rede convolucional (plano 16).

Mesma receita do artigo — clone do joker-saver, REINFORCE com vantagem relativa
ao grupo, recompensa em pontos — trocando apenas a arquitetura. A comparação com
`plano15/grande_s*` isola a representação da entrada.
"""
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .arena import jogar
from .conv import PoliticaConv, RedeConv
from .politicas import ESTRATEGIAS, criar
from .rede import retorno_do_jogo, vantagens_grupo
from .tensor import tensor_da_acao


class _GravadorConv:
    """Grava (tensores das ações, índice da escolhida) para a clonagem."""

    def __init__(self, politica, rng: random.Random, maximo: int):
        self.politica, self.rng, self.maximo = politica, rng, maximo
        self.amostras: list[tuple[list, int]] = []

    def __call__(self, jogo, acoes):
        acao = self.politica(jogo, acoes)
        if len(acoes) > 1:
            escolhido = next(i for i, a in enumerate(acoes) if a is acao)
            outros = [i for i in range(len(acoes)) if i != escolhido]
            indices = [escolhido] + self.rng.sample(outros, min(len(acoes), self.maximo) - 1)
            indices.sort()
            self.amostras.append(([tensor_da_acao(jogo, acoes[i]) for i in indices],
                                  indices.index(escolhido)))
        return acao


def _partida_clone(tarefa):
    demonstrador, oponentes, semente, maximo = tarefa
    gravador = _GravadorConv(criar(demonstrador, random.Random(semente)),
                             random.Random(semente), maximo)
    rival = criar(oponentes[semente % len(oponentes)], random.Random(semente + 1))
    ordem = [gravador, rival] if semente % 2 == 0 else [rival, gravador]
    jogar(ordem, semente, desempate="pontos")
    return gravador.amostras


def concordancia(rede: RedeConv, amostras) -> float:
    acertos = sum(max(range(len(X)), key=rede.pontuacoes(X).__getitem__) == i for X, i in amostras)
    return acertos / len(amostras)


def clonar_conv(saida: Path, partidas: int, epocas: int, processos: int, semente: int = 0,
                lr: float = 3e-3, maximo_acoes: int = 32, inicio: int = 160_000) -> RedeConv:
    """Clona o joker-saver na arquitetura convolucional."""
    saida.mkdir(parents=True, exist_ok=True)
    oponentes = list(ESTRATEGIAS)
    tarefas = [("poupar_coringa", oponentes, inicio + semente * 10_000 + s, maximo_acoes)
               for s in range(partidas)]
    if processos > 1:
        with Pool(processos) as pool:
            blocos = pool.map(_partida_clone, tarefas)
    else:
        blocos = [_partida_clone(t) for t in tarefas]
    amostras = [a for b in blocos for a in b]
    corte = int(len(amostras) * 0.9)
    treino, teste = amostras[:corte], amostras[corte:]

    rng = random.Random(semente)
    rede = RedeConv.nova(rng)
    passo = 0
    for _ in range(epocas):
        ordem = list(range(len(treino)))
        rng.shuffle(ordem)
        for i in range(0, len(ordem), 64):
            grupo = ordem[i:i + 64]
            grad = np.zeros(len(rede.theta))
            for k in grupo:
                X, indice = treino[k]
                grad += np.asarray(rede.gradiente(X, indice, 1.0))
            passo += 1
            rede.passo(grad / len(grupo), passo, lr)

    rede.salvar(saida / "final.json")
    (saida / "clonagem.json").write_text(json.dumps({
        "arquitetura": "conv", "demonstrador": "poupar_coringa", "partidas": partidas,
        "amostras": len(amostras), "epocas": epocas, "lr": lr,
        "concordancia_treino": concordancia(rede, treino),
        "concordancia_teste": concordancia(rede, teste) if teste else None}))
    return rede


def _jogar_treino_conv(tarefa):
    theta, especificacoes, semente, assento, recompensa = tarefa
    rede = RedeConv(theta)
    agente = PoliticaConv(rede, random.Random(semente), gravar=True)
    rivais = []
    for especificacao in especificacoes:
        if isinstance(especificacao, str):
            rivais.append(criar(especificacao, random.Random(semente + 7)))
        else:
            rivais.append(PoliticaConv(RedeConv(especificacao), random.Random(semente + 7),
                                       guloso=True))
    ordem = rivais[:assento] + [agente] + rivais[assento:]
    r = jogar(ordem, semente, desempate="pontos", rearranjar=False)
    retorno = retorno_do_jogo(recompensa, r["vencedor"], assento, r["pontos_oficiais"])
    return retorno, agente.trajetoria, r["vencedor"] == assento


def treinar_conv(saida: Path, iteracoes: int, jogos: int, processos: int, semente: int = 0,
                 lr: float = 3e-3, grupo: int = 4, beta: float = 0.01,
                 inicial: Path | None = None, recompensa: str = "pontos",
                 checkpoint_a_cada: int = 250) -> RedeConv:
    """REINFORCE com vantagem relativa ao grupo, igual ao plano 15."""
    saida.mkdir(parents=True, exist_ok=True)
    rng = random.Random(semente)
    rede = RedeConv.carregar(inicial) if inicial else RedeConv.nova(rng)
    heuristicas = list(ESTRATEGIAS)
    antigas: list[list[float]] = []
    passos = 0
    contador = 0
    with Pool(processos) as pool, (saida / "treino.jsonl").open("w") as log:
        for it in range(1, iteracoes + 1):
            tarefas = []
            for _ in range(jogos // grupo):
                sorteio = rng.random()
                if sorteio < 0.5 or (sorteio >= 0.75 and not antigas):
                    especificacao = rng.choice(heuristicas)
                elif sorteio < 0.75:
                    especificacao = list(rede.theta)
                else:
                    especificacao = list(rng.choice(antigas))
                contador += 1
                assento = rng.randrange(2)
                tarefas += [(list(rede.theta), [especificacao], 6_000_000 + contador * 10 + v,
                             assento, recompensa) for v in range(grupo)]
            resultados = pool.map(_jogar_treino_conv, tarefas)

            grad = np.zeros(len(rede.theta))
            decisoes = 0
            for i in range(0, len(resultados), grupo):
                bloco = resultados[i:i + grupo]
                relativas = vantagens_grupo([r[0] for r in bloco])
                for v, (_, trajetoria, _) in enumerate(bloco):
                    for X, indice in trajetoria:
                        grad += np.asarray(rede.gradiente(X, indice, relativas[v], beta))
                        decisoes += 1
            if decisoes:
                passos += 1
                rede.passo(grad / decisoes, passos, lr)
            if it % 25 == 0 or it == 1:
                antigas.append(list(rede.theta))
                antigas[:] = antigas[-8:]
            log.write(json.dumps({
                "iteracao": it, "vitorias": sum(r[2] for r in resultados) / len(resultados),
                "retorno": float(np.mean([r[0] for r in resultados])),
                "decisoes": decisoes, "grupo": grupo, "recompensa": recompensa}) + "\n")
            log.flush()
            if it % checkpoint_a_cada == 0:
                rede.salvar(saida / f"iter_{it:06d}.json")
    rede.salvar(saida / "final.json")
    return rede
