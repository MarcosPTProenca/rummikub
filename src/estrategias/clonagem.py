import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .arena import jogar
from .politicas import criar
from .rede import Rede, atributos


class _Gravador:
    """Embrulha uma política e grava (atributos das ações, índice da escolhida); ações demais são subamostradas."""

    def __init__(self, politica, rng: random.Random, maximo: int):
        self.politica, self.rng, self.maximo = politica, rng, maximo
        self.amostras: list[tuple[list[list[float]], int]] = []

    def __call__(self, jogo, acoes):
        acao = self.politica(jogo, acoes)
        if len(acoes) > 1:
            escolhido = next(i for i, a in enumerate(acoes) if a is acao)
            indices = [escolhido] + self.rng.sample([i for i in range(len(acoes)) if i != escolhido],
                                                    min(len(acoes), self.maximo) - 1)
            indices.sort()
            self.amostras.append(([atributos(jogo, acoes[i]) for i in indices], indices.index(escolhido)))
        return acao


def _partida(tarefa):
    demonstrador, oponentes, semente, jogadores, maximo, desempate = tarefa
    assento = semente % jogadores
    gravador = _Gravador(criar(demonstrador, random.Random(semente)), random.Random(semente), maximo)
    rivais = [criar(oponentes[(semente + k) % len(oponentes)], random.Random(semente + 1 + k)) for k in range(jogadores - 1)]
    jogar(rivais[:assento] + [gravador] + rivais[assento:], semente, desempate=desempate)
    return gravador.amostras


def coletar(demonstrador: str, oponentes: list[str], sementes: int, inicio: int = 6_000_000, jogadores: int = 2,
            maximo_acoes: int = 32, desempate: str = "pontos", processos: int = 1) -> list:
    """Decisões do demonstrador em partidas completas (com rearranjos) contra oponentes heurísticos."""
    tarefas = [(demonstrador, oponentes, inicio + s, jogadores, maximo_acoes, desempate) for s in range(sementes)]
    if processos > 1:
        with Pool(processos) as pool:
            partidas = pool.map(_partida, tarefas)
    else:
        partidas = [_partida(t) for t in tarefas]
    return [a for partida in partidas for a in partida]


def concordancia(rede: Rede, amostras) -> float:
    acertos = sum(max(range(len(X)), key=rede.pontuacoes(X).__getitem__) == i for X, i in amostras)
    return acertos / len(amostras)


def treinar_clone(rede: Rede, amostras, epocas: int, lr: float, rng: random.Random, lote: int = 64) -> None:
    """Máxima verossimilhança da escolha do demonstrador: o gradiente do REINFORCE com vantagem 1 e sem entropia."""
    passo = 0
    for _ in range(epocas):
        ordem = list(range(len(amostras)))
        rng.shuffle(ordem)
        for inicio in range(0, len(ordem), lote):
            grupo = ordem[inicio:inicio + lote]
            grad = np.zeros(len(rede.theta))
            for k in grupo:
                X, indice = amostras[k]
                grad += rede.gradiente(X, indice, 1.0)
            passo += 1
            rede.passo(grad / len(grupo), passo, lr)


def clonar(saida: Path, demonstrador: str, oponentes: list[str], partidas: int, epocas: int, processos: int,
           ocultos: int = 16, semente: int = 0, lr: float = 3e-3, jogadores: int = 2, desempate: str = "pontos") -> Rede:
    saida.mkdir(parents=True, exist_ok=True)
    amostras = coletar(demonstrador, oponentes, partidas, jogadores=jogadores, desempate=desempate, processos=processos)
    corte = int(len(amostras) * 0.9)
    treino, teste = amostras[:corte], amostras[corte:]
    rng = random.Random(semente)
    rede = Rede.nova(rng, ocultos)
    treinar_clone(rede, treino, epocas, lr, rng)
    rede.salvar(saida / "final.json", iteracao=0, clonada_de=demonstrador)
    (saida / "clonagem.json").write_text(json.dumps({
        "demonstrador": demonstrador, "partidas": partidas, "amostras": len(amostras), "epocas": epocas, "lr": lr,
        "jogadores": jogadores, "concordancia_treino": concordancia(rede, treino),
        "concordancia_teste": concordancia(rede, teste) if teste else None}))
    return rede
