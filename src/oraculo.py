"""Oráculo offline: reenumera as jogadas de estados saturados com limites maiores."""
import argparse
import copy
import json
import random
import signal
import time
from pathlib import Path

from src.estrategias.arena import jogar


def avaliar_estado(jogo, perfis: dict, tempo_max: float = 30.0) -> dict:
    def estourou(*_):
        raise TimeoutError

    antigo = signal.signal(signal.SIGALRM, estourou)
    saida = {}
    try:
        for nome, limites in perfis.items():
            sat = {}
            inicio = time.perf_counter()
            try:
                signal.setitimer(signal.ITIMER_REAL, max(tempo_max, 1e-6))
                acoes = jogo.listar_acoes_validas(True, limites=limites, saturacao=sat)
                saida[nome] = {"acoes": len(acoes), "max_pecas": max(len(a["ids_pecas"]) for a in acoes),
                               "saturou": any(sat.values()), "estourou": False}
            except TimeoutError:
                saida[nome] = {"acoes": None, "max_pecas": None, "saturou": None, "estourou": True}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            saida[nome]["segundos"] = time.perf_counter() - inicio
    finally:
        signal.signal(signal.SIGALRM, antigo)
    return saida


def amostrar_estados(partidas, por_partida: int) -> list[dict]:
    from src.experimentos import construir

    estados = []
    for a, b, semente, assento_a in partidas:
        sorteio = random.Random(semente)
        escolhidos, vistos = [], 0

        def espiar(politica):
            def decidir(jogo, acoes):
                nonlocal vistos
                sat = {}
                jogo.listar_acoes_validas(True, saturacao=sat)
                if sat["planos"] or sat["nos"]:
                    vistos += 1
                    registro = {"semente": semente, "a": a, "b": b, "assento_a": assento_a,
                                "turno": jogo.indice_jogador_atual, "jogo": copy.deepcopy(jogo)}
                    if len(escolhidos) < por_partida:
                        escolhidos.append(registro)
                    elif (k := sorteio.randrange(vistos)) < por_partida:
                        escolhidos[k] = registro
                return politica(jogo, acoes)
            return decidir

        jogadores = [None, None]
        jogadores[assento_a] = espiar(construir(a, semente, assento_a, None))
        jogadores[1 - assento_a] = espiar(construir(b, semente, 1 - assento_a, None))
        jogar(jogadores, semente)
        estados.extend(escolhidos)
    return estados


def main() -> None:
    from src.experimentos import PERFIS_LIMITES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origem", type=Path, default=Path("resultados/limites.jsonl"))
    parser.add_argument("--partidas", type=int, default=60)
    parser.add_argument("--por-partida", type=int, default=8)
    parser.add_argument("--tempo-max", type=float, default=30.0)
    parser.add_argument("--saida", type=Path, default=Path("resultados/oraculo.jsonl"))
    args = parser.parse_args()
    registros = [json.loads(l) for l in args.origem.read_text().splitlines()]
    candidatas = [r for r in registros if r["saturacao"]["planos"] or r["saturacao"]["nos"]]
    escolhidas = random.Random(0).sample(candidatas, min(args.partidas, len(candidatas)))
    partidas = [(r["a"], r["b"], r["semente"], r["assento_a"]) for r in escolhidas]
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w") as arquivo:
        for estado in amostrar_estados(partidas, args.por_partida):
            r = avaliar_estado(estado.pop("jogo"), PERFIS_LIMITES, args.tempo_max)
            arquivo.write(json.dumps({**estado, "perfis": r}) + "\n")
            arquivo.flush()


if __name__ == "__main__":
    main()
