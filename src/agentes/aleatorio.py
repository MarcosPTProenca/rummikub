import random

from src.rl.contratos import Acao


class AgenteAleatorio:
    """Adversário inicial, sem aprendizado."""

    def __init__(self, semente: int | None = None):
        self.aleatorio = random.Random(semente)

    def escolher_acao(self, acoes: list[Acao]) -> Acao:
        if not acoes:
            raise ValueError("Não há ações disponíveis.")
        return self.aleatorio.choice(acoes)
