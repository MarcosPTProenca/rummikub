import random
from copy import copy

from src.jogo.jogo import Jogo
from .arena import executar
from .politicas import ESTRATEGIAS, medidas, oponentes


def clonar(jogo: Jogo) -> Jogo:
    """Cópia barata: peças e combinações são imutáveis, só as listas e os jogadores são duplicados."""
    novo = copy(jogo)
    novo.jogadores = [j.model_copy(update={"mao": list(j.mao)}) for j in jogo.jogadores]
    novo.monte, novo.mesa = list(jogo.monte), list(jogo.mesa)
    return novo


def determinizar(jogo: Jogo, rng: random.Random) -> Jogo:
    """Sorteia uma distribuição plausível das peças que o jogador atual não vê."""
    mundo = clonar(jogo)
    outros = oponentes(mundo)
    oculto = sorted([p for o in outros for p in o.mao] + mundo.monte, key=lambda p: p.id)
    rng.shuffle(oculto)
    inicio = 0
    for o in outros:
        n = len(o.mao)
        o.mao, inicio = oculto[inicio:inicio + n], inicio + n
    mundo.monte = oculto[inicio:]
    return mundo


def rollout(jogo: Jogo, eu: int, rng: random.Random, limite: int = 250) -> float:
    """Joga o resto com joker-saver dos dois lados (sem rearranjos de mesa) e devolve 1/0,5/0 para `eu`."""
    politica = ESTRATEGIAS["poupar_coringa"]
    for _ in range(limite):
        if jogo.terminou:
            break
        executar(jogo, politica(jogo, jogo.listar_acoes_validas(rearranjar=False), rng))
    if not jogo.terminou or jogo.vencedor is None:
        return 0.5
    return float(int(jogo.vencedor.id) == eu)


class PIMC:
    """Monte Carlo com informação perfeita determinizada: avalia poucas jogadas por rollouts em mundos sorteados."""

    def __init__(self, rng: random.Random, determinizacoes: int = 4, candidatas: int = 5, ve_tudo: bool = False):
        self.rng, self.determinizacoes, self.candidatas, self.ve_tudo = rng, determinizacoes, candidatas, ve_tudo

    def mundo(self, jogo, rng):
        """Mundo para os rollouts; `ve_tudo` (teto de informação, não é um jogador legal) usa o mundo verdadeiro."""
        return clonar(jogo) if self.ve_tudo else determinizar(jogo, rng)

    def _candidatas(self, jogo, acoes):
        sorteio = random.Random(0)
        saver = ESTRATEGIAS["poupar_coringa"](jogo, acoes, sorteio)
        jogadas = [a for a in acoes if a["tipo"] != "comprar"]
        melhores = sorted(jogadas, key=lambda a: medidas(jogo, a)[:2], reverse=True)
        escolhidas = [saver]
        for a in melhores + [acoes[-1]]:
            if len(escolhidas) < self.candidatas and all(a is not e for e in escolhidas):
                escolhidas.append(a)
        if all(acoes[-1] is not e for e in escolhidas):
            escolhidas[-1] = acoes[-1]
        return escolhidas

    def __call__(self, jogo, acoes):
        candidatas = self._candidatas(jogo, acoes)
        if len(candidatas) == 1:
            return candidatas[0]
        eu = jogo.indice_jogador_atual
        base = self.rng.getrandbits(32)
        totais = [0.0] * len(candidatas)
        for d in range(self.determinizacoes):
            mundo = self.mundo(jogo, random.Random(base + d))
            for i, acao in enumerate(candidatas):
                copia = clonar(mundo)
                executar(copia, acao)
                totais[i] += rollout(copia, eu, random.Random(base + 1000 + d))
        return candidatas[totais.index(max(totais))]
