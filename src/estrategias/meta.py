import random

from src.agentes.q_learning import AgenteQLearning
from .arena import jogar
from .politicas import COMPRAR, ESTRATEGIAS, criar, medidas, oponentes

OPCOES = ("max_pecas", "max_pontos", "poupar_coringa", "so_baixar", "minimo", "cauteloso", "comprar")
POLITICAS_META = {**ESTRATEGIAS, "comprar": COMPRAR}


def _faixa(valor: int, cortes: tuple[int, ...]) -> int:
    return sum(valor > c for c in cortes)


def estado_meta(jogo, acoes) -> tuple:
    """Resumo pequeno da decisão: o Q aprende qual estratégia usar em cada situação."""
    eu, outros = jogo.jogador_atual, oponentes(jogo)
    n = [medidas(jogo, a)[0] for a in acoes if a["tipo"] != "comprar"]
    return (eu.abriu, all(o.abriu for o in outros), _faixa(len(eu.mao), (4, 8, 12)),
            _faixa(min(len(o.mao) for o in outros), (5, 10)),
            _faixa(max(n, default=0), (0, 2, 4)), _faixa(len(jogo.monte), (0, 30)),
            len(eu.mao) in n, any(p.cor == "coringa" for p in eu.mao))


class MetaQ:
    """Política que escolhe uma estratégia da lista a cada decisão, via Q-learning."""

    def __init__(self, agente: AgenteQLearning, rng: random.Random, treino: bool = False):
        self.agente, self.rng, self.treino = agente, rng, treino
        self.trajetoria: list[tuple] = []

    def escolher(self, jogo, acoes) -> str:
        estado = estado_meta(jogo, acoes)
        opcao = self.agente.escolher_acao(estado, list(OPCOES), explorar=self.treino)
        self.trajetoria.append((estado, opcao))
        return opcao

    def __call__(self, jogo, acoes):
        return POLITICAS_META[self.escolher(jogo, acoes)](jogo, acoes, self.rng)


def treinar_meta(agente: AgenteQLearning, episodios: int, oponentes: list[str],
                 semente_inicial: int = 0, epsilon: tuple[float, float] = (0.3, 0.05),
                 desempate: str = "pontos", jogadores: int = 2) -> list[float]:
    """Treinar contra oponentes sorteados do conjunto, alternando assentos; devolve o retorno de cada partida."""
    retornos = []
    for ep in range(episodios):
        agente.epsilon = epsilon[0] + (epsilon[1] - epsilon[0]) * ep / max(episodios - 1, 1)
        semente = semente_inicial + ep
        sorteio = random.Random(semente)
        assento = sorteio.randrange(jogadores)
        meta = MetaQ(agente, random.Random(semente), treino=True)
        adversarios = [criar(oponentes[sorteio.randrange(len(oponentes))], random.Random(semente + 1 + k))
                       for k in range(jogadores - 1)]
        politicas = adversarios[:assento] + [meta] + adversarios[assento:]
        vencedor = jogar(politicas, semente, desempate=desempate)["vencedor"]
        retorno = 0.0 if vencedor is None else (1.0 if vencedor == assento else -1.0)
        passos = meta.trajetoria
        for i in reversed(range(len(passos))):
            ultimo = i == len(passos) - 1
            proximo = passos[i + 1][0] if not ultimo else passos[i][0]
            agente.atualizar(passos[i][0], passos[i][1], retorno if ultimo else 0.0,
                             proximo, list(OPCOES), ultimo)
        retornos.append(retorno)
    return retornos
