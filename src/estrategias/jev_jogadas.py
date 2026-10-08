import random
import urllib.error

from src.jogo.combinacao import Combinacao
from .jev import COR, MODELO, _peca, estado_oponentes
from .politicas import ESTRATEGIAS, medidas

def regras(n: int) -> str:
    return (
    f"Rummikub for {n} players. 106 tiles: numbers 1-13 in 4 colors (two copies of each) plus 2 jokers. "
    "Each player starts with 14 tiles. A run is 3 or more consecutive numbers of one color. "
    "A group is 3 or 4 tiles with the same number and different colors. A joker stands for any tile. "
    "A player's first play must be new sets worth at least 30 points, using only tiles from the hand. "
    "After that, a player may add tiles to sets on the table and rearrange table sets, as long as every "
    "set on the table is valid at the end of the turn. A turn is either playing at least one tile or "
    "drawing one tile. The first player to empty the hand wins. If the pool is empty and " + ("both" if n == 2 else "all") + " players "
    "pass in a row, the player with the lower hand total wins (a joker counts as 30)."
    )


REGRAS = regras(2)


def _nome_conjunto(pecas) -> str:
    combinacao = Combinacao(pecas=list(pecas))
    normais = sorted((p for p in pecas if p.cor != "coringa"), key=lambda p: (p.numero, p.cor))
    coringas = len(pecas) - len(normais)
    if combinacao.eh_grupo():
        cores = sorted(COR[p.cor] for p in normais)
        return f"group of {normais[0].numero}s ({', '.join(cores + ['joker'] * coringas)})"
    numeros = {p.numero for p in normais}
    inicio = min(min(numeros), 14 - len(pecas))
    posicoes = [str(n) if n in numeros else "joker" for n in range(inicio, inicio + len(pecas))]
    return f"{COR[normais[0].cor]} {'-'.join(posicoes)}"


def _ids(pecas):
    return frozenset(p.id for p in pecas)


class JevJogadas:
    """Jev escolhe a jogada concreta entre todas as ações legais; o código só executa ações já validadas."""

    def __init__(self, transporte, rng, orcamento, confianca_minima: float = 0.0):
        self.transporte, self.rng, self.orcamento = transporte, rng, orcamento
        self.confianca_minima = confianca_minima
        self.contagem = {"chamadas": 0, "fallbacks": 0, "tokens": 0, "falhas": 0, "sem_escolha": 0}
        self.confiancas: list[float] = []
        self.concordancia = {"igual_max_pecas": 0, "igual_poupar_coringa": 0, "comprou": 0, "coringas_jogados": 0}

    def _descrever(self, jogo, acao, por_id) -> str:
        eu = jogo.jogador_atual
        if acao["tipo"] == "comprar":
            if jogo.monte:
                return f"Draw a tile from the pool and end the turn (hand grows to {len(eu.mao) + 1} tiles)."
            return "Draw a tile: the pool is empty, so this is a pass; my hand stays at " f"{len(eu.mao)} tiles."
        n, soma, coringas = medidas(jogo, acao)
        da_mao = ", ".join(sorted(_peca(por_id[i]) for i in acao["ids_pecas"]))
        restam = len(eu.mao) - n
        partes = [f"Play {n} tile(s) (tile numbers sum to {soma}, {coringas} joker(s) used); tiles from hand: {da_mao}; "
                  f"{restam} hand tiles left" + ("; this empties my hand and wins" if restam == 0 else "")]
        atual = {_ids(c.pecas): c for c in jogo.mesa}
        if acao["tipo"] == "baixar":
            novas = [Combinacao(pecas=[por_id[i] for i in acao["ids_pecas"]])]
            sai = []
        else:
            layout = [Combinacao(pecas=[por_id[i] for i in c]) for c in acao["combinacoes"]]
            novas = [c for c in layout if _ids(c.pecas) not in atual]
            sai = [c for k, c in atual.items() if k not in {_ids(c.pecas) for c in layout}]
        if sai:
            partes.append("table sets removed: " + "; ".join(_nome_conjunto(c.pecas) for c in sai))
        partes.append("table sets added: " + "; ".join(_nome_conjunto(c.pecas) for c in novas))
        return "; ".join(partes) + "."

    def _corpo(self, jogo, acoes):
        eu = jogo.jogador_atual
        por_id = {p.id: p for p in eu.mao} | {p.id: p for c in jogo.mesa for p in c.pecas}
        ordem = list(acoes)
        self.rng.shuffle(ordem)
        chaves = {f"m{i + 1:03d}": acao for i, acao in enumerate(ordem)}
        criterios = {chave: self._descrever(jogo, acao, por_id) for chave, acao in chaves.items()}
        corpo = {
            "model": MODELO,
            "state": {
                "rules": regras(len(jogo.jogadores)),
                "my_hand": sorted(_peca(p) for p in eu.mao),
                "hand_points_if_game_ends": sum(30 if p.cor == "coringa" else p.numero for p in eu.mao),
                "table_sets": [[_peca(p) for p in c.pecas] for c in jogo.mesa],
                "i_have_made_my_initial_30_point_meld": eu.abriu,
                **estado_oponentes(jogo),
                "tiles_left_in_pool": len(jogo.monte),
            },
            "questions": {"jogada": {
                "type": "choice",
                "instructions": "Each candidate is one complete turn I can play now. Which candidate gives me "
                                "the best chance of winning this game?",
                "criteria": criterios,
            }},
        }
        return corpo, chaves

    def _fallback(self, jogo, acoes):
        self.contagem["fallbacks"] += 1
        return ESTRATEGIAS["max_pecas"](jogo, acoes, self.rng)

    def _registrar(self, jogo, acoes, acao):
        c = self.concordancia
        c["comprou"] += acao["tipo"] == "comprar"
        c["coringas_jogados"] += medidas(jogo, acao)[2]
        c["igual_max_pecas"] += acao is ESTRATEGIAS["max_pecas"](jogo, acoes, random.Random(0))
        c["igual_poupar_coringa"] += acao is ESTRATEGIAS["poupar_coringa"](jogo, acoes, random.Random(0))

    def __call__(self, jogo, acoes):
        if len(acoes) == 1:
            self.contagem["sem_escolha"] += 1
            return acoes[0]
        if not self.orcamento.disponivel():
            return self._fallback(jogo, acoes)
        corpo, chaves = self._corpo(jogo, acoes)
        try:
            resposta = self.transporte(corpo)
        except (urllib.error.URLError, TimeoutError):
            self.contagem["falhas"] += 1
            return self._fallback(jogo, acoes)
        tokens = resposta["usage"]["input_tokens"]
        self.orcamento.somar(tokens)
        self.contagem["chamadas"] += 1
        self.contagem["tokens"] += tokens
        resposta = resposta["answers"]["jogada"]
        if resposta["choice"] not in chaves or resposta["confidence"] < self.confianca_minima:
            return self._fallback(jogo, acoes)
        self.confiancas.append(round(resposta["confidence"], 3))
        acao = chaves[resposta["choice"]]
        self._registrar(jogo, acoes, acao)
        return acao
