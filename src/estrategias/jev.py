import json
import multiprocessing
import random
import time
import urllib.error
import urllib.request

from .meta import OPCOES, POLITICAS_META
from .politicas import medidas, oponentes

MODELO = "jev-1.13.0"
USD_POR_TOKEN = 0.042 / 1_000_000
URL = "https://api.typesafe.ai/v1/systemone"


class Orcamento:
    """Gasto em dólares compartilhado entre processos; recusa chamadas acima do limite."""

    def __init__(self, limite: float):
        self.limite = limite
        self._gasto = multiprocessing.Value("d", 0.0)

    @property
    def gasto(self) -> float:
        return self._gasto.value

    def disponivel(self) -> bool:
        return self.gasto < self.limite

    def somar(self, tokens: int) -> None:
        with self._gasto.get_lock():
            self._gasto.value += tokens * USD_POR_TOKEN


def transporte_http(chave: str, tentativas: int = 6):
    def enviar(corpo: dict) -> dict:
        pedido = urllib.request.Request(
            URL, json.dumps(corpo).encode(),
            {"Authorization": f"Bearer {chave}", "Content-Type": "application/json"})
        for tentativa in range(tentativas):
            try:
                with urllib.request.urlopen(pedido, timeout=60) as resposta:
                    return json.load(resposta)
            except urllib.error.HTTPError as erro:
                if erro.code not in (429, 529) or tentativa == tentativas - 1:
                    raise
            except urllib.error.URLError:
                if tentativa == tentativas - 1:
                    raise
            time.sleep(min(2 ** tentativa, 20))
        raise RuntimeError("inalcançável")
    return enviar


COR = {"azul": "blue", "vermelho": "red", "verde": "green", "amarelo": "yellow", "coringa": "joker"}


def _peca(peca) -> str:
    return "joker" if peca.cor == "coringa" else f"{COR[peca.cor]} {peca.numero}"


def estado_oponentes(jogo) -> dict:
    """Com 2 jogadores mantém o formato original do payload; com mais, lista cada oponente."""
    outros = oponentes(jogo)
    if len(outros) == 1:
        return {"opponent_tiles_in_hand": len(outros[0].mao), "opponent_has_made_initial_meld": outros[0].abriu}
    return {"opponents": [{"tiles_in_hand": len(o.mao), "has_made_initial_meld": o.abriu} for o in outros]}


def _descricao(jogo, acao) -> str:
    n, soma, coringas = medidas(jogo, acao)
    if acao["tipo"] == "comprar":
        return "Draw a tile from the pool and end the turn."
    extra = "rearranging the table" if acao["tipo"] == "mesa" else "as new sets"
    return f"Play {n} tile(s) from the hand {extra} (tile sum {soma}, {coringas} joker(s) used)."


DESCRICOES = {
    "max_pecas": "Play as many tiles as possible this turn.",
    "max_pontos": "Play the highest-value tiles this turn.",
    "poupar_coringa": "Play as many tiles as possible but keep jokers unless it wins the game.",
    "so_baixar": "Only lay down new sets; never touch the sets already on the table.",
    "minimo": "Play the smallest possible move and keep the rest of the hand.",
    "cauteloso": "Play only when a move uses at least 4 tiles (or wins); otherwise draw.",
    "comprar": "Draw a tile now.",
}


class JevMeta:
    """Jev escolhe a estratégia a cada decisão; abaixo da confiança mínima, usa max_pecas."""

    def __init__(self, transporte, rng, orcamento: Orcamento, confianca_minima: float = 0.3):
        self.transporte, self.rng, self.orcamento = transporte, rng, orcamento
        self.confianca_minima = confianca_minima
        self.contagem = {"chamadas": 0, "fallbacks": 0, "tokens": 0, "falhas": 0}
        self.escolhas = {o: 0 for o in OPCOES}

    def _corpo(self, jogo, acoes) -> dict:
        eu = jogo.jogador_atual
        criterios = {}
        for opcao in OPCOES:
            acao = POLITICAS_META[opcao](jogo, acoes, random.Random(0))
            criterios[opcao] = f"{DESCRICOES[opcao]} Right now this would: {_descricao(jogo, acao)}"
        return {
            "model": MODELO,
            "state": {
                "game": f"Rummikub, {len(jogo.jogadores)} players, 106 tiles with 2 jokers; first to empty the hand wins.",
                "my_hand": sorted(_peca(p) for p in eu.mao),
                "table_sets": [[_peca(p) for p in c.pecas] for c in jogo.mesa],
                "i_have_made_my_initial_30_point_meld": eu.abriu,
                **estado_oponentes(jogo),
                "tiles_left_in_pool": len(jogo.monte),
            },
            "questions": {"estrategia": {
                "type": "choice",
                "instructions": "Which option gives me the best chance of winning this game?",
                "criteria": criterios,
            }},
        }

    def _fallback(self, jogo, acoes):
        self.contagem["fallbacks"] += 1
        self.escolhas["max_pecas"] += 1
        return POLITICAS_META["max_pecas"](jogo, acoes, self.rng)

    def __call__(self, jogo, acoes):
        if not self.orcamento.disponivel():
            return self._fallback(jogo, acoes)
        try:
            resposta = self.transporte(self._corpo(jogo, acoes))
        except (urllib.error.URLError, TimeoutError):
            self.contagem["falhas"] += 1
            return self._fallback(jogo, acoes)
        tokens = resposta["usage"]["input_tokens"]
        self.orcamento.somar(tokens)
        self.contagem["chamadas"] += 1
        self.contagem["tokens"] += tokens
        resposta = resposta["answers"]["estrategia"]
        if resposta["confidence"] < self.confianca_minima:
            return self._fallback(jogo, acoes)
        self.escolhas[resposta["choice"]] += 1
        return POLITICAS_META[resposta["choice"]](jogo, acoes, self.rng)
