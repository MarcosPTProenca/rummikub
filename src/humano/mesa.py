"""Mesa de um humano contra os agentes do estudo.

O frontend não conhece regra de Rummikub: ele arrasta peças e manda a mesa
final inteira, como `Jogo.jogar_turno` já espera. A validação é a do motor, a
mesma que os agentes enfrentam, então a partida do humano é comparável às dos
experimentos.
"""
import json
import random
from pathlib import Path

from src.estrategias import arena, politicas
from src.jogo.jogo import Jogo
from src.jogo.pontuacao import pontos_oficiais, valor_mao

# As sete heurísticas do artigo, com a taxa de vitória do torneio original
# (resultados/resumo.json) como dica de dificuldade para a interface.
OPONENTES = {
    "poupar_coringa": {"nome": "joker-saver", "vitorias": 0.726,
                       "sobre": "Guarda o coringa para a jogada final. O campeão do estudo."},
    "max_pecas": {"nome": "max-tiles", "vitorias": 0.676,
                  "sobre": "Baixa o máximo de peças que consegue por turno."},
    "max_pontos": {"nome": "max-points", "vitorias": 0.672,
                   "sobre": "Baixa a jogada de maior soma de pontos."},
    "minimo": {"nome": "min-play", "vitorias": 0.555,
               "sobre": "Baixa o mínimo possível, guardando peças na mão."},
    "aleatorio": {"nome": "random", "vitorias": 0.461,
                  "sobre": "Escolhe ao acaso entre as jogadas legais."},
    "so_baixar": {"nome": "no-rearrange", "vitorias": 0.271,
                  "sobre": "Nunca reorganiza a mesa; só baixa combinações novas."},
    "cauteloso": {"nome": "cautious", "vitorias": 0.140,
                  "sobre": "Compra sempre que a mão ainda cabe no limite."},
}


class Partida:
    """Uma partida com o humano no assento 0 e um agente em cada assento seguinte."""

    def __init__(self, oponentes: list[str], semente: int | None = None,
                 registro: Path | None = None, rearranjar: bool = True):
        desconhecidos = [o for o in oponentes if o not in OPONENTES]
        if desconhecidos:
            raise ValueError(f"Oponente desconhecido: {', '.join(desconhecidos)}.")
        if not 1 <= len(oponentes) <= 9:
            raise ValueError("A mesa aceita de 1 a 9 oponentes.")

        self.humano = 0
        self.oponentes = list(oponentes)
        self.semente = semente if semente is not None else random.randrange(10**9)
        self.registro = Path(registro) if registro else None
        self.rearranjar = rearranjar
        self.jogo = Jogo(semente=self.semente, quantidade_jogadores=len(oponentes) + 1)
        # Semente própria para os desempates das políticas: a do jogo já foi gasta nas mãos.
        rng = random.Random(self.semente + 1)
        self.politicas = [None] + [politicas.criar(o, rng) for o in self.oponentes]
        self.lances: list[dict] = []
        self.replay: list[dict] = []
        self.decisoes = 0
        self.gravado = False

    # --- leitura -------------------------------------------------------

    def _peca(self, p) -> dict:
        return {"id": p.id, "cor": p.cor, "numero": p.numero}

    def nome(self, assento: int) -> str:
        return "você" if assento == self.humano else OPONENTES[self.oponentes[assento - 1]]["nome"]

    def estado(self) -> dict:
        jogo = self.jogo
        return {
            "mao": [self._peca(p) for p in jogo.jogadores[self.humano].mao],
            "mesa": [[self._peca(p) for p in c.pecas] for c in jogo.mesa],
            "oponentes": [{"assento": i, "nome": self.nome(i), "pecas": len(j.mao), "abriu": j.abriu}
                          for i, j in enumerate(jogo.jogadores) if i != self.humano],
            "monte": len(jogo.monte),
            "vez": jogo.indice_jogador_atual,
            "abriu": jogo.jogadores[self.humano].abriu,
            "terminou": jogo.terminou,
            "vencedor": int(jogo.vencedor.id) if jogo.vencedor else None,
            "pontos": pontos_oficiais(jogo) if jogo.terminou else None,
            "valor_mao": valor_mao(jogo.jogadores[self.humano].mao),
            "lances": self.lances[-12:],
            "replay": self.replay,
            "decisoes": self.decisoes,
        }

    def sugestao(self) -> list[list[str]] | None:
        """A mesa que o joker-saver jogaria com a mão do humano, ou None se só couber comprar.

        Não é exposta pelo servidor de propósito: entregar a jogada do agente
        esvazia o sentido de jogar contra ele. Serve aos testes, que precisam
        de uma jogada legal para percorrer uma partida inteira.
        """
        acoes = self.jogo.listar_acoes_validas(self.rearranjar)
        acao = politicas.criar("poupar_coringa", random.Random(0))(self.jogo, acoes)
        if acao["tipo"] == "comprar":
            return None
        if acao["tipo"] == "baixar":
            return [[p.id for p in c.pecas] for c in self.jogo.mesa] + [acao["ids_pecas"]]
        return acao["combinacoes"]

    # --- escrita -------------------------------------------------------

    def _minha_vez(self) -> dict | None:
        if self.jogo.terminou:
            return {"ok": False, "erro": "A partida já terminou."}
        if self.jogo.indice_jogador_atual != self.humano:
            return {"ok": False, "erro": "Não é a sua vez."}
        return None

    def jogar(self, combinacoes: list[list[str]]) -> dict:
        """Aplicar a mesa final proposta pelo humano. Recusa devolve a razão do motor."""
        if (erro := self._minha_vez()):
            return erro
        antes = {p.id for p in self.jogo.jogadores[self.humano].mao}
        try:
            self.jogo.jogar_turno(combinacoes)
        except ValueError as e:
            return {"ok": False, "erro": str(e)}
        baixadas = antes - {p.id for p in self.jogo.jogadores[self.humano].mao}
        self._anotar(self.humano, "jogar", len(baixadas))
        self._rodar_agentes()
        return {"ok": True}

    def comprar(self) -> dict:
        if (erro := self._minha_vez()):
            return erro
        self.jogo.comprar_peca()
        self._anotar(self.humano, "comprar", 1)
        self._rodar_agentes()
        return {"ok": True}

    def _anotar(self, assento: int, tipo: str, pecas: int) -> None:
        self.decisoes += 1
        self.lances.append({"assento": assento, "nome": self.nome(assento), "tipo": tipo,
                            "pecas": pecas, "mao": len(self.jogo.jogadores[assento].mao)})

    def _rodar_agentes(self) -> None:
        """Deixar os agentes jogarem até a vez voltar ao humano ou a partida acabar.

        Cada lance guarda a mesa como ficou naquele instante, para a interface
        poder reproduzir a sequência em vez de trocar a mesa de uma vez."""
        limite = 100 * len(self.jogo.jogadores)
        self.replay = []
        while not self.jogo.terminou and self.jogo.indice_jogador_atual != self.humano:
            if self.decisoes > limite:  # a mesma salvaguarda da arena, contra partida travada
                break
            assento = self.jogo.indice_jogador_atual
            acoes = self.jogo.listar_acoes_validas(self.rearranjar)
            acao = self.politicas[assento](self.jogo, acoes)
            mao_antes = len(self.jogo.jogadores[assento].mao)
            antes = {p.id for c in self.jogo.mesa for p in c.pecas}
            arena.executar(self.jogo, acao)
            depois = len(self.jogo.jogadores[assento].mao)
            tipo = "comprar" if acao["tipo"] == "comprar" else "jogar"
            # A peça comprada é informação oculta; só o que foi à mesa aparece.
            novos = [p.id for c in self.jogo.mesa for p in c.pecas if p.id not in antes]
            self.replay.append({
                "assento": assento,
                "nome": self.nome(assento),
                "tipo": tipo,
                "ids": novos if tipo == "jogar" else [],
                "mesa": [[self._peca(p) for p in c.pecas] for c in self.jogo.mesa],
                "mao": depois,
            })
            self._anotar(assento, tipo, abs(mao_antes - depois))
        if self.jogo.terminou:
            self.encerrar()

    def encerrar(self) -> None:
        """Gravar a partida, uma única vez, se houver destino configurado."""
        if self.gravado or not self.registro:
            self.gravado = True
            return
        self.gravado = True
        self.registro.parent.mkdir(parents=True, exist_ok=True)
        linha = {
            "semente": self.semente,
            "oponentes": self.oponentes,
            "vencedor": int(self.jogo.vencedor.id) if self.jogo.vencedor else None,
            "humano_venceu": bool(self.jogo.vencedor) and int(self.jogo.vencedor.id) == self.humano,
            "pontos_oficiais": pontos_oficiais(self.jogo),
            "maos": [len(j.mao) for j in self.jogo.jogadores],
            "coringas_mao": [sum(p.cor == "coringa" for p in j.mao) for j in self.jogo.jogadores],
            "decisoes": self.decisoes,
            "terminou": self.jogo.terminou,
            "rearranjar": self.rearranjar,
        }
        with self.registro.open("a") as f:
            f.write(json.dumps(linha) + "\n")
