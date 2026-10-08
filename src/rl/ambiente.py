from collections.abc import Callable

from src.agentes.aleatorio import AgenteAleatorio
from src.jogo.jogo import Jogo
from .contratos import Acao, Observacao, ResultadoPasso
from .representacao import chave_acao, observar


class AmbienteRL:
    """Uma decisão do agente seguida da resposta de um adversário."""

    def __init__(
        self,
        fabrica_jogo: Callable[[int | None], Jogo],
        adversario: AgenteAleatorio,
        indice_agente: int = 0,
        numero_maximo: int = 13,
        limite_passos: int = 200,
    ):
        if type(indice_agente) is not int or indice_agente not in (0, 1):
            raise ValueError("O índice do agente deve ser 0 ou 1.")
        if type(numero_maximo) is not int or not 1 <= numero_maximo <= 13:
            raise ValueError("O número máximo deve estar entre 1 e 13.")
        if type(limite_passos) is not int or limite_passos < 1:
            raise ValueError("O limite de passos deve ser um inteiro positivo.")
        self.fabrica_jogo = fabrica_jogo
        self.adversario = adversario
        self.indice_agente = indice_agente
        self.numero_maximo = numero_maximo
        self.limite_passos = limite_passos
        self.jogo: Jogo | None = None
        self.passos = 0
        self.truncado = False

    def reiniciar(self, semente: int | None = None) -> tuple[Observacao, list[Acao]]:
        jogo = self.fabrica_jogo(semente)
        if len(jogo.jogadores) != 2 or jogo.terminou:
            raise ValueError("A fábrica deve produzir uma partida ativa de dois jogadores.")
        observar(jogo, self.indice_agente, self.numero_maximo)
        self.jogo = jogo
        self.passos = 0
        self.truncado = False
        if semente is not None:
            self.adversario.aleatorio.seed(semente)
        if jogo.indice_jogador_atual != self.indice_agente:
            self.executar_acao(self.adversario.escolher_acao(jogo.listar_acoes_validas()))
        if jogo.terminou or jogo.indice_jogador_atual != self.indice_agente:
            raise ValueError("A partida deve chegar à primeira decisão do agente.")
        return observar(jogo, self.indice_agente, self.numero_maximo), jogo.listar_acoes_validas()

    def executar_acao(self, acao: Acao) -> None:
        jogo = self.jogo
        if jogo is None or jogo.terminou or self.truncado:
            raise RuntimeError("Reinicie o ambiente antes de executar uma ação.")
        chave_acao(acao, jogo.jogador_atual.mao, jogo.mesa)
        if acao["tipo"] == "comprar":
            jogo.comprar_peca()
        elif acao["tipo"] == "baixar":
            jogo.baixar_combinacao(acao["ids_pecas"])
        else:
            jogo.jogar_turno(acao["combinacoes"])

    def calcular_recompensa(self) -> float:
        if self.jogo is None:
            raise RuntimeError("Reinicie o ambiente antes de calcular a recompensa.")
        if not self.jogo.terminou or self.jogo.vencedor is None:
            return 0.0
        agente = self.jogo.jogadores[self.indice_agente]
        return 1.0 if self.jogo.vencedor.id == agente.id else -1.0

    def passo(self, acao: Acao) -> ResultadoPasso:
        jogo = self.jogo
        if jogo is None or jogo.terminou or self.truncado:
            raise RuntimeError("Reinicie o ambiente antes de executar um passo.")
        if jogo.indice_jogador_atual != self.indice_agente:
            raise RuntimeError("Não é a vez do agente.")
        # A busca automática é uma amostra; o Jogo valida qualquer plano legal
        # atomicamente, inclusive reorganizações não enumeradas pelo agente.
        self.executar_acao(acao)
        if not jogo.terminou:
            self.executar_acao(self.adversario.escolher_acao(jogo.listar_acoes_validas()))
        self.passos += 1
        self.truncado = self.passos >= self.limite_passos and not jogo.terminou
        return {
            "observacao": observar(jogo, self.indice_agente, self.numero_maximo),
            "recompensa": self.calcular_recompensa(),
            "terminado": jogo.terminou,
            "truncado": self.truncado,
            "acoes_validas": jogo.listar_acoes_validas(),
        }
