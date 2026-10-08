from typing import TypedDict

from src.agentes.q_learning import AgenteQLearning
from .ambiente import AmbienteRL
from .representacao import agrupar_acoes, chave_observacao


class ResumoEpisodio(TypedDict):
    retorno: float
    passos: int
    terminado: bool
    truncado: bool


def treinar(
    ambiente: AmbienteRL,
    agente: AgenteQLearning,
    episodios: int,
    semente_inicial: int = 0,
) -> list[ResumoEpisodio]:
    """Aprender a cada decisão; truncamentos preservam o valor futuro."""
    if type(episodios) is not int or episodios < 1:
        raise ValueError("O número de episódios deve ser um inteiro positivo.")
    resumos: list[ResumoEpisodio] = []
    for episodio in range(episodios):
        observacao, acoes = ambiente.reiniciar(semente_inicial + episodio)
        jogo = ambiente.jogo
        assert jogo is not None
        chave = chave_observacao(observacao)
        agrupadas = agrupar_acoes(acoes, jogo.jogadores[ambiente.indice_agente].mao, jogo.mesa)
        retorno = 0.0
        while True:
            escolhida = agente.escolher_acao(chave, list(agrupadas), explorar=True)
            resultado = ambiente.passo(agrupadas[escolhida])
            proxima = chave_observacao(resultado["observacao"])
            futuras = agrupar_acoes(
                resultado["acoes_validas"], jogo.jogadores[ambiente.indice_agente].mao, jogo.mesa
            )
            agente.atualizar(chave, escolhida, resultado["recompensa"],
                             proxima, list(futuras), resultado["terminado"])
            retorno += resultado["recompensa"]
            if resultado["terminado"] or resultado["truncado"]:
                resumos.append({
                    "retorno": retorno,
                    "passos": ambiente.passos,
                    "terminado": resultado["terminado"],
                    "truncado": resultado["truncado"],
                })
                break
            chave, agrupadas = proxima, futuras
    return resumos
