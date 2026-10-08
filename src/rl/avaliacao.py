from typing import TypedDict

from src.agentes.q_learning import AgenteQLearning
from .ambiente import AmbienteRL
from .representacao import agrupar_acoes, chave_observacao


class Metricas(TypedDict):
    vitorias: int
    derrotas: int
    empates: int
    truncados: int
    taxa_vitoria: float
    media_passos: float


def avaliar(
    ambiente: AmbienteRL,
    agente: AgenteQLearning,
    episodios: int,
    semente_inicial: int = 100_000,
) -> Metricas:
    """Medir a política greedy, sem atualizar Q nem alterar epsilon."""
    if type(episodios) is not int or episodios < 1:
        raise ValueError("O número de episódios deve ser um inteiro positivo.")
    metricas: Metricas = {
        "vitorias": 0, "derrotas": 0, "empates": 0, "truncados": 0,
        "taxa_vitoria": 0.0, "media_passos": 0.0,
    }
    passos = 0
    for episodio in range(episodios):
        observacao, acoes = ambiente.reiniciar(semente_inicial + episodio)
        jogo = ambiente.jogo
        assert jogo is not None
        while True:
            agrupadas = agrupar_acoes(acoes, jogo.jogadores[ambiente.indice_agente].mao, jogo.mesa)
            escolhida = agente.escolher_acao(chave_observacao(observacao),
                                            list(agrupadas), explorar=False)
            resultado = ambiente.passo(agrupadas[escolhida])
            if resultado["terminado"] or resultado["truncado"]:
                passos += ambiente.passos
                if resultado["truncado"]:
                    metricas["truncados"] += 1
                elif resultado["recompensa"] > 0:
                    metricas["vitorias"] += 1
                elif resultado["recompensa"] < 0:
                    metricas["derrotas"] += 1
                else:
                    metricas["empates"] += 1
                break
            observacao, acoes = resultado["observacao"], resultado["acoes_validas"]
    metricas["taxa_vitoria"] = metricas["vitorias"] / episodios
    metricas["media_passos"] = passos / episodios
    return metricas
