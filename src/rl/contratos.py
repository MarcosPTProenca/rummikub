from typing import Literal, TypedDict


class DadosMesa(TypedDict, total=False):
    combinacoes: list[list[str]]


class Acao(DadosMesa):
    """Compra, baixa simples ou proposta da mesa final de um turno."""
    tipo: Literal["comprar", "baixar", "mesa"]
    ids_pecas: list[str]


MesaObservada = tuple[tuple[tuple[str, int], ...], ...]


class Observacao(TypedDict):
    mao: tuple[int, ...]
    pecas_adversarios: tuple[int, ...]
    pecas_monte: int
    passagens_consecutivas: int
    aberturas: tuple[bool, ...]
    mesa: MesaObservada


class ResultadoPasso(TypedDict):
    observacao: Observacao
    recompensa: float
    terminado: bool
    truncado: bool
    acoes_validas: list[Acao]


ChaveObservacao = tuple[tuple[int, ...], tuple[int, ...], int, int,
                       tuple[bool, ...], MesaObservada]
# Para mesa: primeiro as peças da mão utilizadas, depois as combinações finais.
ChaveAcao = tuple[str, tuple[tuple[str, int], ...] | MesaObservada]
