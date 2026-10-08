from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from src.jogo.combinacao import Combinacao
from .contratos import Acao, ChaveAcao, ChaveObservacao, Observacao


CORES = ("azul", "vermelho", "verde", "amarelo")


def observar(jogo: Jogo, indice_agente: int, numero_maximo: int = 13) -> Observacao:
    """Observar uma partida com dois a quatro jogadores, sem revelar outras mãos."""
    quantidade_jogadores = len(jogo.jogadores)
    if not 2 <= quantidade_jogadores <= 4:
        raise ValueError("O ambiente exige entre dois e quatro jogadores.")
    if not 0 <= indice_agente < quantidade_jogadores:
        raise ValueError("O índice do agente é inválido.")
    if not 1 <= numero_maximo <= 13:
        raise ValueError("O número máximo deve estar entre 1 e 13.")
    contagens = {(cor, numero): 0 for cor in CORES for numero in range(1, numero_maximo + 1)}
    # A última posição da mão conta coringas, independentemente do número máximo.
    contagens[("coringa", 0)] = 0
    for peca in jogo.jogadores[indice_agente].mao:
        tipo = (peca.cor, peca.numero)
        if tipo not in contagens:
            raise ValueError(f"Peça fora da configuração: {peca.cor} {peca.numero}.")
        contagens[tipo] += 1
    return {
        "mao": tuple(contagens.values()),
        "pecas_adversarios": tuple(
            len(jogo.jogadores[(indice_agente + deslocamento) % quantidade_jogadores].mao)
            for deslocamento in range(1, quantidade_jogadores)
        ),
        "pecas_monte": len(jogo.monte),
        "passagens_consecutivas": jogo.passagens_consecutivas,
        "aberturas": tuple(jogo.jogadores[(indice_agente + i) % quantidade_jogadores].abriu
                           for i in range(quantidade_jogadores)),
        "mesa": tuple(sorted(tuple(sorted((p.cor, p.numero) for p in c.pecas)) for c in jogo.mesa)),
    }


def chave_observacao(observacao: Observacao) -> ChaveObservacao:
    return (observacao["mao"], observacao["pecas_adversarios"],
            observacao["pecas_monte"], observacao["passagens_consecutivas"],
            observacao["aberturas"], observacao["mesa"])


def chave_acao(acao: Acao, mao: list[Peca], mesa: list[Combinacao] | None = None) -> ChaveAcao:
    """Representar peças utilizadas e partição final sem depender dos IDs físicos."""
    if not isinstance(acao, dict) or acao.get("tipo") not in ("comprar", "baixar", "mesa"):
        raise ValueError("Tipo de ação inválido.")
    ids = acao.get("ids_pecas")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("Os IDs das peças devem ser uma lista de strings.")
    if len(ids) != len(set(ids)):
        raise ValueError("Uma peça não pode ser selecionada duas vezes.")
    if acao["tipo"] == "comprar":
        if ids or "combinacoes" in acao:
            raise ValueError("A compra não pode selecionar peças ou combinações.")
        return ("comprar", ())
    por_id = {peca.id: peca for peca in mao}
    if not ids or not set(ids).issubset(por_id):
        raise ValueError("Selecione peças disponíveis na mão.")
    pecas_utilizadas = tuple(sorted((por_id[i].cor, por_id[i].numero) for i in ids))
    if acao["tipo"] == "baixar":
        if "combinacoes" in acao:
            raise ValueError("Use uma ação de mesa para baixar várias combinações.")
        return ("baixar", pecas_utilizadas)
    layout = acao.get("combinacoes")
    if not isinstance(layout, list) or not layout or not all(
        isinstance(c, list) and c and all(isinstance(i, str) for i in c) for c in layout
    ):
        raise ValueError("Informe a mesa final como listas não vazias de IDs.")
    selecionados = [i for c in layout for i in c]
    antigas = {p.id: p for c in mesa or [] for p in c.pecas}
    if len(selecionados) != len(set(selecionados)) or set(selecionados) != antigas.keys() | set(ids):
        raise ValueError("A mesa final deve conservar suas peças e usar as peças selecionadas da mão.")
    todas = antigas | por_id
    final = tuple(sorted(tuple(sorted((todas[i].cor, todas[i].numero) for i in c)) for c in layout))
    return ("mesa", (pecas_utilizadas,) + final)


def agrupar_acoes(acoes: list[Acao], mao: list[Peca], mesa: list[Combinacao] | None = None) -> dict[ChaveAcao, Acao]:
    """Associar cada decisão distinta à primeira ação concreta equivalente."""
    agrupadas = {}
    for acao in acoes:
        agrupadas.setdefault(chave_acao(acao, mao, mesa), acao)
    return agrupadas
