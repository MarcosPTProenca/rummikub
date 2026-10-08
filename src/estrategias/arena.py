from src.jogo.jogo import Jogo
from .politicas import medidas


def executar(jogo: Jogo, acao: dict) -> None:
    if acao["tipo"] == "comprar":
        jogo.comprar_peca()
    elif acao["tipo"] == "baixar":
        jogo.baixar_combinacao(acao["ids_pecas"])
    else:
        jogo.jogar_turno(acao["combinacoes"])


def registrar_compra(jogo, acao, telemetria: dict) -> None:
    """Conta o coringa que a compra vai trazer (o topo do monte), antes de executar."""
    if acao["tipo"] == "comprar" and jogo.monte and jogo.monte[-1].cor == "coringa":
        telemetria["coringas_comprados"] += 1


def jogar(politicas, semente: int, limite_decisoes: int | None = None, desempate: str = "pontos", rearranjar: bool = True,
          observador=None, limites: dict | None = None) -> dict:
    """Partida com len(politicas) jogadores (2 a 10); o assento 0 começa. Vencedor None = empate/truncada.
    Sem limite explícito, 400 decisões ou 100 por jogador (o que for maior)."""
    limite_decisoes = limite_decisoes or max(400, 100 * len(politicas))
    jogo = Jogo(semente=semente, desempate=desempate, quantidade_jogadores=len(politicas))
    decisoes = 0
    telemetria = [{"jogadas": 0, "compras": 0, "coringas_jogados": 0, "coringas_comprados": 0} for _ in politicas]
    coringas_na_ultima = 0
    saturacao = {"planos": 0, "visitas": 0, "pecas": 0, "nos": 0}
    while not jogo.terminou and decisoes < limite_decisoes:
        lado = telemetria[jogo.indice_jogador_atual]
        sat = {}
        acoes = jogo.listar_acoes_validas(rearranjar, limites=limites, saturacao=sat)
        for chave, atingido in sat.items():
            saturacao[chave] += atingido
        acao = politicas[jogo.indice_jogador_atual](jogo, acoes)
        n, soma, coringas_na_ultima = medidas(jogo, acao)
        lado["compras" if acao["tipo"] == "comprar" else "jogadas"] += 1
        lado["coringas_jogados"] += coringas_na_ultima
        registrar_compra(jogo, acao, lado)
        indice = jogo.indice_jogador_atual
        executar(jogo, acao)
        if observador:
            observador(jogo, indice, acao, (n, soma, coringas_na_ultima))
        decisoes += 1
    return {
        "vencedor": int(jogo.vencedor.id) if jogo.vencedor else None,
        "decisoes": decisoes,
        "truncada": not jogo.terminou,
        "maos": [len(j.mao) for j in jogo.jogadores],
        "telemetria": telemetria,
        "saturacao": saturacao,
        "compras_total": sum(t["compras"] for t in telemetria),
        "coringas_mao": [sum(p.cor == "coringa" for p in j.mao) for j in jogo.jogadores],
        "saida_com_coringa": bool(jogo.vencedor) and coringas_na_ultima > 0,
        "pecas_total": sum(len(j.mao) for j in jogo.jogadores) + len(jogo.monte)
        + sum(len(c.pecas) for c in jogo.mesa),
    }
