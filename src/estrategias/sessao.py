"""Sessão de várias mãos, como Rummikub é realmente jogado.

Joga K mãos acumulando a pontuação oficial; vence a sessão quem tem o maior
total. Duas escolhas de protocolo importam:

- A semente de cada mão vem de (semente da sessão, índice da mão), então
  estratégias diferentes enfrentam as MESMAS mãos na mesma posição da sessão.
  É o pareamento do torneio estendido à sessão.
- O assento inicial alterna a cada mão, como a regra oficial passa a vez; com K
  par isso cancela o efeito de primeiro jogador dentro da própria sessão.

A sessão também cria informação que nenhuma estratégia de mão única tem: o
saldo acumulado. O motor escreve `jogo.saldo_sessao` antes de cada mão (pontos
do jogador atual menos o melhor oponente); políticas que não leem o atributo
não mudam de comportamento.
"""
import random
from functools import lru_cache
from pathlib import Path

from src.jogo.jogo import Jogo
from src.jogo.pontuacao import pontos_oficiais

from .politicas import criar
from .rede import PoliticaRede, carregar_rede


def semente_da_mao(semente: int, indice: int) -> int:
    """Determinística e independente de quem joga, para pares enfrentarem as mesmas mãos."""
    return int(random.Random(f"{semente}|{indice}").randrange(10 ** 9))


@lru_cache(maxsize=None)
def _rede_de(caminho: str):
    """Checkpoint carregado uma vez por processo: a avaliação joga milhares de mãos com a mesma rede.

    O cache é por caminho, então só serve para avaliação; quem treina e regrava o mesmo
    arquivo no mesmo processo precisa de `_rede_de.cache_clear()`."""
    return carregar_rede(Path(caminho))


def _construir(nome: str, rng: random.Random):
    """`rede:<caminho>` é uma rede treinada (comum ou de sessão, jogando gulosa); o resto são heurísticas."""
    if nome.startswith("rede:"):
        return PoliticaRede.da_rede(_rede_de(nome[5:]), rng, guloso=True)
    return criar(nome, rng)


def _jogar_mao(nomes: list[str], semente: int, primeiro: int, saldos: list[float],
               limite: int | None = None, rearranjar: bool = True, politicas: list | None = None,
               maos_restantes: int | None = None, maos_total: int | None = None) -> dict:
    """Uma mão; `primeiro` é o assento que começa, obtido rotacionando as políticas.

    `politicas` (indexadas pelo assento lógico) substitui a construção por nome, para o treino
    poder gravar trajetórias; `maos_restantes`/`maos_total` ficam visíveis à política."""
    n = len(nomes)
    ordem = [(primeiro + k) % n for k in range(n)]
    if politicas is None:
        politicas = [_construir(nomes[assento], random.Random(semente * 10 + assento)) for assento in range(n)]
    jogo = Jogo(semente=semente, desempate="pontos", quantidade_jogadores=n)
    # O assento 0 do motor sempre começa, então mapeia assento lógico -> posição na mesa.
    posicao = {assento: k for k, assento in enumerate(ordem)}
    jogo.saldo_sessao = 0.0
    if maos_total is not None:
        jogo.maos_restantes, jogo.maos_total = maos_restantes, maos_total
    decisoes, limite = 0, limite or max(400, 100 * n)
    while not jogo.terminou and decisoes < limite:
        atual = ordem[jogo.indice_jogador_atual]
        jogo.saldo_sessao = saldos[atual] - max(s for k, s in enumerate(saldos) if k != atual)
        acao = politicas[atual](jogo, jogo.listar_acoes_validas(rearranjar))
        if acao["tipo"] == "comprar":
            jogo.comprar_peca()
        elif acao["tipo"] == "baixar":
            jogo.baixar_combinacao(acao["ids_pecas"])
        else:
            jogo.jogar_turno(acao["combinacoes"])
        decisoes += 1
    na_mesa = pontos_oficiais(jogo)
    vencedor = ordem[int(jogo.vencedor.id)] if jogo.vencedor is not None else None
    return {"semente": semente, "primeiro": primeiro, "vencedor": vencedor,
            "pontos": [na_mesa[posicao[assento]] for assento in range(n)],
            "coringas_mao": [sum(p.cor == "coringa" for p in jogo.jogadores[posicao[a]].mao) for a in range(n)],
            "decisoes": decisoes, "truncada": not jogo.terminou}


def sessao(nomes: list[str], semente: int, maos: int = 10, espiao=None) -> dict:
    """K mãos com pontos acumulados; vence o maior total (None se houver empate no topo)."""
    n = len(nomes)
    saldos = [0.0] * n
    registros = []
    for indice in range(maos):
        if espiao:
            espiao(list(saldos))
        mao = _jogar_mao(nomes, semente_da_mao(semente, indice), indice % n, saldos, None,
                         maos_restantes=maos - indice, maos_total=maos)
        for assento in range(n):
            saldos[assento] += mao["pontos"][assento]
        registros.append(mao)
    melhor = max(saldos)
    vencedor = saldos.index(melhor) if saldos.count(melhor) == 1 else None
    return {"nomes": list(nomes), "semente": semente, "maos": registros, "pontos": saldos,
            "vencedor": vencedor,
            "maos_ganhas": [sum(m["vencedor"] == assento for m in registros) for assento in range(n)],
            "coringas_mao": [sum(m["coringas_mao"][assento] for m in registros) / maos for assento in range(n)]}
