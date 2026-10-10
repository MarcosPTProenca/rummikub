import random
from collections.abc import Callable

from .contagem import (contagem_escondidas, copias_por_tipo, desbloqueios_por_numero, extensoes_da_mesa,
                       potencial_ponderado, travantes_gastos)

Politica = Callable[..., dict]


def oponentes(jogo) -> list:
    """Os outros jogadores, em ordem de assento."""
    return [j for k, j in enumerate(jogo.jogadores) if k != jogo.indice_jogador_atual]


def medidas(jogo, acao):
    """Peças da mão, soma dos números e coringas usados pela ação."""
    por_id = {p.id: p for p in jogo.jogador_atual.mao}
    usadas = [por_id[i] for i in acao["ids_pecas"]] if acao["tipo"] != "comprar" else []
    return len(usadas), sum(p.numero for p in usadas), sum(p.cor == "coringa" for p in usadas)


def _melhor(jogo, jogadas, chave, rng):
    if not jogadas:
        return None
    pontuadas = [(chave(*medidas(jogo, a)), a) for a in jogadas]
    maior = max(p for p, _ in pontuadas)
    return rng.choice([a for p, a in pontuadas if p == maior])


def _jogadas(acoes):
    return [a for a in acoes if a["tipo"] != "comprar"]


def _comprar(acoes):
    return acoes[-1]


def _por(chave, filtro=lambda jogo, a: True):
    def politica(jogo, acoes, rng):
        jogadas = [a for a in _jogadas(acoes) if filtro(jogo, a)]
        return _melhor(jogo, jogadas, chave, rng) or _comprar(acoes)
    return politica


def _poupar_coringa(jogo, acoes, rng, quando=lambda jogo: True, exceto_vitoria=True):
    if not quando(jogo):
        return _por(lambda n, s, c: (n, s))(jogo, acoes, rng)
    jogadas = _jogadas(acoes)
    vence = [a for a in jogadas if exceto_vitoria and len(a["ids_pecas"]) == len(jogo.jogador_atual.mao)]
    sem = [a for a in jogadas if medidas(jogo, a)[2] == 0]
    return _melhor(jogo, vence or sem, lambda n, s, c: (n, s), rng) or _comprar(acoes)


def _cauteloso(jogo, acoes, rng):
    jogada = _melhor(jogo, _jogadas(acoes), lambda n, s, c: (n, s), rng)
    if jogada and (len(jogada["ids_pecas"]) >= 4 or len(jogada["ids_pecas"]) == len(jogo.jogador_atual.mao)):
        return jogada
    return _comprar(acoes)


ESTRATEGIAS: dict[str, Politica] = {
    "aleatorio": lambda jogo, acoes, rng: rng.choice(acoes),
    "max_pecas": _por(lambda n, s, c: (n, s)),
    "max_pontos": _por(lambda n, s, c: (s, n)),
    "minimo": _por(lambda n, s, c: (-n, -s)),
    "poupar_coringa": _poupar_coringa,
    "so_baixar": _por(lambda n, s, c: (n, s), lambda jogo, a: a["tipo"] == "baixar"),
    "cauteloso": _cauteloso,
}
VARIANTES: dict[str, Politica] = {
    "poupar_sem_final": lambda jogo, acoes, rng: _poupar_coringa(jogo, acoes, rng, exceto_vitoria=False),
    "poupar_antes": lambda jogo, acoes, rng: _poupar_coringa(jogo, acoes, rng, lambda j: not j.jogador_atual.abriu),
    "poupar_depois": lambda jogo, acoes, rng: _poupar_coringa(jogo, acoes, rng, lambda j: j.jogador_atual.abriu),
    **{f"penal_{x}": _por(lambda n, s, c, x=x: (n - x * c, s)) for x in (0.5, 1, 2, 4)},
    **{f"bonus_{x}": _por(lambda n, s, c, x=x: (n + x * c, s)) for x in (1, 2)},
}


def _coringas_movidos(jogo, acao) -> int:
    """Quantos coringas da mesa saem do conjunto em que estavam, pela ação."""
    if acao["tipo"] != "mesa":
        return 0
    antigos = [{p.id for p in c.pecas} for c in jogo.mesa]
    return sum(not any(i in novo and conjunto <= set(novo) for novo in acao["combinacoes"])
               for c, conjunto in zip(jogo.mesa, antigos) for i in (p.id for p in c.pecas if p.cor == "coringa"))


def sem_roubo(politica):
    """Regra contrafactual: ninguém pode tirar um coringa do conjunto em que ele está na mesa."""
    def filtrada(jogo, acoes):
        antigos = [{p.id for p in c.pecas} for c in jogo.mesa]
        coringas = [(i, conjunto) for c, conjunto in zip(jogo.mesa, antigos)
                    for i in (p.id for p in c.pecas if p.cor == "coringa")]

        def preserva(acao):
            if acao["tipo"] != "mesa":
                return True
            return all(any(i in novo and conjunto <= set(novo) for novo in acao["combinacoes"])
                       for i, conjunto in coringas)
        return politica(jogo, [a for a in acoes if preserva(a)])
    return filtrada


def _potencial(pecas) -> int:
    """Pares da mão que podem virar conjunto: mesmo número em cores diferentes, ou mesma cor a até 2 de distância."""
    comuns = [p for p in pecas if p.cor != "coringa"]
    return sum(a.numero == b.numero and a.cor != b.cor or a.cor == b.cor and 0 < abs(a.numero - b.numero) <= 2
               for i, a in enumerate(comuns) for b in comuns[i + 1:])


def _adaptativo(jogo, acoes, rng):
    perto = any(len(j.mao) <= 4 for j in jogo.jogadores if j is not jogo.jogador_atual)
    return _por(lambda n, s, c: (n, s))(jogo, acoes, rng) if perto else _poupar_coringa(jogo, acoes, rng)


def _reorganizador(jogo, acoes, rng):
    def chave(a):
        n, soma, coringas = medidas(jogo, a)
        return (-coringas, _coringas_movidos(jogo, a), n, soma)
    return _melhor_por_acao(jogo, _jogadas(acoes), chave, rng) or _comprar(acoes)


def _melhor_por_acao(jogo, jogadas, chave, rng):
    if not jogadas:
        return None
    pontuadas = [(chave(a), a) for a in jogadas]
    maior = max(p for p, _ in pontuadas)
    return rng.choice([a for p, a in pontuadas if p == maior])


def _flexivel(jogo, acoes, rng):
    def chave(a):
        sobra = [p for p in jogo.jogador_atual.mao if p.id not in set(a["ids_pecas"])]
        return (len(a["ids_pecas"]), _potencial(sobra))
    return _melhor_por_acao(jogo, _jogadas(acoes), chave, rng) or _comprar(acoes)


def _conta_rara(jogo, acoes, rng):
    """Max-tiles que desempata pelo que a mão restante ainda pode completar, dada a contagem de peças escondidas."""
    escondidas, copias = contagem_escondidas(jogo), copias_por_tipo(jogo)

    def chave(a):
        sobra = [p for p in jogo.jogador_atual.mao if p.id not in set(a["ids_pecas"])]
        return (len(a["ids_pecas"]), potencial_ponderado(sobra, escondidas, copias))
    return _melhor_por_acao(jogo, _jogadas(acoes), chave, rng) or _comprar(acoes)


def _defensiva(gasto):
    """Esvaziar a mão vem primeiro; depois, jogadas que não gastam peça travante; só então o tamanho."""
    def politica(jogo, acoes, rng):
        mao, custo = len(jogo.jogador_atual.mao), gasto(jogo)
        chave = lambda a: (len(a["ids_pecas"]) == mao, custo(a["ids_pecas"]) == 0, len(a["ids_pecas"]))
        return _melhor_por_acao(jogo, _jogadas(acoes), chave, rng) or _comprar(acoes)
    return politica


def _gasto_numero(jogo):
    return lambda ids: desbloqueios_por_numero(jogo, ids)


def _gasto_copia(jogo):
    escondidas, extensoes = contagem_escondidas(jogo), extensoes_da_mesa(jogo)
    return lambda ids: travantes_gastos(jogo, ids, escondidas, extensoes)


EXTRAS: dict[str, Politica] = {
    "defensiva_numero": _defensiva(_gasto_numero),
    "defensiva_copia": _defensiva(_gasto_copia),
    "conta_rara": _conta_rara,
    "adaptativo": _adaptativo,
    "reorganizador": _reorganizador,
    "flexivel": _flexivel,
    **{f"poupar_ate_{n}": (lambda jogo, acoes, rng, n=n: _poupar_coringa(
        jogo, acoes, rng, lambda j: len(j.jogador_atual.mao) > n)) for n in (2, 3)},
}
COMPRAR: Politica = lambda jogo, acoes, rng: _comprar(acoes)


def criar(nome: str, rng: random.Random):
    """Fixar o gerador de empates; a política recebe (jogo, acoes)."""
    politica = {**ESTRATEGIAS, **VARIANTES, **EXTRAS}[nome]
    return lambda jogo, acoes: politica(jogo, acoes, rng)
