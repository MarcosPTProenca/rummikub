"""Representação estruturada de uma jogada: tensor cor × número (plano 16).

O vetor de atributos do artigo é escalar agregado (`n/14`, `soma/60`,
`min(maos)/14`), o que apaga duas coisas que são a estrutura do jogo: números
vizinhos formam sequência, e cores distintas do mesmo número formam grupo. Aqui
a jogada vira um tensor `CANAIS × 4 × 13`, em que convolução 1×3 ao longo dos
números vê sequências e 4×1 ao longo das cores vê grupos.

Os 15 atributos originais seguem como escalares, de propósito: a comparação com
o plano 15 tem de ser *adição* de estrutura, não troca de informação.
"""
import numpy as np

CORES = {"azul": 0, "vermelho": 1, "verde": 2, "amarelo": 3}
NUMEROS = 13
CANAIS = 7
# 15 originais, mais coringas na mão/jogada que o tensor não representa
# (coringa não tem cor nem número) e três resumos de combinação.
ESCALARES = 15 + 5

_COPIAS = 2.0  # duas cópias de cada tipo por conjunto de 106 peças


def _marcar(plano, pecas, peso=1.0 / _COPIAS):
    for p in pecas:
        if p.cor != "coringa":
            plano[CORES[p.cor], p.numero - 1] += peso


def _completa_combinacao(mao_restante):
    """Peças que completariam uma combinação do que sobra na mão (os 'live outs').

    Marca a célula de cada peça ausente que fecharia um par em grupo (mesmo
    número, cor faltante) ou uma ponta de sequência (mesma cor, número vizinho).
    """
    plano = np.zeros((4, NUMEROS))
    por_numero = {}
    por_cor = {}
    for p in mao_restante:
        if p.cor == "coringa":
            continue
        por_numero.setdefault(p.numero, set()).add(p.cor)
        por_cor.setdefault(p.cor, set()).add(p.numero)
    for numero, cores in por_numero.items():
        if len(cores) >= 2:  # par de grupo: falta uma cor
            for cor in CORES:
                if cor not in cores:
                    plano[CORES[cor], numero - 1] = 1.0
    for cor, numeros in por_cor.items():
        for n in numeros:
            if n + 1 in numeros and n + 2 <= NUMEROS and n + 2 not in numeros:
                plano[CORES[cor], n + 1] = 1.0  # ponta de cima
            if n - 1 in numeros and n - 2 >= 1 and n - 2 not in numeros:
                plano[CORES[cor], n - 3] = 1.0  # ponta de baixo
    return plano


def _em_par(mao_restante):
    """Peças da mão que já estão em dupla (mesma cor e número, ou par de grupo)."""
    plano = np.zeros((4, NUMEROS))
    contagem = {}
    for p in mao_restante:
        if p.cor != "coringa":
            contagem[(p.cor, p.numero)] = contagem.get((p.cor, p.numero), 0) + 1
    por_numero = {}
    for (cor, numero), quantas in contagem.items():
        por_numero.setdefault(numero, []).append(cor)
        if quantas >= 2:
            plano[CORES[cor], numero - 1] = 1.0
    for numero, cores in por_numero.items():
        if len(cores) >= 2:
            for cor in cores:
                plano[CORES[cor], numero - 1] = max(plano[CORES[cor], numero - 1], 0.5)
    return plano


def tensor_da_acao(jogo, acao):
    """Tensor `CANAIS × 4 × 13` e os escalares de uma jogada.

    Canais: mão restante, peças baixadas, mesa final, peças não vistas, peças
    que completariam combinação na mão restante, peças já em par, e a mesa
    coberta por combinação válida.
    """
    from .rede import atributos

    eu = jogo.jogador_atual
    mao = {p.id: p for p in eu.mao}
    usadas = [mao[i] for i in acao["ids_pecas"] if i in mao] if acao["tipo"] != "comprar" else []
    ids_usadas = {p.id for p in usadas}
    restante = [p for p in eu.mao if p.id not in ids_usadas]

    mesa_pecas = [p for c in jogo.mesa for p in c.pecas]
    if acao["tipo"] == "mesa":
        por_id = {p.id: p for p in mesa_pecas} | mao
        mesa_final = [[por_id[i] for i in c] for c in acao["combinacoes"]]
    elif acao["tipo"] == "baixar":
        mesa_final = [[p for p in c.pecas] for c in jogo.mesa] + [usadas]
    else:
        mesa_final = [[p for p in c.pecas] for c in jogo.mesa]

    t = np.zeros((CANAIS, 4, NUMEROS))
    _marcar(t[0], restante)
    _marcar(t[1], usadas)
    for combinacao in mesa_final:
        _marcar(t[2], combinacao)
    # Não vistas: duas cópias menos o que está na minha mão e na mesa.
    t[3] = 1.0
    _marcar(t[3], eu.mao, -1.0 / _COPIAS)
    _marcar(t[3], mesa_pecas, -1.0 / _COPIAS)
    np.clip(t[3], 0.0, 1.0, out=t[3])
    t[4] = _completa_combinacao(restante)
    t[5] = _em_par(restante)
    for combinacao in mesa_final:
        from src.jogo.combinacao import Combinacao
        if len(combinacao) >= 3 and Combinacao(pecas=list(combinacao)).eh_valido():
            _marcar(t[6], combinacao)

    coringas_mao = sum(p.cor == "coringa" for p in restante)
    coringas_jogada = sum(p.cor == "coringa" for p in usadas)
    escalares = list(atributos(jogo, acao)[:15]) + [
        coringas_mao / 2.0,
        coringas_jogada / 2.0,
        len(mesa_final) / 20.0,
        sum(len(c) for c in mesa_final) / 60.0,
        float(t[4].sum() > 0),
    ]
    return t, escalares


def politica_conv(rede, rng, guloso: bool = False, gravar: bool = False):
    """Política que pontua cada jogada legal pela rede convolucional."""
    from .conv import PoliticaConv
    return PoliticaConv(rede, rng, guloso, gravar)
