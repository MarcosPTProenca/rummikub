"""Pontuação oficial de Rummikub: soma zero em pontos de peças.

Quem sai soma o valor das peças que sobraram nas mãos dos oponentes; cada
perdedor subtrai o valor da própria mão, com o coringa valendo 30. Com o monte
esgotado e ninguém tendo saído, vence a menor contagem: cada outro perde a
diferença para ela, e o vencedor ganha a soma dessas diferenças. A regra oficial
não cobre empate na menor contagem; aqui os empatados dividem o prêmio em partes
iguais (com um só empatado, reduz à regra oficial).
"""
from .peca import Peca

CORINGA = 30


def valor_mao(mao: list[Peca]) -> int:
    """Pontos de uma mão: o número da peça, ou 30 por coringa."""
    return sum(CORINGA if p.cor == "coringa" else p.numero for p in mao)


def pontos_oficiais(jogo) -> list[float]:
    """Pontos de cada assento, soma zero. Só faz sentido com a partida encerrada."""
    valores = [valor_mao(j.mao) for j in jogo.jogadores]
    if jogo.vencedor is not None:
        vencedor = int(jogo.vencedor.id)
        return [float(sum(valores) - valores[vencedor]) if i == vencedor else float(-v)
                for i, v in enumerate(valores)]
    # Monte esgotado sem saída: a menor mão ganha as diferenças das outras.
    menor = min(valores)
    empatados = [i for i, v in enumerate(valores) if v == menor]
    premio = sum(v - menor for v in valores) / len(empatados)
    return [premio if i in empatados else float(menor - v) for i, v in enumerate(valores)]
