"""Contagem de peças: o que ainda pode aparecer, por (cor, número), vendo só a mesa e a mão de quem joga."""
from itertools import product

CORES = ("azul", "vermelho", "verde", "amarelo")
TIPOS = tuple(product(CORES, range(1, 14)))


def copias_por_tipo(jogo) -> int:
    """Cada conjunto de 106 peças traz 2 cópias de cada (cor, número)."""
    return 2 * jogo.conjuntos


def contagem_escondidas(jogo) -> dict[tuple[str, int], int]:
    """Cópias de cada tipo fora da mesa e da mão atual: estão no monte ou na mão dos oponentes. Coringas ficam de fora."""
    escondidas = dict.fromkeys(TIPOS, copias_por_tipo(jogo))
    visiveis = list(jogo.jogador_atual.mao) + [p for c in jogo.mesa for p in c.pecas]
    for p in visiveis:
        if p.cor != "coringa":
            escondidas[(p.cor, p.numero)] -= 1
    return escondidas


def raridade(jogo, escondidas, tipo) -> float:
    """0 = todas as cópias ainda podem vir, 1 = nenhuma pode (quem a tem na mão detém as últimas)."""
    return 1 - escondidas[tipo] / copias_por_tipo(jogo)
