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


def _disponibilidade(escondidas, faltantes, copias) -> float:
    """Quanto ainda dá para completar: cópias escondidas dos tipos que fecham a combinação, no máximo 1."""
    return min(1.0, sum(escondidas[t] for t in faltantes) / copias)


def potencial_ponderado(pecas, escondidas, copias) -> float:
    """Como `_potencial` (pares que podem virar combinação), mas cada par vale só o quanto ainda se completa."""
    comuns = [p for p in pecas if p.cor != "coringa"]
    total = 0.0
    for i, a in enumerate(comuns):
        for b in comuns[i + 1:]:
            if a.numero == b.numero and a.cor != b.cor:
                faltantes = [(c, a.numero) for c in CORES if c not in (a.cor, b.cor)]
            elif a.cor == b.cor and 0 < abs(a.numero - b.numero) <= 2:
                baixo, alto = sorted((a.numero, b.numero))
                vizinhos = [baixo + 1] if alto - baixo == 2 else [baixo - 1, alto + 1]
                faltantes = [(a.cor, n) for n in vizinhos if 1 <= n <= 13]
            else:
                continue
            total += _disponibilidade(escondidas, faltantes, copias)
    return total
