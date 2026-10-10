"""Contagem de peças: o que ainda pode aparecer, por (cor, número), vendo só a mesa e a mão de quem joga."""
from collections import Counter
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


def extensoes_da_mesa(jogo) -> set[tuple[str, int]]:
    """Tipos (cor, número) que ampliariam alguma combinação da mesa. Sequências com coringa ficam de fora (a ponta é ambígua)."""
    tipos = set()
    for comb in jogo.mesa:
        normais = [p for p in comb.pecas if p.cor != "coringa"]
        if comb.eh_grupo():
            if len(comb.pecas) < 4:
                tipos |= {(c, normais[0].numero) for c in CORES if c not in {p.cor for p in normais}}
        elif comb.eh_sequencia() and len(normais) == len(comb.pecas):
            cor, numeros = normais[0].cor, [p.numero for p in normais]
            tipos |= {(cor, n) for n in (min(numeros) - 1, max(numeros) + 1) if 1 <= n <= 13}
    return tipos


def _tipos(jogo, ids) -> Counter:
    por_id = {p.id: p for p in jogo.jogador_atual.mao}
    return Counter((p.cor, p.numero) for p in (por_id[i] for i in ids) if p.cor != "coringa")


def desbloqueios_por_numero(jogo, ids) -> int:
    """Números cujo grupo (3 cores) estava travado pela minha mão e que a jogada libera.

    Um número está travado por mim quando as cópias fora da minha mão (escondidas ou na mesa, as que os oponentes
    ainda alcançam) cobrem menos de 3 cores; gastar peças dele devolve cores ao alcance deles."""
    copias = copias_por_tipo(jogo)
    minha = _tipos(jogo, [p.id for p in jogo.jogador_atual.mao])
    gasta = _tipos(jogo, ids)
    desbloqueados = 0
    for n in range(1, 14):
        antes = sum(copias - minha[(c, n)] > 0 for c in CORES)
        depois = sum(copias - (minha[(c, n)] - gasta[(c, n)]) > 0 for c in CORES)
        desbloqueados += antes < 3 <= depois
    return desbloqueados


def travantes_gastos(jogo, ids, escondidas, extensoes) -> int:
    """Peças gastas de tipos que só eu e a mesa temos (nenhuma cópia escondida) e que estenderiam a mesa."""
    gasta = _tipos(jogo, ids)
    return sum(k for tipo, k in gasta.items() if escondidas[tipo] == 0 and tipo in extensoes)


def atributos_contagem(jogo, acao) -> list[float]:
    """4 atributos para a rede, todos em [0, 1]: raridade média das peças gastas e das retidas, fração dos
    travantes (cópia esgotada que estende a mesa) que a jogada gasta, e quantos tipos desses eu seguro (/6)."""
    escondidas, extensoes = contagem_escondidas(jogo), extensoes_da_mesa(jogo)
    ids = set(acao["ids_pecas"])
    normais = [p for p in jogo.jogador_atual.mao if p.cor != "coringa"]
    media = lambda ps: sum(raridade(jogo, escondidas, (p.cor, p.numero)) for p in ps) / len(ps) if ps else 0.0
    travantes = [p for p in normais if escondidas[(p.cor, p.numero)] == 0 and (p.cor, p.numero) in extensoes]
    gastos = sum(p.id in ids for p in travantes)
    return [media([p for p in normais if p.id in ids]), media([p for p in normais if p.id not in ids]),
            gastos / len(travantes) if travantes else 0.0, min(len({(p.cor, p.numero) for p in travantes}), 6) / 6]
