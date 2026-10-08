"""Posições das peças no quadro 4:5 (10,8 × 13,5 unidades; origem no centro, y para cima). Sem dependências."""
LARGURA, ALTURA = 10.8, 13.5
BASE_W, BASE_H = 0.62, 0.88
FOLGA = 0.05
MARGEM = 0.35
USAVEL = LARGURA - 2 * MARGEM
TOPO = ALTURA / 2
ZONA_OPONENTE = (5.1, 3.3)
ZONA_MESA = (2.9, -3.4)
ZONA_JOGADOR = (-3.8, -5.5)
MONTE = (LARGURA / 2 - 0.75, -ALTURA / 2 + 0.55)
ORDEM_COR = {"azul": 0, "vermelho": 1, "verde": 2, "amarelo": 3, "coringa": 4}


def _ordenar(mao):
    return sorted(mao, key=lambda p: (ORDEM_COR[p[1]], p[2]))


def _escala_maxima(escalas, cabe):
    for s in escalas:
        if cabe(s):
            return s
    return escalas[-1]


def _mao(mao, zona):
    topo, base = zona
    altura = topo - base
    n = len(mao)
    passo = lambda s: BASE_W * s + FOLGA
    linhas = lambda s: max(1, -(-n // max(1, int(USAVEL // passo(s)))))
    escalas = [1 - 0.05 * k for k in range(16)]
    s = _escala_maxima(escalas, lambda s: linhas(s) * (BASE_H * s + FOLGA) <= altura)
    por_linha = max(1, int(USAVEL // passo(s)))
    saida = {}
    for k, p in enumerate(_ordenar(mao)):
        linha, col = divmod(k, por_linha)
        da_linha = min(por_linha, n - linha * por_linha)
        x = (col - (da_linha - 1) / 2) * passo(s)
        y = topo - (BASE_H * s + FOLGA) * (linha + 0.5) - (altura - linhas(s) * (BASE_H * s + FOLGA)) / 2
        saida[p[0]] = (x, y, s)
    return saida


def _mesa(conjuntos):
    topo, base = ZONA_MESA
    altura = topo - base

    def empacotar(s):
        passo, intervalo = BASE_W * s + FOLGA, 0.22 * s + 0.1
        linhas, x, linha = [[]], 0.0, 0
        for c in conjuntos:
            largura = len(c) * passo
            if largura > USAVEL:
                return None
            if x and x + largura > USAVEL:
                linhas.append([])
                x = 0.0
            linhas[-1].append((x, c))
            x += largura + intervalo
        return linhas if len(linhas) * (BASE_H * s + 0.15) <= altura else None

    escalas = [0.9 - 0.02 * k for k in range(32)]
    s = next((s for s in escalas if empacotar(s)), escalas[-1])
    linhas = empacotar(s) or empacotar(escalas[-1]) or [[(0.0, c)] for c in conjuntos]
    passo = BASE_W * s + FOLGA
    saida = {}
    for r, linha in enumerate(linhas):
        y = topo - (BASE_H * s + 0.15) * (r + 0.5)
        for x0, c in linha:
            for i, p in enumerate(c):
                saida[p[0]] = (-USAVEL / 2 + x0 + (i + 0.5) * passo, y, s)
    return saida


def layout(estado: dict, baixo: int = 0) -> dict:
    """`baixo` é o assento do jogador estudado (mão na parte de baixo)."""
    pecas = {}
    pecas.update(_mao(estado["maos"][1 - baixo], ZONA_OPONENTE))
    pecas.update(_mao(estado["maos"][baixo], ZONA_JOGADOR))
    pecas.update(_mesa(estado["mesa"]))
    return {"pecas": pecas, "monte": MONTE}
