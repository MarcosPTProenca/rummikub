"""Rede convolucional sobre o tabuleiro cor × número (plano 16).

Dois ramos, como o IRumAI (arXiv 2606.21975), adaptados à geometria de
Rummikub: conv 1×3 ao longo dos números detecta sequências, conv 4×1 ao longo
das cores detecta grupos. A ablação deles mostra que esta arquitetura vale 8 a
11 pontos percentuais sobre um MLP de mesma largura — mais que reward shaping
ou warm-start, e o maior contribuinte isolado que reportam.

numpy puro, como o resto do projeto: `theta` é um vetor achatado, para a rede
viajar entre processos do `multiprocessing` como as outras.
"""
import json
import math
import random
from pathlib import Path

import numpy as np

from .tensor import CANAIS, ESCALARES, NUMEROS

FILTROS = 16   # por ramo; fixado antes de rodar
OCULTOS = 64   # camada densa do tronco


def _fatias():
    """Fronteiras de theta: (w_seq, b_seq, w_set, b_set, W, b, v, c)."""
    n_seq = FILTROS * CANAIS * 3
    n_set = FILTROS * CANAIS * 4
    entrada = FILTROS * NUMEROS * 2 + ESCALARES
    n_W = OCULTOS * entrada
    partes = [n_seq, FILTROS, n_set, FILTROS, n_W, OCULTOS, OCULTOS, 1]
    cortes, inicio = [], 0
    for n in partes:
        cortes.append((inicio, inicio + n))
        inicio += n
    return cortes, entrada


class RedeConv:
    """Pontua (estado, jogada) a partir do tensor cor × número."""

    def __init__(self, theta: list[float]):
        self.theta = theta
        self.m, self.v = np.zeros(len(theta)), np.zeros(len(theta))

    @classmethod
    def tamanho(cls) -> int:
        cortes, _ = _fatias()
        return cortes[-1][1]

    @classmethod
    def nova(cls, rng: random.Random):
        cortes, entrada = _fatias()
        theta = np.zeros(cortes[-1][1])
        # He para as convoluções (ReLU), Xavier para o tronco (tanh).
        for i, (a, b) in enumerate(cortes):
            if i in (0, 2):
                escala = math.sqrt(2.0 / (CANAIS * (3 if i == 0 else 4)))
            elif i == 4:
                escala = math.sqrt(1.0 / entrada)
            elif i == 6:
                escala = math.sqrt(1.0 / OCULTOS)
            else:
                escala = 0.0
            if escala:
                theta[a:b] = [rng.gauss(0, escala) for _ in range(b - a)]
        return cls(theta.tolist())

    def _partes(self):
        cortes, entrada = _fatias()
        t = np.asarray(self.theta)
        w_seq = t[slice(*cortes[0])].reshape(FILTROS, CANAIS, 3)
        b_seq = t[slice(*cortes[1])]
        w_set = t[slice(*cortes[2])].reshape(FILTROS, CANAIS, 4)
        b_set = t[slice(*cortes[3])]
        W = t[slice(*cortes[4])].reshape(OCULTOS, entrada)
        b = t[slice(*cortes[5])]
        v = t[slice(*cortes[6])]
        c = t[cortes[7][0]]
        return w_seq, b_seq, w_set, b_set, W, b, v, c

    @staticmethod
    def _empilhar(X):
        """X é uma lista de (tensor, escalares); devolve os lotes."""
        T = np.stack([t for t, _ in X])
        S = np.asarray([s for _, s in X], dtype=float)
        return T, S

    def _frente(self, X):
        w_seq, b_seq, w_set, b_set, W, b, v, c = self._partes()
        T, S = self._empilhar(X)
        n = T.shape[0]

        # Ramo sequência: 1×3 ao longo dos números, com zero-padding nas pontas.
        # `janelas[n, canal, cor, numero, d]` é o vizinho d∈{-1,0,+1} daquele número.
        pad = np.pad(T, ((0, 0), (0, 0), (0, 0), (1, 1)))
        janelas = np.stack([pad[:, :, :, d:d + NUMEROS] for d in range(3)], axis=-1)
        por_cor = np.einsum("ncxpd,fcd->nfxp", janelas, w_seq)
        # Max-pool sobre cores vem depois do ReLU, como no IRumAI.
        seq = np.maximum(por_cor + b_seq[None, :, None, None], 0.0).max(axis=2)

        # Ramo grupo: 4×1 ao longo das cores, colapsa a dimensão de cor.
        grupo = np.maximum(np.einsum("ncxp,fcx->nfp", T, w_set) + b_set[None, :, None], 0.0)

        fundidos = np.concatenate([seq.reshape(n, -1), grupo.reshape(n, -1), S], axis=1)
        ocultas = np.tanh(fundidos @ W.T + b)
        return (T, S, janelas, por_cor, seq, grupo, fundidos, ocultas), (ocultas @ v + c).tolist()

    def pontuacoes(self, X, deslocamento=None):
        s = self._frente(X)[1]
        return s if deslocamento is None else [a + d for a, d in zip(s, deslocamento)]

    @staticmethod
    def _softmax(s):
        m = max(s)
        e = [math.exp(x - m) for x in s]
        total = sum(e)
        return [x / total for x in e]

    def probabilidades(self, X, deslocamento=None):
        return self._softmax(self.pontuacoes(X, deslocamento))

    def gradiente(self, X, escolhida: int, vantagem: float, beta: float = 0.0):
        """Gradiente de `vantagem * log p(escolhida)`, mais entropia se `beta` > 0."""
        w_seq, b_seq, w_set, b_set, W, b, v, c = self._partes()
        cache, s = self._frente(X)
        T, S, janelas, por_cor, seq, grupo, fundidos, ocultas = cache
        p = np.asarray(self._softmax(s))
        n = len(s)

        # d(log p_escolhida)/d(pontuacao_i) = 1[i == escolhida] - p_i
        ds = -p.copy()
        ds[escolhida] += 1.0
        ds *= vantagem
        if beta:
            logp = np.log(np.clip(p, 1e-12, None))
            dH = -p * (logp + float(p @ logp))
            ds += beta * dH

        cortes, entrada = _fatias()
        g = np.zeros(len(self.theta))

        # Saída: s = ocultas @ v + c
        g[slice(*cortes[6])] = ds @ ocultas
        g[cortes[7][0]] = ds.sum()

        # Tronco: ocultas = tanh(fundidos @ W.T + b)
        d_ocultas = np.outer(ds, v) * (1 - ocultas ** 2)
        g[slice(*cortes[4])] = (d_ocultas.T @ fundidos).reshape(-1)
        g[slice(*cortes[5])] = d_ocultas.sum(axis=0)

        d_fundidos = d_ocultas @ W
        corte_seq = FILTROS * NUMEROS
        d_seq = d_fundidos[:, :corte_seq].reshape(n, FILTROS, NUMEROS)
        d_grupo = d_fundidos[:, corte_seq:2 * corte_seq].reshape(n, FILTROS, NUMEROS)

        # Ramo grupo: ReLU e conv 4×1.
        d_pre_grupo = d_grupo * (grupo > 0)
        g[slice(*cortes[2])] = np.einsum("nfp,ncxp->fcx", d_pre_grupo, T).reshape(-1)
        g[slice(*cortes[3])] = d_pre_grupo.sum(axis=(0, 2))

        # Ramo sequência: o max-pool sobre cores manda o gradiente ao argmax.
        ativado = por_cor + b_seq[None, :, None, None]
        arg = ativado.argmax(axis=2, keepdims=True)
        mascara = np.zeros_like(ativado)
        np.put_along_axis(mascara, arg, 1.0, axis=2)
        mascara *= (ativado > 0)
        d_pre_seq = mascara * d_seq[:, :, None, :]
        g[slice(*cortes[0])] = np.einsum("nfxp,ncxpd->fcd", d_pre_seq, janelas).reshape(-1)
        g[slice(*cortes[1])] = d_pre_seq.sum(axis=(0, 2, 3))
        return g.tolist()

    def passo(self, grad, passos: int, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        """Adam, igual ao da rede do artigo, para a comparação isolar a arquitetura."""
        g = np.asarray(grad)
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        m = self.m / (1 - b1 ** passos)
        v = self.v / (1 - b2 ** passos)
        self.theta = (np.asarray(self.theta) + lr * m / (np.sqrt(v) + eps)).tolist()

    def salvar(self, arquivo: Path) -> None:
        Path(arquivo).parent.mkdir(parents=True, exist_ok=True)
        Path(arquivo).write_text(json.dumps({
            "arquitetura": "conv", "theta": self.theta, "filtros": FILTROS,
            "ocultos": OCULTOS, "canais": CANAIS, "escalares": ESCALARES}))

    @classmethod
    def carregar(cls, arquivo: Path):
        dados = json.loads(Path(arquivo).read_text())
        assert dados["arquitetura"] == "conv"
        assert dados["filtros"] == FILTROS and dados["ocultos"] == OCULTOS
        assert dados["canais"] == CANAIS and dados["escalares"] == ESCALARES
        return cls(dados["theta"])


class PoliticaConv:
    """Escolhe a jogada pela rede convolucional; grava a trajetória para o REINFORCE."""

    def __init__(self, rede: RedeConv, rng: random.Random, guloso: bool = False,
                 gravar: bool = False):
        self.rede, self.rng, self.guloso, self.gravar = rede, rng, guloso, gravar
        self.trajetoria: list[tuple[list, int]] = []
        self.logps: list[float | None] = []

    def __call__(self, jogo, acoes):
        from .tensor import tensor_da_acao
        if len(acoes) == 1:
            return acoes[0]
        X = [tensor_da_acao(jogo, a) for a in acoes]
        if self.guloso:
            s = self.rede.pontuacoes(X)
            indice = s.index(max(s))
            logp = None
        else:
            p = self.rede.probabilidades(X)
            indice = self.rng.choices(range(len(acoes)), p)[0]
            logp = math.log(p[indice])
        if self.gravar:
            self.trajetoria.append((X, indice))
            self.logps.append(logp)
        return acoes[indice]
