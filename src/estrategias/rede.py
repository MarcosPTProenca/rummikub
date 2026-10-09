import json
import math
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .arena import jogar
from .politicas import ESTRATEGIAS, criar, medidas, oponentes

ENTRADAS = 15


def atributos(jogo, acao) -> list[float]:
    """Vetor (estado, ação) visto pelo jogador atual; dos oponentes só a menor mão e a fração que já abriu."""
    eu = jogo.jogador_atual
    outros = oponentes(jogo)
    mao = len(eu.mao)
    n, soma, coringas = medidas(jogo, acao)
    em_mao = sum(p.cor == "coringa" for p in eu.mao)
    esvazia = float(acao["tipo"] != "comprar" and n == mao)
    return [
        float(acao["tipo"] == "comprar"), float(acao["tipo"] == "mesa"), n / 14, soma / 60, float(coringas),
        esvazia, (mao - n) / 14, (em_mao - coringas) / 2, float(coringas > 0 and not esvazia),
        float(eu.abriu), min(len(o.mao) for o in outros) / 14, sum(o.abriu for o in outros) / len(outros), len(jogo.monte) / 60,
        float(not jogo.monte), mao / 14,
    ]


def estado(jogo) -> list[float]:
    """Só o estado (sem ação) visto pelo jogador atual, para o crítico; preenchido com zeros até ENTRADAS."""
    eu = jogo.jogador_atual
    outros = oponentes(jogo)
    v = [float(eu.abriu), min(len(o.mao) for o in outros) / 14, sum(o.abriu for o in outros) / len(outros),
         len(jogo.monte) / 60, float(not jogo.monte), len(eu.mao) / 14, sum(p.cor == "coringa" for p in eu.mao) / 2]
    return v + [0.0] * (ENTRADAS - len(v))


class Rede:
    """MLP ações→pontuação com uma camada oculta (tanh); theta = [W1, b1, w2, b2] achatados."""

    def __init__(self, theta: list[float], ocultos: int, residual: float = 0.0):
        self.theta, self.ocultos, self.residual = theta, ocultos, residual
        self.m, self.v = np.zeros(len(theta)), np.zeros(len(theta))

    @classmethod
    def nova(cls, rng: random.Random, ocultos: int = 16, zero: bool = False):
        tamanho = ocultos * ENTRADAS + 2 * ocultos + 1
        escala = 1 / math.sqrt(ENTRADAS)
        theta = [rng.gauss(0, escala) for _ in range(ocultos * ENTRADAS)] + [0.0] * ocultos
        theta += [0.0] * (ocultos + 1) if zero else [rng.gauss(0, 0.1) for _ in range(ocultos)] + [0.0]
        assert len(theta) == tamanho
        return cls(theta, ocultos)

    def _partes(self):
        h, f = self.ocultos, ENTRADAS
        t = np.asarray(self.theta)
        return t[:h * f].reshape(h, f), t[h * f:h * f + h], t[h * f + h:h * f + 2 * h], t[-1]

    def _frente(self, X):
        w1, b1, w2, b2 = self._partes()
        ocultas = np.tanh(np.asarray(X) @ w1.T + b1)
        return ocultas, (ocultas @ w2 + b2).tolist()

    @staticmethod
    def _softmax(s):
        m = max(s)
        e = [math.exp(x - m) for x in s]
        z = sum(e)
        return [x / z for x in e]

    def pontuacoes(self, X, deslocamento=None):
        s = self._frente(X)[1]
        return s if deslocamento is None else [a + d for a, d in zip(s, deslocamento)]

    def probabilidades(self, X, deslocamento=None):
        return self._softmax(self.pontuacoes(X, deslocamento))

    def gradiente(self, X, escolhida: int, vantagem: float, beta: float = 0.0, deslocamento=None,
                  kl: float = 0.0, log_ref=None) -> list[float]:
        """Gradiente de vantagem*log π(escolhida) + beta*H(π) - kl*KL(π‖ref) em relação a theta.
        `deslocamento` soma-se às pontuações; `log_ref` são os log-probabilidades do modelo de referência."""
        ocultas, s = self._frente(X)
        if deslocamento is not None:
            s = [a + d for a, d in zip(s, deslocamento)]
        p = self._softmax(s)
        entropia = -sum(q * math.log(q) for q in p if q > 0)
        g = [vantagem * ((k == escolhida) - q) - beta * q * (math.log(q) + entropia) if q > 0 else 0.0
             for k, q in enumerate(p)]
        if kl and log_ref is not None:
            divergencia = sum(q * (math.log(q) - r) for q, r in zip(p, log_ref) if q > 0)
            g = [gk - kl * q * (math.log(q) - r - divergencia) if q > 0 else gk for gk, q, r in zip(g, p, log_ref)]
        return self._retro(ocultas, X, g)

    def gradiente_q(self, x, alvo: float) -> list[float]:
        """Gradiente de -0,5*(q(x)-alvo)² (subir = regredir q ao alvo), com q a pontuação da ação x."""
        ocultas, s = self._frente([x])
        return self._retro(ocultas, [x], [alvo - s[0]])

    def _retro(self, ocultas, X, g) -> list[float]:
        w2 = self._partes()[2]
        g = np.asarray(g)
        dz = (g[:, None] * w2) * (1 - ocultas * ocultas)
        return np.concatenate([(dz.T @ np.asarray(X)).ravel(), dz.sum(0), g @ ocultas, [g.sum()]])

    def passo(self, grad, t: int, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999) -> None:
        """Adam, subindo o objetivo."""
        grad = np.asarray(grad)
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad * grad
        passo = lr * (self.m / (1 - b1 ** t)) / (np.sqrt(self.v / (1 - b2 ** t)) + 1e-8)
        self.theta[:] = (np.asarray(self.theta) + passo).tolist()

    def salvar(self, arquivo: Path, **meta) -> None:
        Path(arquivo).write_text(json.dumps({"ocultos": self.ocultos, "entradas": ENTRADAS, "theta": list(self.theta),
                                             "residual": self.residual, **meta}))

    @classmethod
    def carregar(cls, arquivo: Path):
        dados = json.loads(Path(arquivo).read_text())
        assert dados["entradas"] == ENTRADAS
        return cls(dados["theta"], dados["ocultos"], dados.get("residual", 0.0))


class PoliticaRede:
    """Escolhe pela rede; com `base`, soma `peso_base` à jogada que a heurística escolheria (RL residual)."""

    def __init__(self, rede: Rede, rng: random.Random, guloso: bool = False, gravar: bool = False,
                 base=None, peso_base: float = 0.0, epsilon: float | None = None):
        self.rede, self.rng, self.guloso, self.gravar = rede, rng, guloso, gravar
        self.base, self.peso_base, self.epsilon = base, peso_base, epsilon
        self.trajetoria: list[tuple[list[list[float]], int]] = []
        self.deslocamentos: list[list[float] | None] = []
        self.estados: list[list[float]] = []
        self.logps: list[float | None] = []

    @classmethod
    def da_rede(cls, rede: Rede, rng: random.Random, guloso: bool = False, gravar: bool = False,
                epsilon: float | None = None):
        """Política com a base residual do próprio checkpoint (`rede.residual` > 0 liga o poupar_coringa)."""
        base = ESTRATEGIAS["poupar_coringa"] if rede.residual > 0 else None
        return cls(rede, rng, guloso, gravar, base, rede.residual, epsilon)

    def __call__(self, jogo, acoes):
        X = [atributos(jogo, a) for a in acoes]
        desloc = None
        if self.base is not None:
            escolha = self.base(jogo, acoes, random.Random(0))
            desloc = [self.peso_base if a is escolha else 0.0 for a in acoes]
        logp = None
        if len(acoes) == 1:
            indice = 0
        elif self.epsilon is not None:
            s = self.rede.pontuacoes(X, desloc)
            indice = self.rng.randrange(len(acoes)) if self.rng.random() < self.epsilon else s.index(max(s))
        elif self.guloso:
            s = self.rede.pontuacoes(X, desloc)
            indice = s.index(max(s))
        else:
            p = self.rede.probabilidades(X, desloc)
            indice = self.rng.choices(range(len(acoes)), p)[0]
            logp = math.log(p[indice])
        if self.gravar and len(acoes) > 1:
            self.trajetoria.append((X, indice))
            self.deslocamentos.append(desloc)
            self.estados.append(estado(jogo))
            self.logps.append(logp)
        return acoes[indice]


def vantagens_grupo(retornos: list[float]) -> list[float]:
    """Retorno menos a média do grupo, dividido pelo desvio; grupo sem diferença não dá sinal."""
    media = sum(retornos) / len(retornos)
    desvio = math.sqrt(sum((r - media) ** 2 for r in retornos) / len(retornos))
    return [(r - media) / desvio if desvio > 1e-9 else 0.0 for r in retornos]


def retorno_formado(retorno: float, mao_final: int, forma: float, inicial: int = 14) -> float:
    """Retorno do jogo mais forma * (queda do tamanho da mão / mão inicial): potencial -mão, soma telescópica."""
    return retorno + forma * (inicial - mao_final) / inicial


def gradiente_ppo(rede: Rede, X, escolhida: int, vantagem: float, log_old: float, clip: float,
                  beta: float = 0.0, kl: float = 0.0, log_ref=None) -> list[float]:
    """Gradiente do objetivo clipado do PPO: a razão multiplica a vantagem; fora da faixa a parte da política some."""
    razao = math.exp(math.log(rede.probabilidades(X)[escolhida]) - log_old)
    fora = (vantagem > 0 and razao > 1 + clip) or (vantagem < 0 and razao < 1 - clip)
    return rede.gradiente(X, escolhida, 0.0 if fora else vantagem * razao, beta, None, kl, log_ref)


def _jogar_treino(tarefa):
    """Uma partida de treino; devolve (trajetórias gravadas com retorno, vencedor do agente, oponente)."""
    theta, ocultos, especificacoes, semente, assento, desempate, variante, rearranjar, forma, residual, epsilon = tarefa
    rng = random.Random(semente if variante == 0 else f"{semente}|{variante}")
    agente = PoliticaRede.da_rede(Rede(theta, ocultos, residual), random.Random(rng.random()), gravar=True,
                                  epsilon=epsilon)
    adversarios = []
    for especificacao in especificacoes:
        if isinstance(especificacao, str):
            adversarios.append(criar(especificacao, random.Random(rng.random())))
        else:
            outro_theta, treinar_ambos = especificacao
            adversarios.append(PoliticaRede.da_rede(Rede(outro_theta, ocultos, residual), random.Random(rng.random()),
                                                    gravar=treinar_ambos, epsilon=epsilon))
    politicas = adversarios[:assento] + [agente] + adversarios[assento:]
    resultado = jogar(politicas, semente, desempate=desempate, rearranjar=rearranjar)
    vencedor = resultado["vencedor"]
    lados = [(k, p) for k, p in enumerate(politicas) if p is agente or (isinstance(p, PoliticaRede) and p.gravar)]
    saida = []
    for lado, politica in lados:
        retorno = 0.0 if vencedor is None else (1.0 if vencedor == lado else -1.0)
        saida.append((retorno_formado(retorno, resultado["maos"][lado], forma),
                      list(zip(politica.trajetoria, politica.deslocamentos, politica.estados, politica.logps))))
    return saida, (0.5 if vencedor is None else float(vencedor == assento)), resultado["decisoes"]


def treinar_rede(saida: Path, iteracoes: int, jogos: int, processos: int, ocultos: int = 16, semente: int = 0,
                 desempate: str = "pontos", checkpoint_a_cada: int = 50, lr: float = 3e-3, beta: float = 0.01,
                 inicial: Path | None = None, jogadores: int = 2, grupo: int = 1,
                 rearranjar: bool = False, forma: float = 0.0,
                 liga: tuple[str, ...] = (), residual: float = 0.0, algoritmo: str = "reinforce",
                 epsilon: float = 0.05, clip: float = 0.2, kl: float = 0.1, epocas: int = 4,
                 mesas: tuple[int, ...] = ()) -> Rede:
    """Self-play e liga (heurísticas + eu atual + versões antigas) com REINFORCE (baseline por média móvel ou grupo),
    Deep Monte Carlo (Q regredido ao retorno, ε-guloso) ou PPO (crítico, razão clipada, KL à rede inicial). Com `mesas`, cada jogo sorteia o tamanho da mesa dessa lista."""
    saida.mkdir(parents=True, exist_ok=True)
    rng = random.Random(semente)
    rede = Rede.carregar(inicial) if inicial else Rede.nova(rng, ocultos, zero=residual > 0)
    rede.residual = residual
    heuristicas = [*ESTRATEGIAS, *liga]
    antigas: list[list[float]] = []
    referencia = Rede(list(rede.theta), rede.ocultos, residual)
    critico = Rede.nova(random.Random(semente + 1), rede.ocultos, zero=True)
    passos = 0
    baseline = 0.0
    contador = 0
    with Pool(processos) as pool, (saida / "treino.jsonl").open("w") as log:
        for it in range(1, iteracoes + 1):
            tarefas = []
            tamanhos = []
            for k in range(jogos // grupo):
                n = rng.choice(mesas) if mesas else jogadores
                tamanhos.append(n)
                especificacoes = []
                for _ in range(n - 1):
                    sorteio = rng.random()
                    if sorteio < 0.5 or (sorteio >= 0.75 and not antigas):
                        especificacoes.append(rng.choice(heuristicas))
                    elif sorteio < 0.75:
                        especificacoes.append((list(rede.theta), True))
                    else:
                        especificacoes.append((list(rng.choice(antigas)), False))
                contador += 1
                assento = rng.randrange(n)
                tarefas += [(list(rede.theta), rede.ocultos, especificacoes, 5_000_000 + contador, assento, desempate, v, rearranjar, forma, residual,
                             epsilon if algoritmo == "dmc" else None)
                            for v in range(grupo)]
            resultados = pool.map(_jogar_treino, tarefas)
            grad = np.zeros(len(rede.theta))
            decisoes = 0
            perda_valor = None
            retornos = [r for saida_jogo, _, _ in resultados for r, _ in saida_jogo]
            if algoritmo == "reinforce":
                for inicio in range(0, len(resultados), grupo):
                    bloco = resultados[inicio:inicio + grupo]
                    relativas = vantagens_grupo([jogo[0][0][0] for jogo in bloco]) if grupo > 1 else None
                    for v, (saida_jogo, _, _) in enumerate(bloco):
                        for lado, (retorno, trajetoria) in enumerate(saida_jogo):
                            vantagem = relativas[v] if relativas and lado == 0 else retorno - baseline
                            for (X, indice), desloc, _, _ in trajetoria:
                                g = rede.gradiente(X, indice, vantagem, beta, desloc)
                                grad += g
                                decisoes += 1
                if decisoes:
                    passos += 1
                    rede.passo(grad / decisoes, passos, lr)
            else:
                dados = [(X, indice, est, logp, retorno) for saida_jogo, _, _ in resultados
                         for retorno, trajetoria in saida_jogo for (X, indice), _, est, logp in trajetoria]
                decisoes = len(dados)
                if dados and algoritmo == "dmc":
                    perda_valor = sum((rede.pontuacoes([X[i]])[0] - G) ** 2 for X, i, _, _, G in dados) / decisoes
                    for _ in range(epocas):
                        grad = np.zeros(len(rede.theta))
                        for X, i, _, _, G in dados:
                            grad += rede.gradiente_q(X[i], G)
                        passos += 1
                        rede.passo(grad / decisoes, passos, lr)
                elif dados:
                    valores = [critico.pontuacoes([est])[0] for _, _, est, _, _ in dados]
                    perda_valor = sum((v - G) ** 2 for v, (*_, G) in zip(valores, dados)) / decisoes
                    vant = [G - v for v, (*_, G) in zip(valores, dados)]
                    media = sum(vant) / decisoes
                    desvio = math.sqrt(sum((a - media) ** 2 for a in vant) / decisoes) or 1.0
                    vant = [(a - media) / desvio for a in vant]
                    log_refs = [[math.log(q) for q in referencia.probabilidades(X)] for X, *_ in dados]
                    for _ in range(epocas):
                        grad, grad_v = np.zeros(len(rede.theta)), np.zeros(len(critico.theta))
                        for (X, i, est, logp, G), a, lr_ in zip(dados, vant, log_refs):
                            g = gradiente_ppo(rede, X, i, a, logp, clip, beta, kl, lr_)
                            grad += g
                            grad_v += critico.gradiente_q(est, G)
                        passos += 1
                        rede.passo(grad / decisoes, passos, lr)
                        critico.passo(grad_v / decisoes, passos, lr)
            baseline = 0.95 * baseline + 0.05 * (sum(retornos) / len(retornos))
            if it % 10 == 0:
                antigas = (antigas + [list(rede.theta)])[-10:]
            contra_heuristica = [v for (_, v, _), t in zip(resultados, tarefas) if all(isinstance(e, str) for e in t[2])]
            registro = {"iteracao": it, "vitorias": sum(v for _, v, _ in resultados) / len(resultados),
                        "vitorias_heuristicas": sum(contra_heuristica) / len(contra_heuristica) if contra_heuristica else None,
                        "n_heuristicas": len(contra_heuristica), "decisoes": decisoes, "baseline": round(baseline, 4),
                        "grupo": grupo, "rearranjar": rearranjar, "forma": forma, "liga": list(liga), "residual": residual,
                        "algoritmo": algoritmo, "mesas": list(mesas), "tamanhos": tamanhos}
            if perda_valor is not None:
                registro["perda_valor"] = round(perda_valor, 4)
            log.write(json.dumps(registro) + "\n")
            log.flush()
            if it % checkpoint_a_cada == 0:
                rede.salvar(saida / f"iter_{it:06d}.json", iteracao=it)
    rede.salvar(saida / "final.json", iteracao=iteracoes)
    return rede
