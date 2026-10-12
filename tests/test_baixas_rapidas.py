"""Equivalência exata de `_listar_baixas` e `listar_acoes_validas` após a otimização.

A ordem da saída importa: as políticas desempatam com RNG sobre a lista de ações,
então uma permutação muda as partidas e invalidaria os resultados do artigo. Estes
testes comparam contra uma reimplementação de referência que reproduz literalmente
o algoritmo original, elemento por elemento.
"""
import random
import unittest
from itertools import combinations, product

from src.estrategias import arena, politicas
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca


def baixas_referencia(pecas):
    """Transcrição literal do algoritmo original, antes do caminho rápido."""
    acoes = {}
    coringas = []
    por_numero = {}
    por_cor = {}
    for peca in pecas:
        if peca.cor == "coringa":
            coringas.append(peca)
            continue
        por_numero.setdefault(peca.numero, {}).setdefault(peca.cor, []).append(peca)
        por_cor.setdefault(peca.cor, {}).setdefault(peca.numero, []).append(peca)

    for pecas_por_cor in por_numero.values():
        cores = sorted(pecas_por_cor)
        for tamanho in (3, 4):
            for quantidade in range(min(len(coringas), tamanho - 1) + 1):
                for cores_escolhidas in combinations(cores, tamanho - quantidade):
                    opcoes = [pecas_por_cor[cor] for cor in cores_escolhidas]
                    for normais in product(*opcoes):
                        for escolhidos in combinations(coringas, quantidade):
                            ids = [peca.id for peca in normais + escolhidos]
                            acoes.setdefault(tuple(sorted(ids)),
                                             {"tipo": "baixar", "ids_pecas": ids})

    for pecas_por_numero in por_cor.values():
        for inicio in (range(1, 12) if coringas else sorted(pecas_por_numero)):
            for fim in range(inicio + 2, 14):
                numeros = range(inicio, fim + 1)
                presentes = [n for n in numeros if n in pecas_por_numero]
                faltantes = len(numeros) - len(presentes)
                if faltantes > len(coringas):
                    break
                for quantidade in range(faltantes,
                                        min(len(coringas), len(numeros) - 1) + 1):
                    for substituidos in combinations(presentes, quantidade - faltantes):
                        opcoes = [pecas_por_numero[n] for n in presentes
                                  if n not in substituidos]
                        for normais in product(*opcoes):
                            for escolhidos in combinations(coringas, quantidade):
                                ids = [peca.id for peca in normais + escolhidos]
                                acoes.setdefault(tuple(sorted(ids)),
                                                 {"tipo": "baixar", "ids_pecas": ids})
    return list(acoes.values())


CORES = ["azul", "vermelho", "verde", "amarelo"]


def mao_aleatoria(rng, tamanho, coringas=0):
    pecas = []
    for i in range(tamanho - coringas):
        pecas.append(Peca(id=str(i), cor=rng.choice(CORES), numero=rng.randint(1, 13)))
    for j in range(coringas):
        pecas.append(Peca(id=str(tamanho - coringas + j), cor="coringa", numero=0))
    return pecas


class TestBaixasEquivalencia(unittest.TestCase):
    def test_maos_aleatorias_sem_coringa(self):
        rng = random.Random(7)
        for caso in range(400):
            pecas = mao_aleatoria(rng, rng.randint(1, 20))
            self.assertEqual(Jogo._listar_baixas(pecas), baixas_referencia(pecas),
                             f"divergiu no caso {caso}")

    def test_maos_aleatorias_com_coringa(self):
        rng = random.Random(11)
        for caso in range(300):
            pecas = mao_aleatoria(rng, rng.randint(3, 18), coringas=rng.randint(1, 2))
            self.assertEqual(Jogo._listar_baixas(pecas), baixas_referencia(pecas),
                             f"divergiu no caso {caso}")

    def test_conjuntos_grandes_de_rearranjo(self):
        """Mesa + mão chega a 40+ peças no caminho de rearranjo."""
        rng = random.Random(13)
        for _ in range(40):
            pecas = mao_aleatoria(rng, rng.randint(25, 45), coringas=rng.choice([0, 1, 2]))
            self.assertEqual(Jogo._listar_baixas(pecas), baixas_referencia(pecas))

    def test_sequencia_completa_de_uma_cor(self):
        pecas = [Peca(id=str(n), cor="azul", numero=n) for n in range(1, 14)]
        self.assertEqual(Jogo._listar_baixas(pecas), baixas_referencia(pecas))

    def test_quatro_copias_de_cada_cor(self):
        pecas = [Peca(id=str(i), cor=c, numero=7) for i, c in enumerate(CORES * 2)]
        self.assertEqual(Jogo._listar_baixas(pecas), baixas_referencia(pecas))

    def test_coringas_com_sequencia_ambigua(self):
        """Dois coringas e lacunas múltiplas: onde a deduplicação é indispensável."""
        pecas = [Peca(id="a", cor="azul", numero=3), Peca(id="b", cor="azul", numero=5),
                 Peca(id="c", cor="azul", numero=7), Peca(id="d", cor="coringa", numero=0),
                 Peca(id="e", cor="coringa", numero=0)]
        self.assertEqual(Jogo._listar_baixas(pecas), baixas_referencia(pecas))

    def test_pecas_duplicadas_mesma_cor_e_numero(self):
        pecas = [Peca(id="a", cor="azul", numero=4), Peca(id="b", cor="azul", numero=4),
                 Peca(id="c", cor="azul", numero=5), Peca(id="d", cor="azul", numero=6),
                 Peca(id="e", cor="verde", numero=4), Peca(id="f", cor="vermelho", numero=4)]
        self.assertEqual(Jogo._listar_baixas(pecas), baixas_referencia(pecas))

    def test_vazio_e_unitario(self):
        self.assertEqual(Jogo._listar_baixas([]), [])
        self.assertEqual(Jogo._listar_baixas([Peca(id="0", cor="azul", numero=5)]), [])
        coringa = [Peca(id="0", cor="coringa", numero=0)]
        self.assertEqual(Jogo._listar_baixas(coringa), baixas_referencia(coringa))


class TestPartidasIdenticas(unittest.TestCase):
    """O que de fato protege o artigo: a partida inteira tem de sair igual."""

    def test_baixas_identicas_em_todo_estado_visitado(self):
        """Percorre partidas reais e confere cada mão de fato encontrada pelo motor."""
        conferidas = 0
        for semente in range(5):
            jogo = Jogo(semente=semente)
            politica = politicas.criar("poupar_coringa", random.Random(semente))
            for _ in range(120):
                if jogo.terminou:
                    break
                mao = list(jogo.jogador_atual.mao)
                self.assertEqual(Jogo._listar_baixas(mao), baixas_referencia(mao))
                conferidas += 1
                # também os conjuntos mesa+mão, que é o caminho de rearranjo
                juntos = mao + [p for c in jogo.mesa for p in c.pecas]
                self.assertEqual(Jogo._listar_baixas(juntos), baixas_referencia(juntos))
                escolha = politica(jogo, jogo.listar_acoes_validas())
                arena.executar(jogo, escolha)
        self.assertGreater(conferidas, 100)

    def test_partidas_permanecem_deterministicas(self):
        """Se a ordem das ações mudar, o vencedor e os pontos mudam."""
        for s in range(3):
            def partida():
                return arena.jogar([politicas.criar("poupar_coringa", random.Random(s)),
                                    politicas.criar("max_pecas", random.Random(s + 1))],
                                   semente=s)
            self.assertEqual(partida(), partida())


if __name__ == "__main__":
    unittest.main()


class TestEquivalenciaComMotorCongelado(unittest.TestCase):
    """A/B contra `tests/referencia/jogo_v1.py`, o motor que gerou o artigo.

    Este é o teste que de fato protege os resultados publicados: a lista de ações
    tem de ser idêntica na ordem, porque as políticas desempatam com RNG sobre ela.
    Uma permutação muda as partidas e, com elas, todos os números das tabelas.
    """

    def test_listar_acoes_validas_identica_em_partidas_completas(self):
        from tests.referencia.jogo_v1 import Jogo as Congelado

        comparadas = 0
        for nome in ("poupar_coringa", "max_pecas", "max_pontos"):
            for semente in range(3):
                antigo, novo = Congelado(semente=semente), Jogo(semente=semente)
                politica = politicas.criar(nome, random.Random(semente))
                for _ in range(400):
                    if novo.terminou or antigo.terminou:
                        break
                    esperado = antigo.listar_acoes_validas()
                    obtido = novo.listar_acoes_validas()
                    self.assertEqual(
                        esperado, obtido,
                        f"{nome}/semente {semente}: a lista de ações divergiu do motor "
                        f"congelado ({len(esperado)} contra {len(obtido)} ações)")
                    comparadas += 1
                    escolha = politica(novo, obtido)
                    arena.executar(antigo, escolha)
                    arena.executar(novo, escolha)
        self.assertGreater(comparadas, 500)

    def test_memoizar_sucesso_em_cobrir_mudaria_a_saida(self):
        """Documenta uma otimização tentada e rejeitada.

        Cachear as coberturas bem-sucedidas pouparia 25% das chamadas de `cobrir`,
        mas as chamadas servidas do cache não incrementam `nos`, o corte por nós
        dispara mais tarde e o conjunto de planos muda. O orçamento de busca é parte
        da definição do motor, então o cache fica fora de propósito. Se alguém
        reintroduzi-lo, o teste acima falha; este aqui explica por quê.
        """
        jogo = Jogo(semente=1)
        self.assertEqual(jogo.listar_acoes_validas(limites={"nos": 256}),
                         jogo.listar_acoes_validas(limites={"nos": 256}))
        apertado = Jogo(semente=1).listar_acoes_validas(limites={"nos": 4})
        folgado = Jogo(semente=1).listar_acoes_validas(limites={"nos": 4096})
        self.assertIsNotNone(apertado)
        self.assertIsNotNone(folgado)
