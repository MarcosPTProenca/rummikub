"""`Combinacao.valida` tem de concordar com `eh_valido` em todo caso possível.

A regra está escrita duas vezes: uma no modelo pydantic, que é a forma legível e
usada em `jogar_turno`, e outra como função estática, usada nos laços quentes de
`listar_acoes_validas`. Duplicação assim só se sustenta com um teste que force as
duas a andarem juntas.
"""
import itertools
import random
import unittest

from src.jogo.combinacao import Combinacao
from src.jogo.peca import Peca

CORES = ["azul", "vermelho", "verde", "amarelo"]


class TestValidaConcordaComEhValido(unittest.TestCase):
    def comparar(self, pecas, contexto=""):
        self.assertEqual(Combinacao(pecas=pecas).eh_valido(), Combinacao.valida(pecas),
                         f"divergiu em {contexto or [(p.cor, p.numero) for p in pecas]}")

    def test_exaustivo_ate_quatro_pecas(self):
        """Todas as combinações de cor e número até 4 peças, sem repetir id."""
        universo = [Peca(id=f"{c}{n}", cor=c, numero=n) for c in CORES for n in range(1, 14)]
        universo.append(Peca(id="j1", cor="coringa", numero=0))
        rng = random.Random(3)
        amostra = rng.sample(universo, 18)
        for tamanho in (0, 1, 2, 3, 4):
            for pecas in itertools.combinations(amostra, tamanho):
                self.comparar(list(pecas))

    def test_sequencias_longas_de_toda_cor(self):
        for cor in CORES:
            for inicio in range(1, 12):
                for fim in range(inicio, 14):
                    pecas = [Peca(id=f"{cor}{n}", cor=cor, numero=n)
                             for n in range(inicio, fim + 1)]
                    self.comparar(pecas, f"{cor} {inicio}-{fim}")

    def test_com_um_e_dois_coringas(self):
        rng = random.Random(5)
        for _ in range(600):
            tamanho = rng.randint(3, 8)
            quantos = rng.randint(1, 2)
            pecas = [Peca(id=f"c{j}", cor="coringa", numero=0) for j in range(quantos)]
            for i in range(tamanho - quantos):
                pecas.append(Peca(id=f"p{i}", cor=rng.choice(CORES), numero=rng.randint(1, 13)))
            self.comparar(pecas)

    def test_ids_repetidos_sao_invalidos_nas_duas_formas(self):
        pecas = [Peca(id="x", cor="azul", numero=5), Peca(id="x", cor="vermelho", numero=5),
                 Peca(id="y", cor="verde", numero=5)]
        self.comparar(pecas)
        self.assertFalse(Combinacao.valida(pecas))

    def test_grupo_de_cinco_e_sequencia_de_quatorze_sao_invalidos(self):
        cinco = [Peca(id=str(i), cor=c, numero=7)
                 for i, c in enumerate(CORES + ["azul"])]
        self.comparar(cinco)
        quatorze = [Peca(id=f"a{n}", cor="azul", numero=n) for n in range(1, 14)]
        quatorze.append(Peca(id="extra", cor="azul", numero=13))
        self.comparar(quatorze)

    def test_apenas_coringas(self):
        for quantos in (1, 2, 3, 4):
            pecas = [Peca(id=f"c{j}", cor="coringa", numero=0) for j in range(quantos)]
            self.comparar(pecas, f"{quantos} coringas")


class TestPontosInalterados(unittest.TestCase):
    def test_pontuacao_de_casos_conhecidos(self):
        grupo = Combinacao(pecas=[Peca(id="1", cor="azul", numero=7),
                                  Peca(id="2", cor="vermelho", numero=7),
                                  Peca(id="3", cor="verde", numero=7)])
        self.assertEqual(grupo.calcula_pontos(), 21)
        sequencia = Combinacao(pecas=[Peca(id="4", cor="azul", numero=3),
                                      Peca(id="5", cor="azul", numero=4),
                                      Peca(id="6", cor="azul", numero=5)])
        self.assertEqual(sequencia.calcula_pontos(), 12)


if __name__ == "__main__":
    unittest.main()
