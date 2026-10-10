import unittest

from src.estrategias.contagem import contagem_escondidas, copias_por_tipo, raridade
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from tests.test_turnos import grupo, partida, pecas


class TestContagem(unittest.TestCase):
    def test_mesa_e_mao_descontam_das_copias(self):
        jogo = partida(pecas([('azul', 3)], 'm'), [grupo(3)])
        escondidas = contagem_escondidas(jogo)
        self.assertEqual(escondidas[('azul', 3)], 0)
        self.assertEqual(escondidas[('verde', 3)], 1)
        self.assertEqual(escondidas[('amarelo', 3)], 2)

    def test_so_pecas_normais_em_52_tipos(self):
        escondidas = contagem_escondidas(Jogo(semente=1))
        self.assertEqual(len(escondidas), 52)
        self.assertTrue(all(c >= 0 for c in escondidas.values()))

    def test_escondidas_sao_as_pecas_dos_oponentes_e_do_monte(self):
        jogo = Jogo(semente=7)
        oculto = [p for o in jogo.jogadores[1:] for p in o.mao] + jogo.monte
        esperado = sum(p.cor != 'coringa' for p in oculto)
        self.assertEqual(sum(contagem_escondidas(jogo).values()), esperado)

    def test_copias_crescem_com_os_conjuntos(self):
        self.assertEqual(copias_por_tipo(Jogo(semente=1)), 2)
        self.assertEqual(copias_por_tipo(Jogo(quantidade_jogadores=5, semente=1)), 4)

    def test_raridade_vai_de_0_a_1(self):
        jogo = partida(pecas([('azul', 3)], 'm'), [grupo(3)])
        escondidas = contagem_escondidas(jogo)
        self.assertEqual(raridade(jogo, escondidas, ('azul', 3)), 1.0)
        self.assertEqual(raridade(jogo, escondidas, ('verde', 3)), 0.5)
        self.assertEqual(raridade(jogo, escondidas, ('amarelo', 3)), 0.0)


if __name__ == '__main__':
    unittest.main()
