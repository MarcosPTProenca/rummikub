import random
import unittest

from src.estrategias.contagem import contagem_escondidas, copias_por_tipo, potencial_ponderado, raridade
from src.estrategias.politicas import criar
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


def mesa_sem(tipos_esgotados):
    """Mesa com as duas cópias de cada tipo dado, em grupos de 3 (cada grupo traz 1 cópia dele)."""
    grupos = []
    for k, (cor, n) in enumerate(tipos_esgotados):
        outras = [c for c in ('azul', 'vermelho', 'verde', 'amarelo') if c != cor][:2]
        for r in range(2):
            grupos.append(pecas([(cor, n), (outras[0], n), (outras[1], n)], f'x{k}{r}'))
    return grupos


class TestPotencialPonderado(unittest.TestCase):
    def test_par_so_vale_se_ainda_da_para_completar(self):
        mao = pecas([('azul', 5), ('azul', 6)])
        cheio = {t: 2 for t in contagem_escondidas(partida([])).keys()}
        sem_completar = {**cheio, ('azul', 4): 0, ('azul', 7): 0}
        self.assertEqual(potencial_ponderado(mao, cheio, 2), 1.0)
        self.assertEqual(potencial_ponderado(mao, sem_completar, 2), 0.0)

    def test_grupo_usa_as_outras_cores(self):
        mao = pecas([('azul', 3), ('verde', 3)])
        cheio = dict.fromkeys(contagem_escondidas(partida([])), 2)
        parcial = {**cheio, ('vermelho', 3): 0, ('amarelo', 3): 1}
        self.assertEqual(potencial_ponderado(mao, cheio, 2), 1.0)
        self.assertEqual(potencial_ponderado(mao, parcial, 2), 0.5)

    def test_ignora_coringa(self):
        mao = pecas([('azul', 3), ('coringa', 0)])
        self.assertEqual(potencial_ponderado(mao, dict.fromkeys(contagem_escondidas(partida([])), 2), 2), 0.0)


class TestContaRara(unittest.TestCase):
    def test_prefere_manter_o_par_que_ainda_completa(self):
        mao = pecas([('azul', 5), ('azul', 6), ('azul', 7), ('verde', 9), ('verde', 10), ('verde', 11)], 'm')
        # azul 4 e azul 8 esgotados: os pares azuis não completam mais; os verdes sim
        jogo = partida(mao, mesa_sem([('azul', 4), ('azul', 8)]), abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['m0', 'm1', 'm2']},
                 {'tipo': 'baixar', 'ids_pecas': ['m3', 'm4', 'm5']}]
        for semente in range(8):
            escolhida = criar('conta_rara', random.Random(semente))(jogo, acoes)
            self.assertEqual(escolhida['ids_pecas'], ['m0', 'm1', 'm2'])

    def test_nao_troca_tamanho_por_potencial(self):
        mao = pecas([('azul', 5), ('azul', 6), ('azul', 7), ('verde', 9), ('verde', 10), ('verde', 11)], 'm')
        jogo = partida(mao, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['m0', 'm1']}, {'tipo': 'baixar', 'ids_pecas': ['m3', 'm4', 'm5']}]
        self.assertEqual(criar('conta_rara', random.Random(0))(jogo, acoes)['ids_pecas'], ['m3', 'm4', 'm5'])

    def test_compra_quando_nao_ha_jogada(self):
        jogo = partida(pecas([('azul', 5)], 'm'), abriu=True)
        compra = {'tipo': 'comprar', 'ids_pecas': []}
        self.assertEqual(criar('conta_rara', random.Random(0))(jogo, [compra]), compra)

    def test_joga_partida_completa_conservando_pecas(self):
        from src.estrategias.arena import jogar
        resultado = jogar([criar('conta_rara', random.Random(1)), criar('max_pecas', random.Random(2))], semente=3)
        self.assertEqual(resultado['pecas_total'], 106)
        self.assertGreater(resultado['decisoes'], 0)
