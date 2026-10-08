import random
import unittest

from src.estrategias.arena import jogar
from src.estrategias.politicas import EXTRAS, criar
from tests.test_turnos import partida, pecas


def escolher(nome, jogo, acoes, semente=0):
    return criar(nome, random.Random(semente))(jogo, acoes)


class TestExtras(unittest.TestCase):
    def test_partida_completa_conserva_pecas_e_so_escolhe_acoes_validas(self):
        for nome in EXTRAS:
            with self.subTest(nome=nome):
                r = jogar([criar(nome, random.Random(1)), criar('max_pecas', random.Random(2))], semente=3)
                self.assertEqual(r['pecas_total'], 106)
                self.assertGreater(r['decisoes'], 0)

    def test_adaptativo_so_gasta_coringa_quando_oponente_esta_perto_de_sair(self):
        mao = pecas([('azul', 5), ('azul', 6), ('coringa', 0), ('verde', 1)])
        jogo = partida(mao, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['p0', 'p1', 'p2']}, {'tipo': 'comprar', 'ids_pecas': []}]
        jogo.jogadores[1].mao = pecas([('azul', 1)] * 5, 'o')
        self.assertEqual(escolher('adaptativo', jogo, acoes)['tipo'], 'comprar')
        jogo.jogadores[1].mao = pecas([('azul', 1)] * 4, 'o')
        self.assertEqual(escolher('adaptativo', jogo, acoes)['tipo'], 'baixar')

    def test_poupar_ate_n_libera_o_coringa_quando_a_mao_chega_a_n(self):
        mao = pecas([('azul', 4), ('coringa', 0), ('verde', 9)])
        jogo = partida(mao, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['p0']}, {'tipo': 'baixar', 'ids_pecas': ['p0', 'p1']},
                 {'tipo': 'comprar', 'ids_pecas': []}]
        self.assertEqual(escolher('poupar_ate_3', jogo, acoes)['ids_pecas'], ['p0', 'p1'])
        self.assertEqual(escolher('poupar_ate_2', jogo, acoes)['ids_pecas'], ['p0'])
        self.assertEqual(escolher('poupar_coringa', jogo, acoes)['ids_pecas'], ['p0'])

    def test_reorganizador_prefere_mover_coringa_da_mesa_a_mais_pecas(self):
        mesa = [pecas([('verde', 4), ('coringa', 0), ('verde', 6)], 'm')]
        mao = pecas([('verde', 5), ('azul', 5), ('amarelo', 5), ('azul', 9), ('azul', 10), ('azul', 11),
                     ('azul', 12)])
        jogo = partida(mao, mesa, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['p3', 'p4', 'p5', 'p6']},
                 {'tipo': 'mesa', 'ids_pecas': ['p0', 'p1', 'p2'],
                  'combinacoes': [['m0', 'p0', 'm2'], ['p1', 'p2', 'm1']]},
                 {'tipo': 'comprar', 'ids_pecas': []}]
        self.assertEqual(len(escolher('max_pecas', jogo, acoes)['ids_pecas']), 4)
        self.assertEqual(escolher('reorganizador', jogo, acoes)['tipo'], 'mesa')

    def test_flexivel_desempata_pela_mao_que_sobra(self):
        mao = pecas([('azul', 1), ('azul', 2), ('azul', 3), ('verde', 9), ('verde', 10), ('verde', 11),
                     ('azul', 4), ('azul', 5)])
        jogo = partida(mao, abriu=True)
        joga_a = {'tipo': 'baixar', 'ids_pecas': ['p0', 'p1', 'p2']}
        joga_b = {'tipo': 'baixar', 'ids_pecas': ['p3', 'p4', 'p5']}
        for semente in range(8):
            for acoes in ([joga_a, joga_b], [joga_b, joga_a]):
                with self.subTest(semente=semente):
                    self.assertEqual(escolher('flexivel', jogo, acoes + [{'tipo': 'comprar', 'ids_pecas': []}],
                                              semente)['ids_pecas'], ['p3', 'p4', 'p5'])


if __name__ == '__main__':
    unittest.main()


class TestLiga(unittest.TestCase):
    def test_liga_tem_12_estrategias_66_pares_e_sementes_novas(self):
        from src.experimentos import tarefas_liga
        tarefas = tarefas_liga(3)
        self.assertEqual(len(tarefas), 66 * 3 * 2)
        self.assertEqual(len(set(tarefas)), len(tarefas))
        self.assertEqual({t[2] for t in tarefas}, {140_000, 140_001, 140_002})
        self.assertEqual({t[6] for t in tarefas}, {'pontos'})
