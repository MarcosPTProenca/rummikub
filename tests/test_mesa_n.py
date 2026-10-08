import random
import unittest

from src.estrategias.arena import jogar
from src.estrategias.politicas import criar
from src.experimentos import partida_mesa, pontos_mesa, tarefas_mesa


class TestArenaN(unittest.TestCase):
    def test_arena_joga_3_e_4_jogadores_conservando_pecas(self):
        for n in (3, 4):
            with self.subTest(n=n):
                r = jogar([criar('max_pecas', random.Random(i)) for i in range(n)], semente=120_000)
                self.assertEqual(r['pecas_total'], 106)
                self.assertEqual(len(r['telemetria']), n)
                self.assertEqual(len(r['maos']), n)
                self.assertFalse(r['truncada'])
                self.assertIn(r['vencedor'], list(range(n)) + [None])


class TestPartidaMesa(unittest.TestCase):
    def test_pontos_mesa(self):
        self.assertEqual(pontos_mesa(2, 2, 3), 1.0)
        self.assertEqual(pontos_mesa(1, 2, 3), 0.0)
        self.assertAlmostEqual(pontos_mesa(None, 2, 4), 0.25)

    def test_registro_da_mesa_focal(self):
        r = partida_mesa('poupar_coringa', ['max_pecas', 'aleatorio'], 120_000, 1)
        self.assertEqual((r['n'], r['assento_foco'], r['foco']), (3, 1, 'poupar_coringa'))
        self.assertIn(r['pontos_foco'], (0.0, 1.0, 1 / 3))
        self.assertEqual(r['saida_com_coringa_foco'] in (True, False), True)
        self.assertEqual(r, partida_mesa('poupar_coringa', ['max_pecas', 'aleatorio'], 120_000, 1))

    def test_oponentes_ocupam_os_outros_assentos_em_ordem(self):
        r = partida_mesa('max_pecas', ['aleatorio', 'minimo', 'so_baixar'], 120_001, 2)
        self.assertEqual(r['mesa'], ['aleatorio', 'minimo', 'max_pecas', 'so_baixar'])


class TestTarefasMesa(unittest.TestCase):
    def test_contagem_e_rotacao_do_assento(self):
        t = tarefas_mesa(3, ['max_pecas', 'poupar_coringa'], ['iguais', 'misto'], sementes=4)
        self.assertEqual(len(t), 2 * 2 * 4 * 3)
        self.assertEqual({x[3] for x in t}, {0, 1, 2})

    def test_campo_iguais_repete_max_pecas(self):
        t = tarefas_mesa(4, ['poupar_coringa'], ['iguais'], sementes=1)
        self.assertEqual({x[1] for x in t}, {'max_pecas|max_pecas|max_pecas'})

    def test_campo_misto_exclui_o_foco_nao_repete_e_e_deterministico(self):
        t = tarefas_mesa(4, ['poupar_coringa'], ['misto'], sementes=10)
        self.assertEqual(t, tarefas_mesa(4, ['poupar_coringa'], ['misto'], sementes=10))
        for x in t:
            ops = x[1].split('|')
            self.assertEqual(len(ops), 3)
            self.assertEqual(len(set(ops)), 3)
            self.assertNotIn('poupar_coringa', ops)

    def test_oponentes_sao_os_mesmos_nas_rotacoes_da_mesma_semente(self):
        t = tarefas_mesa(3, ['poupar_coringa'], ['misto'], sementes=3)
        por_semente = {}
        for x in t:
            por_semente.setdefault(x[2], set()).add(x[1])
        self.assertTrue(all(len(v) == 1 for v in por_semente.values()))


if __name__ == '__main__':
    unittest.main()
