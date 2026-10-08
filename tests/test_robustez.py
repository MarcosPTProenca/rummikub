import unittest

from src.analise_robustez import familia, linhas_tabela, metricas
from src.estrategias.politicas import ESTRATEGIAS, EXTRAS
from src.experimentos import tarefas_robustez


class TestTarefas(unittest.TestCase):
    def test_cobre_os_doze_oponentes_nos_dois_assentos_sem_jogar_contra_si(self):
        t = tarefas_robustez(['rede:x.json', 'poupar_coringa'], sementes=3)
        self.assertEqual(len(t), 2 * (len(ESTRATEGIAS) + len(EXTRAS) - 0) * 3 * 2 - 3 * 2)
        self.assertEqual({x[1] for x in t if x[0] == 'poupar_coringa'}, {*ESTRATEGIAS, *EXTRAS} - {'poupar_coringa'})
        self.assertEqual({x[1] for x in t if x[0] == 'rede:x.json'}, {*ESTRATEGIAS, *EXTRAS})

    def test_sementes_comecam_em_170000_e_sao_as_mesmas_para_todos_os_agentes(self):
        t = tarefas_robustez(['a', 'b'], sementes=2)
        self.assertEqual({x[2] for x in t}, {170_000, 170_001})

    def test_perfil_e_repassado_na_oitava_posicao(self):
        t = tarefas_robustez(['a'], sementes=1, oponentes=['max_pecas'], perfil='x4')
        self.assertEqual(t[0][7], 'x4')


class TestFamilia(unittest.TestCase):
    def test_agrupa_as_tres_sementes_de_treino_e_deixa_heuristicas_intactas(self):
        self.assertEqual(familia('rede:resultados/rede_clone_g4_s2/final.json'), 'g4')
        self.assertEqual(familia('rede:resultados/rede_clone_rl_3e-3/final.json'), 'rl')
        self.assertEqual(familia('poupar_coringa'), 'poupar_coringa')


class TestMetricas(unittest.TestCase):
    def test_pior_caso_queda_e_medias_de_vistos_e_novos(self):
        celulas = {'max_pecas': 0.6, 'aleatorio': 0.8, 'adaptativo': 0.4, 'reorganizador': 0.5}
        m = metricas(celulas, vistos={'max_pecas', 'aleatorio'})
        self.assertAlmostEqual(m['vistos'], 0.7)
        self.assertAlmostEqual(m['novos'], 0.45)
        self.assertAlmostEqual(m['queda'], 0.25)
        self.assertEqual(m['pior'], ('adaptativo', 0.4))


class TestTabela(unittest.TestCase):
    def test_linhas_trazem_doze_oponentes_e_tres_resumos_com_traco_na_diagonal(self):
        c = {(f, o): [0.4, 0.6, 0.5] for f in ('g4', 'max_pecas') for o in [*ESTRATEGIAS, *EXTRAS] if o != f}
        linhas = linhas_tabela(c, ('g4', 'max_pecas'))
        self.assertEqual(len(linhas), 12 + 3)
        diagonal = next(l for l in linhas if l.startswith('max-tiles'))
        self.assertIn('--', diagonal)
        self.assertTrue(linhas[-1].startswith('Worst case'))


if __name__ == '__main__':
    unittest.main()
