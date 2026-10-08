import unittest

from src.analise_mesa import linhas_aprendidas, por_semente


def reg(foco, mesa, semente, pontos, replica=None):
    r = {'n': len(mesa), 'foco': foco, 'mesa': mesa, 'semente': semente, 'pontos_foco': pontos, 'vencedor': 0}
    if replica is not None:
        r['replica'] = replica
    return r


class TestAprendidos(unittest.TestCase):
    def test_replicas_do_q_nao_se_misturam_na_mesma_semente(self):
        rs = [reg('rl_meta', ['rl_meta', 'max_pecas', 'max_pecas'], 1, p, rep) for rep, p in ((0, 1.0), (1, 0.0))]
        g = por_semente(rs)
        self.assertEqual(sorted(g[(3, 'iguais', 'rl_meta')].values()), [0.0, 1.0])

    def test_linhas_trazem_rede_e_q_com_as_quatro_colunas(self):
        grupos = {}
        for n in (3, 4):
            for c in ('iguais', 'misto'):
                for nome in ('rede', 'rl_meta'):
                    grupos[(n, c, nome)] = {s: float(s % 2) for s in range(10)}
        linhas = linhas_aprendidas(grupos)
        self.assertEqual(len(linhas), 2)
        self.assertTrue(linhas[0].startswith('network'))
        self.assertTrue(linhas[1].startswith('Q-learner'))
        self.assertEqual(linhas[0].count('$'), 8)


if __name__ == '__main__':
    unittest.main()
