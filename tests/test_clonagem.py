import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.estrategias.clonagem import coletar, concordancia, treinar_clone
from src.estrategias.rede import ENTRADAS, Rede, vantagens_grupo


class TestColeta(unittest.TestCase):
    def test_grava_uma_amostra_por_decisao_com_escolha_e_tamanho_fixo(self):
        amostras = coletar('poupar_coringa', ['max_pecas'], sementes=2, inicio=6_000_000, maximo_acoes=8)
        self.assertGreater(len(amostras), 10)
        for X, indice in amostras:
            self.assertGreater(len(X), 1)
            self.assertLessEqual(len(X), 8)
            self.assertTrue(0 <= indice < len(X))
            self.assertTrue(all(len(x) == ENTRADAS for x in X))

    def test_a_coleta_e_deterministica(self):
        a = coletar('poupar_coringa', ['max_pecas'], sementes=1, inicio=6_000_000)
        b = coletar('poupar_coringa', ['max_pecas'], sementes=1, inicio=6_000_000)
        self.assertEqual(a, b)


class TestTreinoSupervisionado(unittest.TestCase):
    def test_aprende_uma_regra_sintetica(self):
        rng = random.Random(1)
        amostras = []
        for _ in range(200):
            X = [[rng.random() for _ in range(ENTRADAS)] for _ in range(4)]
            amostras.append((X, max(range(4), key=lambda k: X[k][2])))
        rede = Rede.nova(random.Random(0), ocultos=8)
        antes = concordancia(rede, amostras)
        treinar_clone(rede, amostras, epocas=60, lr=1e-2, rng=random.Random(2))
        self.assertGreater(concordancia(rede, amostras), max(0.9, antes))


class TestCli(unittest.TestCase):
    def test_clonar_grava_a_rede_e_o_log(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'clonar', '--partidas', '2', '--epocas', '1',
                            '--ocultos', '4', '--processos', '1', '--saida', str(pasta)], check=True, capture_output=True)
            self.assertTrue((pasta / 'final.json').exists())
            log = json.loads((pasta / 'clonagem.json').read_text())
            self.assertIn('concordancia_treino', log)
            self.assertIn('concordancia_teste', log)


class TestVantagemDeGrupo(unittest.TestCase):
    def test_centra_na_media_e_normaliza_pelo_desvio(self):
        v = vantagens_grupo([1.0, -1.0, -1.0, 1.0])
        self.assertEqual(v, [1.0, -1.0, -1.0, 1.0])
        v = vantagens_grupo([1.0, -1.0, -1.0, -1.0])
        self.assertAlmostEqual(sum(v), 0.0)
        self.assertGreater(v[0], 0)

    def test_grupo_sem_diferenca_nao_da_sinal(self):
        self.assertEqual(vantagens_grupo([-1.0, -1.0, -1.0]), [0.0, 0.0, 0.0])

    def test_cli_treina_com_grupos_da_mesma_mao(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--grupo', '2', '--iteracoes', '2',
                            '--jogos', '4', '--processos', '1', '--ocultos', '4', '--saida', str(pasta)],
                           check=True, capture_output=True)
            log = [json.loads(l) for l in (pasta / 'treino.jsonl').read_text().splitlines()]
            self.assertEqual(len(log), 2)
            self.assertEqual(log[0]['grupo'], 2)


class TestTreinoComRearranjos(unittest.TestCase):
    def test_cli_treina_com_rearranjos_ligados(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--rearranjar', '--iteracoes', '1',
                            '--jogos', '2', '--processos', '1', '--ocultos', '4', '--saida', str(pasta)],
                           check=True, capture_output=True)
            log = json.loads((pasta / 'treino.jsonl').read_text().splitlines()[0])
            self.assertTrue(log['rearranjar'])


if __name__ == '__main__':
    unittest.main()
