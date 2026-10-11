import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.estrategias.rede import retorno_do_jogo


class TestRecompensaPontos(unittest.TestCase):
    def test_recompensa_de_vitoria_e_mais_um_para_o_vencedor(self):
        self.assertEqual(retorno_do_jogo("vitoria", 0, 0, [48.0, -48.0]), 1.0)
        self.assertEqual(retorno_do_jogo("vitoria", 1, 0, [48.0, -48.0]), -1.0)

    def test_recompensa_de_vitoria_e_zero_no_empate(self):
        self.assertEqual(retorno_do_jogo("vitoria", None, 0, [5.0, -5.0]), 0.0)

    def test_recompensa_de_pontos_usa_a_pontuacao_oficial_normalizada(self):
        self.assertAlmostEqual(retorno_do_jogo("pontos", 0, 0, [48.0, -48.0]), 0.8)
        self.assertAlmostEqual(retorno_do_jogo("pontos", 0, 1, [48.0, -48.0]), -0.8)

    def test_recompensa_de_pontos_pune_o_coringa_retido_mais_que_a_derrota_simples(self):
        # Perder com um coringa na mão (-30) deve doer mais que perder com peças pequenas (-6).
        com_coringa = retorno_do_jogo("pontos", 1, 0, [-33.0, 33.0])
        sem_coringa = retorno_do_jogo("pontos", 1, 0, [-6.0, 6.0])
        self.assertLess(com_coringa, sem_coringa)

    def test_recompensa_de_pontos_distingue_vitorias_de_tamanhos_diferentes(self):
        magra = retorno_do_jogo("pontos", 0, 0, [8.0, -8.0])
        gorda = retorno_do_jogo("pontos", 0, 0, [60.0, -60.0])
        self.assertLess(magra, gorda)
        self.assertEqual(retorno_do_jogo("vitoria", 0, 0, [8.0, -8.0]),
                         retorno_do_jogo("vitoria", 0, 0, [60.0, -60.0]))

    def test_recompensa_invalida_falha(self):
        with self.assertRaises(KeyError):
            retorno_do_jogo("pecas", 0, 0, [1.0, -1.0])

    def test_cli_aceita_recompensa_de_pontos_e_grava_no_log(self):
        with tempfile.TemporaryDirectory() as pasta:
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '2',
                            '--processos', '1', '--ocultos', '4', '--recompensa', 'pontos',
                            '--saida', pasta], check=True, capture_output=True)
            log = json.loads((Path(pasta) / 'treino.jsonl').read_text().splitlines()[0])
            self.assertEqual(log['recompensa'], 'pontos')

    def test_padrao_do_cli_continua_sendo_vitoria(self):
        with tempfile.TemporaryDirectory() as pasta:
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '2',
                            '--processos', '1', '--ocultos', '4', '--saida', pasta], check=True, capture_output=True)
            log = json.loads((Path(pasta) / 'treino.jsonl').read_text().splitlines()[0])
            self.assertEqual(log['recompensa'], 'vitoria')


if __name__ == '__main__':
    unittest.main()
