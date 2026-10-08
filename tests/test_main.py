import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.main import main


ARGS = ['--episodios', '2', '--avaliacao', '3', '--limite-passos', '1', '--semente', '42']


class TestMain(unittest.TestCase):
    def test_main_conecta_treino_e_avaliacao(self):
        saida = io.StringIO()
        with redirect_stdout(saida), patch('sys.argv', ['rummikub', *ARGS]):
            main()
        resultado = json.loads(saida.getvalue())
        self.assertEqual(resultado['episodios_treinados'], 2)
        self.assertGreater(resultado['entradas_q'], 0)
        self.assertEqual(resultado['avaliacao']['truncados'], 3)
        self.assertEqual(resultado['avaliacao']['media_passos'], 1)

    def test_argumentos_invalidos(self):
        for opcao in ('--episodios', '--avaliacao', '--limite-passos'):
            with self.subTest(opcao=opcao):
                with redirect_stderr(io.StringIO()), patch('sys.argv', ['rummikub', opcao, '0']):
                    with self.assertRaises(SystemExit) as erro:
                        main()
                self.assertEqual(erro.exception.code, 2)

    def test_cli_real_reproduz_resultado(self):
        comando = [sys.executable, '-m', 'src.main', *ARGS]
        raiz = Path(__file__).resolve().parents[1]
        a = subprocess.run(comando, cwd=raiz, capture_output=True, text=True, check=True, timeout=30)
        b = subprocess.run(comando, cwd=raiz, capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual(a.stdout, b.stdout)
        self.assertEqual(json.loads(a.stdout)['avaliacao']['truncados'], 3)
        self.assertEqual(a.stderr, '')


    def test_cli_com_mesa_e_abertura_reproduz_entre_hash_seeds(self):
        comando = [sys.executable, '-m', 'src.main', '--episodios', '2', '--avaliacao', '2', '--semente', '42']
        raiz = Path(__file__).resolve().parents[1]
        resultados = []
        for hash_seed in ('1', '99'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            run = subprocess.run(comando, cwd=raiz, env=env, capture_output=True, text=True,
                                 check=True, timeout=40)
            self.assertEqual(run.stderr, '')
            resultados.append(json.loads(run.stdout))
        self.assertEqual(resultados[0], resultados[1])
        metricas = resultados[0]['avaliacao']
        self.assertEqual(metricas['vitorias'] + metricas['derrotas'], 2)
        self.assertEqual(metricas['empates'] + metricas['truncados'], 0)


if __name__ == '__main__':
    unittest.main()
