import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestMesaJev(unittest.TestCase):
    def test_mesa_com_jev_meta_respeita_orcamento_zero_e_registra_contagem(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = Path(pasta) / 'm.jsonl'
            r = subprocess.run([sys.executable, '-m', 'src.experimentos', 'mesa', '--jogadores', '3', '--focos', 'jev_meta',
                                '--campos', 'iguais', '--sementes', '1', '--processos', '1', '--orcamento', '0',
                                '--saida', str(saida)], capture_output=True, text=True,
                               env={**os.environ, 'TYPESAFE_API_KEY': 'chave-falsa'})
            self.assertEqual(r.returncode, 0, r.stderr)
            registros = [json.loads(l) for l in saida.read_text().splitlines()]
            self.assertEqual(len(registros), 3)
            for reg in registros:
                self.assertEqual(reg['n'], 3)
                self.assertEqual(reg['jev']['chamadas'], 0)
                self.assertGreater(reg['jev']['fallbacks'], 0)
            self.assertIn('gasto Jev: US$ 0.0000', r.stdout)


if __name__ == '__main__':
    unittest.main()


class TestReferenciaDeOponentes(unittest.TestCase):
    def test_referencia_faz_o_foco_enfrentar_os_oponentes_do_foco_de_referencia(self):
        from src.experimentos import tarefas_mesa
        ref = tarefas_mesa(3, ['max_pecas'], ['misto'], 5)
        outro = tarefas_mesa(3, ['jev_meta'], ['misto'], 5, referencia='max_pecas')
        self.assertEqual([t[1:] for t in ref], [t[1:] for t in outro])
        self.assertNotEqual([t[1:] for t in ref], [t[1:] for t in tarefas_mesa(3, ['jev_meta'], ['misto'], 5)])
