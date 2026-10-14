"""Plano 20: `base_semente` em `treinar_rede`.

Em `treinar_rede` as mãos de treino eram `5_000_000 + contador`, fixas. Por isso as três
sementes de treino dos planos 13, 15 e 17 jogavam as MESMAS mãos de treino, e o plano 17
chegou a afirmar "faixa nova" sem que fosse verdade. O plano 20 precisa de sementes
independentes de ponta a ponta, então a base passa a ser parâmetro, com o padrão antigo.

A garantia que protege os resultados publicados: sem o parâmetro, os pesos treinados são
idênticos aos de antes da mudança (hash congelado com o código anterior).
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src import experimentos
from src.estrategias import rede

# Congelado ANTES da mudança, com `iteracoes=3, jogos=8, ocultos=4, semente=11`, liga max_pecas,
# grupo 4, recompensa pontos; depende do número de entradas ativo no módulo.
HASH_ANTERIOR = {
    15: "ba45d6acc17f2e64973fd1ea9ea2a3c6383ac81f4367321ac2a0fb23183a4959",
    23: "d25bc3b512992389e7080f22d71c046e2097c4c6f066b85260ecf6e1e706d2a6",
}


def _treinar(d, nome, **extra):
    return rede.treinar_rede(Path(d) / nome, iteracoes=3, jogos=8, processos=1, ocultos=4, semente=11,
                             liga=("max_pecas",), grupo=4, recompensa="pontos", **extra)


def _hash(r):
    return hashlib.sha256(json.dumps([round(x, 12) for x in r.theta]).encode()).hexdigest()


class TestBaseSemente(unittest.TestCase):
    def test_padrao_reproduz_os_pesos_anteriores_a_mudanca(self):
        esperado = HASH_ANTERIOR.get(rede.ENTRADAS)
        if esperado is None:
            self.skipTest(f"sem hash congelado para ENTRADAS={rede.ENTRADAS}")
        with TemporaryDirectory() as d:
            self.assertEqual(_hash(_treinar(d, "a")), esperado)

    def test_base_explicita_igual_ao_padrao(self):
        with TemporaryDirectory() as d:
            self.assertEqual(list(_treinar(d, "a").theta),
                             list(_treinar(d, "b", base_semente=5_000_000).theta))

    def test_outra_base_muda_o_treino(self):
        with TemporaryDirectory() as d:
            self.assertNotEqual(list(_treinar(d, "a").theta),
                                list(_treinar(d, "b", base_semente=20_000_000).theta))

    def test_manifest_registra_a_base(self):
        with TemporaryDirectory() as d:
            _treinar(d, "a")
            _treinar(d, "b", base_semente=40_100_000)
            ra = json.loads((Path(d) / "a" / "treino.jsonl").read_text().splitlines()[0])
            rb = json.loads((Path(d) / "b" / "treino.jsonl").read_text().splitlines()[0])
            self.assertEqual(ra["base_semente"], 5_000_000)
            self.assertEqual(rb["base_semente"], 40_100_000)

    def test_subcomando_rede_aceita_base_semente(self):
        with TemporaryDirectory() as d:
            saida = Path(d) / "s"
            argv = ["experimentos", "rede", "--iteracoes", "1", "--jogos", "4", "--processos", "1",
                    "--ocultos", "4", "--semente", "3", "--grupo", "4", "--base-semente", "20100000",
                    "--saida", str(saida)]
            with mock.patch.object(sys, "argv", argv):
                experimentos.main()
            r = json.loads((saida / "treino.jsonl").read_text().splitlines()[0])
            self.assertEqual(r["base_semente"], 20_100_000)


if __name__ == "__main__":
    unittest.main()
