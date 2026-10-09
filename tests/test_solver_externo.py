import random
import unittest
from pathlib import Path

from src.estrategias.solver_externo import _id_solver, SolverExterno
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca

RAIZ = Path(__file__).resolve().parents[1]
TEM_SOLVER = (RAIZ / ".venv-solver" / "bin" / "python").exists()


class TestMapeamento(unittest.TestCase):
    def test_id_do_solver_cobre_cores_numeros_e_coringa(self):
        # azul=bloco 0, vermelho=1, verde=2, amarelo=3; coringa=53
        self.assertEqual(_id_solver({"cor": "azul", "numero": 1}), 1)
        self.assertEqual(_id_solver({"cor": "azul", "numero": 13}), 13)
        self.assertEqual(_id_solver({"cor": "vermelho", "numero": 1}), 14)
        self.assertEqual(_id_solver({"cor": "amarelo", "numero": 13}), 52)
        self.assertEqual(_id_solver({"cor": "coringa", "numero": 0}), 53)

    def test_ids_sao_unicos_para_as_104_pecas(self):
        vistos = {_id_solver({"cor": c, "numero": n})
                  for c in ("azul", "vermelho", "verde", "amarelo") for n in range(1, 14)}
        self.assertEqual(len(vistos), 52)


@unittest.skipUnless(TEM_SOLVER, "precisa do .venv-solver com rummikub-solver")
class TestSolverExterno(unittest.TestCase):
    def test_abre_quando_tem_um_grupo_de_30_mais(self):
        jogo = Jogo(quantidade_jogadores=2, semente=1)
        grupo = [Peca(id="A", cor="azul", numero=13), Peca(id="B", cor="vermelho", numero=13),
                 Peca(id="C", cor="verde", numero=13)]
        jogo.jogadores[0].mao = grupo + jogo.jogadores[0].mao[:11]
        escolha = SolverExterno()(jogo, jogo.listar_acoes_validas())
        self.assertEqual(escolha["tipo"], "baixar")
        self.assertGreaterEqual(len(escolha["ids_pecas"]), 3)

    def test_compra_quando_nao_ha_baixa(self):
        jogo = Jogo(quantidade_jogadores=2, semente=1)
        # mão sem nenhum conjunto de 30+: sete peças avulsas de números distintos
        jogo.jogadores[0].mao = [Peca(id=str(i), cor="azul", numero=i) for i in range(1, 8)]
        escolha = SolverExterno()(jogo, jogo.listar_acoes_validas())
        self.assertEqual(escolha["tipo"], "comprar")


if __name__ == "__main__":
    unittest.main()
