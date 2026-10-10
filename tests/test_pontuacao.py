import unittest

from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from src.jogo.pontuacao import pontos_oficiais
from tests.test_turnos import pecas


def mesa_vazia(maos, vencedor=None, desempate="pontos"):
    """Partida encerrada com mãos dadas; vencedor None = monte esgotado sem saída."""
    jogo = Jogo(semente=42, quantidade_jogadores=len(maos), desempate=desempate)
    for jogador, mao in zip(jogo.jogadores, maos):
        jogador.mao = mao
    jogo.monte = []
    jogo.terminou = True
    jogo.vencedor = jogo.jogadores[vencedor] if vencedor is not None else None
    return jogo


class TestPontuacaoOficial(unittest.TestCase):
    def test_vencedor_soma_as_pecas_dos_oponentes(self):
        jogo = mesa_vazia([[], pecas([("azul", 5), ("verde", 7)])], vencedor=0)
        self.assertEqual(pontos_oficiais(jogo), [12.0, -12.0])

    def test_coringa_na_mao_vale_trinta(self):
        jogo = mesa_vazia([[], [Peca(id="c", cor="coringa", numero=0)]], vencedor=0)
        self.assertEqual(pontos_oficiais(jogo), [30.0, -30.0])

    def test_soma_zero_com_tres_jogadores(self):
        jogo = mesa_vazia([[], pecas([("azul", 5)], "a"), pecas([("verde", 7), ("azul", 1)], "b")], vencedor=0)
        self.assertEqual(pontos_oficiais(jogo), [13.0, -5.0, -8.0])
        self.assertEqual(sum(pontos_oficiais(jogo)), 0.0)

    def test_monte_esgotado_vence_a_menor_mao_e_ganha_as_diferencas(self):
        # Mãos de 3, 10 e 20 pontos: o menor ganha (10-3) + (20-3) = 24.
        jogo = mesa_vazia([pecas([("azul", 3)], "a"), pecas([("verde", 10)], "b"), pecas([("azul", 7), ("verde", 13)], "c")])
        self.assertEqual(pontos_oficiais(jogo), [24.0, -7.0, -17.0])
        self.assertEqual(sum(pontos_oficiais(jogo)), 0.0)

    def test_monte_esgotado_com_empate_divide_o_premio(self):
        # Duas mãos de 3 pontos e uma de 13: as duas menores dividem os 10 de diferença.
        jogo = mesa_vazia([pecas([("azul", 3)], "a"), pecas([("verde", 3)], "b"), pecas([("azul", 13)], "c")])
        self.assertEqual(pontos_oficiais(jogo), [5.0, 5.0, -10.0])
        self.assertEqual(sum(pontos_oficiais(jogo)), 0.0)

    def test_mao_vazia_do_vencedor_nao_e_penalizada(self):
        jogo = mesa_vazia([pecas([("azul", 9)]), []], vencedor=1)
        self.assertEqual(pontos_oficiais(jogo), [-9.0, 9.0])

    def test_soma_zero_em_partidas_reais(self):
        from src.estrategias.arena import jogar
        from src.estrategias.politicas import criar
        import random
        for semente in range(400_000, 400_005):
            rng = random.Random(semente)
            resultado = jogar([criar("max_pecas", rng), criar("poupar_coringa", rng)], semente)
            self.assertAlmostEqual(sum(resultado["pontos_oficiais"]), 0.0)
            self.assertEqual(len(resultado["pontos_oficiais"]), 2)

    def test_pontos_do_vencedor_sao_positivos_e_dos_perdedores_negativos(self):
        from src.estrategias.arena import jogar
        from src.estrategias.politicas import criar
        import random
        for semente in range(400_010, 400_020):
            rng = random.Random(semente)
            resultado = jogar([criar("max_pecas", rng), criar("poupar_coringa", rng)], semente)
            if resultado["vencedor"] is None:
                continue
            pontos = resultado["pontos_oficiais"]
            self.assertGreater(pontos[resultado["vencedor"]], 0.0)
            self.assertTrue(all(p < 0 for i, p in enumerate(pontos) if i != resultado["vencedor"]))


if __name__ == "__main__":
    unittest.main()
