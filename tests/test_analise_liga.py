import unittest

from src.analise_liga import bradley_terry, ciclos, holm, p_valor


class TestAnaliseLiga(unittest.TestCase):
    def test_holm_e_mais_conservador_que_p_cru(self):
        self.assertEqual(holm({"a": 0.001, "b": 0.03, "c": 0.04}), {"a"})
        self.assertEqual(holm({"a": 0.001, "b": 0.01, "c": 0.04}), {"a", "b", "c"})

    def test_bradley_terry_ordena_pelo_confronto_direto(self):
        forca = bradley_terry({("x", "y"): 75, ("y", "x"): 25}, ["x", "y"])
        self.assertAlmostEqual(forca["x"] / forca["y"], 3.0, places=3)

    def test_p_valor_com_variancia_zero(self):
        self.assertEqual(p_valor([1.0, 1.0, 1.0]), 0.0)
        self.assertEqual(p_valor([0.5, 0.5, 0.5]), 1.0)

    def test_detecta_ciclo_pedra_papel_tesoura(self):
        self.assertEqual(len(ciclos({("a", "b"), ("b", "c"), ("c", "a")}, ["a", "b", "c"])), 1)
        self.assertEqual(ciclos({("a", "b"), ("b", "c"), ("a", "c")}, ["a", "b", "c"]), [])


if __name__ == "__main__":
    unittest.main()
