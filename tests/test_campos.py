"""Plano 21: campos de oponentes para o torneio de mesa.

A comparação entre candidatos só é pareada se todos enfrentarem EXATAMENTE os mesmos
oponentes na mesma semente. O campo "misto" do artigo sorteava entre as estratégias
restantes, excluindo o foco, então o campo mudava de uma linha para outra. Aqui o campo
depende só de (campo, n, semente) e de um conjunto fixo de estratégias, nunca do candidato.
"""
import unittest
from collections import Counter

from src.estrategias.campos import CAMPOS, POOL_FORTE, oponentes_do_campo


class TestCampos(unittest.TestCase):
    def test_tamanho_e_determinismo(self):
        for n in (2, 3, 4, 6):
            for campo in CAMPOS:
                a = oponentes_do_campo(campo, n, 350_001)
                self.assertEqual(len(a), n - 1)
                self.assertEqual(a, oponentes_do_campo(campo, n, 350_001))

    def test_iguais_e_so_max_pecas(self):
        for n in (2, 3, 4):
            self.assertEqual(oponentes_do_campo("iguais", n, 7), ["max_pecas"] * (n - 1))

    def test_forte_sorteia_do_conjunto_fixo(self):
        for s in range(300):
            self.assertTrue(set(oponentes_do_campo("forte", 4, s)) <= set(POOL_FORTE))

    def test_conjunto_forte_declarado(self):
        self.assertEqual(sorted(POOL_FORTE),
                         sorted(["max_pecas", "max_pontos", "poupar_coringa", "poupar_descarta",
                                 "poupar_descarta_perto", "gastar_coringa"]))

    def test_campo_nao_depende_de_quem_joga(self):
        """A assinatura nem aceita o foco: o campo é função só de (campo, n, semente)."""
        import inspect
        self.assertEqual(list(inspect.signature(oponentes_do_campo).parameters), ["campo", "n", "semente"])

    def test_cobertura_uniforme_do_conjunto(self):
        """Com muitas sementes, cada estratégia aparece em ~1/6 dos assentos de oponente."""
        c = Counter(x for s in range(6000) for x in oponentes_do_campo("forte", 3, s))
        total = sum(c.values())
        for nome in POOL_FORTE:
            self.assertAlmostEqual(c[nome] / total, 1 / len(POOL_FORTE), delta=0.02)

    def test_semente_e_tamanho_mudam_o_campo(self):
        campos = {tuple(oponentes_do_campo("forte", 4, s)) for s in range(50)}
        self.assertGreater(len(campos), 20)
        self.assertNotEqual([oponentes_do_campo("forte", 3, s) for s in range(40)],
                            [oponentes_do_campo("forte", 4, s)[:2] for s in range(40)])

    def test_campo_desconhecido_falha_cedo(self):
        with self.assertRaises(ValueError):
            oponentes_do_campo("inventado", 3, 1)


if __name__ == "__main__":
    unittest.main()
