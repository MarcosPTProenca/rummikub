import itertools
import unittest

from src.animacao.layout import ALTURA, BASE_H, BASE_W, LARGURA, layout

CORES = ["azul", "vermelho", "verde", "amarelo"]


def pecas(n, prefixo):
    return [[f"{prefixo}{i}", CORES[i % 4], 1 + (i * 7) % 13] for i in range(n)]


def estado(mao_a=14, mao_b=14, mesa=(), monte=20):
    return {"maos": [pecas(mao_a, "a"), pecas(mao_b, "b")], "mesa": [pecas(n, f"s{i}_") for i, n in enumerate(mesa)],
            "monte": monte, "abriu": [False, False], "pontos": [0, 0]}


def caixa(pos):
    x, y, s = pos
    return (x - BASE_W * s / 2, y - BASE_H * s / 2, x + BASE_W * s / 2, y + BASE_H * s / 2)


def todas_as_pecas(e):
    return [p[0] for m in e["maos"] for p in m] + [p[0] for c in e["mesa"] for p in c]


class TestLayout(unittest.TestCase):
    def verificar(self, e, baixo=0):
        pos = layout(e, baixo)["pecas"]
        ids = todas_as_pecas(e)
        self.assertEqual(sorted(pos), sorted(ids))
        caixas = {i: caixa(p) for i, p in pos.items()}
        for i, (x0, y0, x1, y1) in caixas.items():
            self.assertGreaterEqual(x0, -LARGURA / 2, i)
            self.assertLessEqual(x1, LARGURA / 2, i)
            self.assertGreaterEqual(y0, -ALTURA / 2, i)
            self.assertLessEqual(y1, ALTURA / 2, i)
        for (i, a), (j, b) in itertools.combinations(caixas.items(), 2):
            separadas = a[2] <= b[0] + 1e-9 or b[2] <= a[0] + 1e-9 or a[3] <= b[1] + 1e-9 or b[3] <= a[1] + 1e-9
            self.assertTrue(separadas, f"{i} sobrepoe {j}")
        return pos

    def test_estado_inicial_cabe_sem_sobreposicao(self):
        self.verificar(estado())

    def test_mesa_com_60_pecas_cabe(self):
        self.verificar(estado(mesa=[3, 4, 5, 3, 3, 4, 3, 5, 4, 3, 3, 4, 3, 5, 3]))

    def test_mesa_com_90_pecas_cabe(self):
        self.verificar(estado(mao_a=5, mao_b=5, mesa=[6] * 15))

    def test_maos_grandes_cabem(self):
        self.verificar(estado(mao_a=34, mao_b=30))

    def test_escala_diminui_quando_a_mesa_enche(self):
        pouca = layout(estado(mesa=[3, 3]), 0)["pecas"]["s0_0"][2]
        muita = layout(estado(mao_a=5, mao_b=5, mesa=[6] * 15), 0)["pecas"]["s0_0"][2]
        self.assertGreater(pouca, muita)

    def test_zonas_na_ordem_oponente_mesa_jogador(self):
        e = estado(mesa=[3, 4])
        pos = layout(e, 0)["pecas"]
        y_baixo = max(pos[p[0]][1] for p in e["maos"][0])
        y_mesa = [pos[p[0]][1] for c in e["mesa"] for p in c]
        y_cima = min(pos[p[0]][1] for p in e["maos"][1])
        self.assertLess(y_baixo, min(y_mesa))
        self.assertLess(max(y_mesa), y_cima)

    def test_assento_do_jogador_estudado_decide_quem_fica_embaixo(self):
        e = estado()
        pos = layout(e, 1)["pecas"]
        self.assertLess(pos["b0"][1], pos["a0"][1])

    def test_pecas_do_mesmo_conjunto_ficam_na_mesma_linha_e_em_ordem(self):
        e = estado(mesa=[4])
        pos = layout(e, 0)["pecas"]
        xs = [pos[p[0]][0] for p in e["mesa"][0]]
        ys = {round(pos[p[0]][1], 6) for p in e["mesa"][0]}
        self.assertEqual(len(ys), 1)
        self.assertEqual(xs, sorted(xs))

    def test_posicao_do_monte_dentro_do_quadro(self):
        x, y = layout(estado(), 0)["monte"]
        self.assertTrue(abs(x) < LARGURA / 2 and abs(y) < ALTURA / 2)

    def test_mao_ordenada_por_cor_e_numero_com_coringa_no_fim(self):
        e = estado(mao_a=0)
        e["maos"][0] = [["c", "coringa", 0], ["x", "verde", 5], ["y", "azul", 9], ["z", "azul", 2]]
        pos = layout(e, 0)["pecas"]
        ordem = sorted(pos[i][0] for i in "cxyz")
        self.assertEqual([i for i in "cxyz" if pos[i][0] == ordem[0]], ["z"])
        self.assertEqual(max("cxyz", key=lambda i: pos[i][0]), "c")


if __name__ == "__main__":
    unittest.main()
