import unittest
from pathlib import Path

from src.animacao.pipeline import (ESTRATEGIAS_ANIMADAS, comando_gif, comando_manim, escolher_semente,
                                   nome_arquivo)


class TestNomes(unittest.TestCase):
    def test_doze_estrategias_sem_repeticao(self):
        self.assertEqual(len(ESTRATEGIAS_ANIMADAS), 12)
        self.assertEqual(len({nome_arquivo(e) for e in ESTRATEGIAS_ANIMADAS}), 12)

    def test_nome_de_arquivo_sem_separadores(self):
        for e in ESTRATEGIAS_ANIMADAS:
            n = nome_arquivo(e)
            self.assertTrue(n.replace("_", "").isalnum(), n)


class TestComandos(unittest.TestCase):
    def test_manim_final_e_teste(self):
        final = comando_manim("m", Path("/x/saida"), final=True)
        teste = comando_manim("m", Path("/x/saida"), final=False)
        self.assertIn("1080,1350", final)
        self.assertEqual(final[final.index("--fps") + 1], "30")
        self.assertEqual(teste[teste.index("--fps") + 1], "15")
        self.assertIn("--disable_caching", final)
        self.assertEqual(final[final.index("--progress_bar") + 1], "none")
        self.assertEqual(final[final.index("-o") + 1], "m")
        self.assertEqual(final[-1], "Partida")

    def test_gif_limita_duracao_e_largura(self):
        c = comando_gif(Path("a.mp4"), Path("a.gif"), segundos=20)
        texto = " ".join(c)
        self.assertIn("-t 20", texto)
        self.assertIn("scale=540", texto)
        self.assertIn("fps=12", texto)
        self.assertIn("palettegen", texto)


class TestSemente(unittest.TestCase):
    def test_primeira_semente_em_que_todas_cabem(self):
        duracoes = {110_000: {"a": 90, "b": 130}, 110_001: {"a": 150, "b": 80}, 110_002: {"a": 100, "b": 120}}
        chamadas = []

        def decisoes(estrategia, semente):
            chamadas.append((estrategia, semente))
            return duracoes[semente][estrategia]

        escolhida, relatorio = escolher_semente(["a", "b"], 110_000, 120, decisoes)
        self.assertEqual(escolhida, 110_002)
        self.assertEqual(relatorio[110_000], {"b": 130})
        self.assertEqual(relatorio[110_001], {"a": 150})
        self.assertNotIn(("b", 110_001), chamadas)

    def test_desiste_apos_tentativas(self):
        with self.assertRaises(RuntimeError):
            escolher_semente(["a"], 1, 10, lambda e, s: 99, tentativas=3)


if __name__ == "__main__":
    unittest.main()
