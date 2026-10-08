import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src import experimentos
from src.animacao.replay import carregar_q, descrever_pecas, gravar, texto_da_jogada

RAIZ = Path(__file__).resolve().parents[1]


def pecas_do_estado(estado):
    return sorted(p[0] for m in estado["maos"] for p in m) + sorted(p[0] for c in estado["mesa"] for p in c)


class TestGravar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = gravar("max_pecas", "poupar_coringa", semente=110_000, assento_a=0)

    def test_mesmo_resultado_da_arena_para_a_mesma_semente(self):
        r = self.replay
        esperado = experimentos.partida("max_pecas", "poupar_coringa", 110_000, 0)
        self.assertEqual(len(r["passos"]), esperado["decisoes"])
        self.assertEqual(r["decisoes"], esperado["decisoes"])
        self.assertEqual(r["vencedor"] == r["assento_a"], esperado["pontos_a"] == 1.0)

    def test_conserva_todas_as_pecas_em_todos_os_passos(self):
        total = None
        for estado in [self.replay["inicio"]] + [p["estado"] for p in self.replay["passos"]]:
            noPlay = len(pecas_do_estado(estado)) + estado["monte"]
            total = total or noPlay
            self.assertEqual(noPlay, total)
        self.assertEqual(total, 106)

    def test_cada_passo_traz_jogador_tipo_e_texto_em_ingles(self):
        for passo in self.replay["passos"]:
            self.assertIn(passo["jogador"], (0, 1))
            self.assertIn(passo["tipo"], ("baixar", "mesa", "comprar"))
            self.assertTrue(passo["texto"].isascii() and passo["texto"])
        self.assertEqual({p["jogador"] for p in self.replay["passos"][:2]}, {0, 1})

    def test_passo_de_compra_aumenta_a_mao_e_diminui_o_monte(self):
        anterior = self.replay["inicio"]
        for passo in self.replay["passos"]:
            if passo["tipo"] == "comprar" and anterior["monte"] > 0:
                j = passo["jogador"]
                self.assertEqual(len(passo["estado"]["maos"][j]), len(anterior["maos"][j]) + 1)
                self.assertEqual(passo["estado"]["monte"], anterior["monte"] - 1)
            anterior = passo["estado"]

    def test_estado_final_tem_mao_vazia_do_vencedor_ou_monte_esgotado(self):
        r = self.replay
        final = r["passos"][-1]["estado"]
        if r["vencedor"] is not None and r["motivo"] == "mao_vazia":
            self.assertEqual(final["maos"][r["vencedor"]], [])

    def test_json_serializavel(self):
        json.dumps(self.replay)


class TestTexto(unittest.TestCase):
    def test_textos_basicos(self):
        self.assertEqual(texto_da_jogada("comprar", 0, 0, 0, guardou_coringa=False), "Draws a tile")
        self.assertEqual(texto_da_jogada("comprar", 0, 0, 0, guardou_coringa=True), "Draws a tile, keeping the joker")
        self.assertIn("3 tiles", texto_da_jogada("baixar", 3, 21, 0, guardou_coringa=False))
        self.assertIn("21 pts", texto_da_jogada("baixar", 3, 21, 0, guardou_coringa=False))
        self.assertIn("joker", texto_da_jogada("baixar", 3, 21, 1, guardou_coringa=False))
        self.assertTrue(texto_da_jogada("mesa", 2, 15, 0, guardou_coringa=False).startswith("Rearranges the table"))


class TestDescreverPecas(unittest.TestCase):
    def test_cor_e_numero_em_ingles_com_coringa(self):
        self.assertEqual(descrever_pecas([[1, "azul", 5], [2, "vermelho", 5], [3, "coringa", 0]]), "Blue 5, Red 5, Joker")

    def test_muitas_pecas_viram_resumo(self):
        pecas = [[i, "verde", i] for i in range(1, 8)]
        self.assertEqual(descrever_pecas(pecas), "Green 1, Green 2, Green 3, Green 4 +3 more")

    def test_ordena_por_cor_e_numero(self):
        self.assertEqual(descrever_pecas([[1, "verde", 9], [2, "azul", 12], [3, "azul", 3]]), "Blue 3, Blue 12, Green 9")


class TestPecasJogadasNoReplay(unittest.TestCase):
    def test_passo_de_jogada_lista_as_pecas_que_saíram_da_mao(self):
        r = gravar("max_pecas", "max_pecas", 110_000, 0)
        for passo in r["passos"]:
            if passo["tipo"] == "comprar":
                self.assertEqual(passo["pecas"], [])
            else:
                self.assertGreaterEqual(len(passo["pecas"]), 1)
                self.assertIn(passo["pecas"][0][1], ("azul", "vermelho", "verde", "amarelo", "coringa"))

    def test_texto_cita_as_pecas(self):
        r = gravar("max_pecas", "max_pecas", 110_000, 0)
        jogada = next(p for p in r["passos"] if p["tipo"] != "comprar")
        self.assertIn(" - ", jogada["texto"])


class TestCompraComJogadaPossivel(unittest.TestCase):
    def test_passo_marca_se_havia_jogada_disponivel(self):
        r = gravar("aleatorio", "max_pecas", 110_000, 0)
        for passo in r["passos"]:
            self.assertIn("tinha_jogada", passo)
            if passo["tipo"] != "comprar":
                self.assertTrue(passo["tinha_jogada"])

    def test_compra_com_jogada_disponivel_e_dita_na_legenda(self):
        compras = [p for s in range(110_000, 110_010) for p in gravar("aleatorio", "max_pecas", s, 0)["passos"]
                   if p["tipo"] == "comprar" and p["jogador"] == 0]
        com = [p for p in compras if p["tinha_jogada"]]
        sem = [p for p in compras if not p["tinha_jogada"]]
        self.assertTrue(com and sem)
        self.assertTrue(all("could have played" in p["texto"] for p in com))
        self.assertTrue(all("could have played" not in p["texto"] for p in sem))


class TestQ(unittest.TestCase):
    def test_reconstroi_a_tabela_com_tipos_originais(self):
        agente = carregar_q(RAIZ / "resultados" / "treino_pontos.jsonl", replica=0)
        (estado, opcao), valor = next(iter(agente.tabela_q.items()))
        self.assertIsInstance(estado[0], bool)
        self.assertIsInstance(estado[2], int)
        self.assertIsInstance(valor, float)


class TestCLI(unittest.TestCase):
    def test_replay_ponta_a_ponta_grava_json(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = Path(pasta) / "r.json"
            subprocess.run([sys.executable, "-m", "src.animacao", "replay", "--estrategia", "minimo", "--oponente",
                            "max_pecas", "--semente", "110001", "--saida", str(saida)], cwd=RAIZ, check=True,
                           capture_output=True, timeout=120)
            dados = json.loads(saida.read_text())
            self.assertEqual((dados["estrategia_a"], dados["estrategia_b"], dados["semente"]), ("minimo", "max_pecas", 110001))
            self.assertGreater(len(dados["passos"]), 5)


if __name__ == "__main__":
    unittest.main()
