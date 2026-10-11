"""Testes da mesa humano-contra-agente.

A camada testada aqui é o que separa o frontend do motor: receber uma mesa
proposta (listas de IDs, como o jogador arrastou), devolver aceitação ou a razão
da recusa, e deixar os agentes jogarem até voltar a vez do humano.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.humano.mesa import Partida


def ids_por_peca(partida, assento=0):
    """Mapa (cor, numero) -> id, para montar jogadas nos testes sem depender de ordem."""
    mapa = {}
    for p in partida.jogo.jogadores[assento].mao:
        mapa.setdefault((p.cor, p.numero), []).append(p.id)
    return mapa


class TestEstado(unittest.TestCase):
    def test_estado_mostra_a_mao_do_humano_e_esconde_a_do_agente(self):
        p = Partida(["max_pecas"], semente=1)
        e = p.estado()
        self.assertEqual(len(e["mao"]), 14)
        self.assertEqual(e["oponentes"], [{"assento": 1, "nome": "max-tiles", "pecas": 14, "abriu": False}])
        self.assertNotIn("mao", e["oponentes"][0])

    def test_estado_traz_a_mesa_o_monte_e_de_quem_e_a_vez(self):
        p = Partida(["max_pecas"], semente=1)
        e = p.estado()
        self.assertEqual(e["mesa"], [])
        self.assertEqual(e["monte"], 106 - 28)
        self.assertEqual(e["vez"], 0)
        self.assertFalse(e["terminou"])

    def test_humano_senta_no_assento_zero_e_comeca(self):
        p = Partida(["max_pecas"], semente=1)
        self.assertEqual(p.humano, 0)
        self.assertEqual(p.estado()["vez"], 0)

    def test_pecas_do_estado_tem_id_cor_e_numero(self):
        p = Partida(["max_pecas"], semente=1)
        peca = p.estado()["mao"][0]
        self.assertEqual(set(peca), {"id", "cor", "numero"})

    def test_aceita_varios_oponentes(self):
        p = Partida(["max_pecas", "poupar_coringa"], semente=1)
        self.assertEqual(len(p.estado()["oponentes"]), 2)
        self.assertEqual(len(p.estado()["mao"]), 14)


class TestJogar(unittest.TestCase):
    def test_mesa_invalida_e_recusada_com_a_razao_e_nao_altera_o_jogo(self):
        p = Partida(["max_pecas"], semente=1)
        mao_antes = [x["id"] for x in p.estado()["mao"]]
        tres = [x["id"] for x in p.estado()["mao"][:3]]
        r = p.jogar([tres])
        self.assertFalse(r["ok"])
        self.assertTrue(r["erro"])
        self.assertEqual([x["id"] for x in p.estado()["mao"]], mao_antes)
        self.assertEqual(p.estado()["vez"], 0)

    def test_abertura_abaixo_de_trinta_pontos_e_recusada(self):
        # Semente 2: a mão tem um grupo de 4s em três cores, 12 pontos, abaixo do mínimo.
        p = Partida(["max_pecas"], semente=2)
        mapa = ids_por_peca(p)
        cores = [c for c in ("azul", "vermelho", "verde", "amarelo") if (c, 4) in mapa]
        ids = [mapa[(c, 4)][0] for c in cores[:3]]
        r = p.jogar([ids])
        self.assertFalse(r["ok"])
        self.assertIn("30", r["erro"])

    def test_jogada_valida_baixa_as_pecas_e_os_agentes_respondem(self):
        p = Partida(["max_pecas"], semente=1)
        combinacoes = p.sugestao()
        if combinacoes is None:
            self.skipTest("sem jogada legal nesta semente")
        antes = len(p.estado()["mao"])
        r = p.jogar(combinacoes)
        self.assertTrue(r["ok"], r.get("erro"))
        self.assertLess(len(p.estado()["mao"]), antes)
        # Depois de uma jogada válida, os agentes já jogaram e a vez voltou ao humano.
        self.assertTrue(p.estado()["terminou"] or p.estado()["vez"] == 0)

    def test_comprar_passa_a_vez_e_traz_uma_peca(self):
        p = Partida(["max_pecas"], semente=1)
        antes = len(p.estado()["mao"])
        monte_antes = p.estado()["monte"]
        r = p.comprar()
        self.assertTrue(r["ok"])
        self.assertEqual(len(p.estado()["mao"]), antes + 1)
        self.assertLess(p.estado()["monte"], monte_antes)

    def test_lances_dos_agentes_ficam_no_log_para_a_interface_mostrar(self):
        p = Partida(["max_pecas"], semente=1)
        p.comprar()
        self.assertTrue(p.lances)
        ultimo = p.lances[-1]
        self.assertEqual(set(ultimo), {"assento", "nome", "tipo", "pecas", "mao"})

    def test_nao_aceita_jogada_fora_da_vez(self):
        p = Partida(["max_pecas"], semente=1)
        p.jogo.indice_jogador_atual = 1
        r = p.jogar([["0"]])
        self.assertFalse(r["ok"])
        self.assertIn("vez", r["erro"].lower())


class TestSugestao(unittest.TestCase):
    def test_sugestao_e_uma_mesa_que_o_motor_aceita(self):
        p = Partida(["max_pecas"], semente=3)
        combinacoes = p.sugestao()
        if combinacoes is None:
            self.skipTest("sem jogada legal nesta semente")
        self.assertTrue(p.jogar(combinacoes)["ok"])

    def test_sugestao_e_none_quando_so_cabe_comprar(self):
        p = Partida(["max_pecas"], semente=1)
        p.jogo.jogadores[0].mao = []
        self.assertIsNone(p.sugestao())


class TestRegistro(unittest.TestCase):
    def test_partida_encerrada_grava_uma_linha_jsonl_com_o_resultado(self):
        with TemporaryDirectory() as d:
            destino = Path(d) / "humano.jsonl"
            p = Partida(["max_pecas"], semente=1, registro=destino)
            p.jogo.jogadores[0].mao = []  # força a saída do humano no próximo encerramento
            p.encerrar()
            linhas = destino.read_text().splitlines()
            self.assertEqual(len(linhas), 1)
            r = json.loads(linhas[0])
            self.assertEqual(r["oponentes"], ["max_pecas"])
            self.assertEqual(r["semente"], 1)
            self.assertIn("pontos_oficiais", r)
            self.assertIn("vencedor", r)
            self.assertIn("decisoes", r)

    def test_sem_registro_nao_grava_nada(self):
        p = Partida(["max_pecas"], semente=1)
        p.encerrar()  # não deve levantar

    def test_nao_grava_a_mesma_partida_duas_vezes(self):
        with TemporaryDirectory() as d:
            destino = Path(d) / "humano.jsonl"
            p = Partida(["max_pecas"], semente=1, registro=destino)
            p.encerrar()
            p.encerrar()
            self.assertEqual(len(destino.read_text().splitlines()), 1)


class TestOponentes(unittest.TestCase):
    def test_lista_de_oponentes_tem_as_sete_heuristicas_com_nome_legivel(self):
        from src.humano.mesa import OPONENTES
        self.assertEqual(set(OPONENTES), {"aleatorio", "max_pecas", "max_pontos", "minimo",
                                          "poupar_coringa", "so_baixar", "cauteloso"})
        self.assertEqual(OPONENTES["poupar_coringa"]["nome"], "joker-saver")
        self.assertIn("vitorias", OPONENTES["poupar_coringa"])

    def test_oponente_desconhecido_e_recusado(self):
        with self.assertRaises(ValueError):
            Partida(["nao_existe"], semente=1)


if __name__ == "__main__":
    unittest.main()
