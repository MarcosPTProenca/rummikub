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
        self.assertEqual(set(ultimo), {"n", "hora", "assento", "nome", "tipo", "pecas", "mao"})

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


class TestReplayDosAgentes(unittest.TestCase):
    """O humano precisa ver o agente jogando, não a mesa mudando de uma vez."""

    def test_cada_lance_do_agente_vira_um_passo_com_a_mesa_daquele_momento(self):
        p = Partida(["max_pecas"], semente=1)
        p.comprar()
        self.assertTrue(p.replay)
        passo = p.replay[-1]
        self.assertEqual(set(passo), {"assento", "nome", "tipo", "ids", "mesa", "mao"})
        self.assertNotEqual(passo["assento"], p.humano)

    def test_passo_de_jogada_traz_os_ids_que_o_agente_baixou(self):
        p = Partida(["max_pecas"], semente=1)
        for _ in range(40):
            if p.jogo.terminou:
                break
            p.comprar()
            jogadas = [x for x in p.replay if x["tipo"] == "jogar"]
            if jogadas:
                passo = jogadas[0]
                self.assertTrue(passo["ids"])
                # Os ids baixados têm de aparecer na mesa daquele passo.
                na_mesa = {q["id"] for c in passo["mesa"] for q in c}
                self.assertTrue(set(passo["ids"]) <= na_mesa)
                return
        self.skipTest("o agente não baixou nada nesta semente")

    def test_compra_nao_revela_a_peca_comprada(self):
        p = Partida(["max_pecas"], semente=1)
        p.comprar()
        for passo in p.replay:
            if passo["tipo"] == "comprar":
                self.assertEqual(passo["ids"], [])
                return
        self.skipTest("o agente não comprou nesta semente")

    def test_replay_cobre_so_os_lances_desde_a_ultima_acao_do_humano(self):
        p = Partida(["max_pecas"], semente=1)
        p.comprar()
        primeiro = list(p.replay)
        p.comprar()
        self.assertTrue(p.replay)
        self.assertNotEqual(p.replay, primeiro + p.replay)
        self.assertTrue(all(x["assento"] != p.humano for x in p.replay))

    def test_estado_expoe_o_replay_para_a_interface(self):
        p = Partida(["max_pecas"], semente=1)
        p.comprar()
        self.assertEqual(p.estado()["replay"], p.replay)

    def test_mesa_do_ultimo_passo_e_a_mesa_atual(self):
        p = Partida(["max_pecas"], semente=1)
        for _ in range(12):
            if p.jogo.terminou:
                break
            p.comprar()
            if p.replay:
                atual = [[q["id"] for q in c] for c in p.estado()["mesa"]]
                ultimo = [[q["id"] for q in c] for c in p.replay[-1]["mesa"]]
                self.assertEqual(ultimo, atual)


class TestHistorico(unittest.TestCase):
    """O log precisa dizer em que ordem e a que hora cada lance aconteceu."""

    def test_cada_lance_tem_numero_sequencial_e_hora(self):
        p = Partida(["max_pecas"], semente=1)
        p.comprar()
        p.comprar()
        for lance in p.lances:
            self.assertIn("n", lance)
            self.assertIn("hora", lance)
            self.assertRegex(lance["hora"], r"^\d{2}:\d{2}:\d{2}$")
        self.assertEqual([x["n"] for x in p.lances], list(range(1, len(p.lances) + 1)))

    def test_estado_devolve_os_lances_mais_recentes(self):
        p = Partida(["max_pecas"], semente=1)
        for _ in range(20):
            if p.jogo.terminou:
                break
            p.comprar()
        self.assertGreater(len(p.lances), 12)
        do_estado = p.estado()["lances"]
        self.assertEqual(do_estado[-1], p.lances[-1])
        self.assertEqual([x["n"] for x in do_estado], sorted(x["n"] for x in do_estado))

    def test_partida_marca_o_instante_de_inicio(self):
        p = Partida(["max_pecas"], semente=1)
        self.assertIsInstance(p.estado()["segundos"], float)
        self.assertGreaterEqual(p.estado()["segundos"], 0.0)

    def test_cronometro_para_quando_a_partida_termina(self):
        import time
        p = Partida(["max_pecas"], semente=1)
        p.jogo.terminou = True
        p.jogo.vencedor = p.jogo.jogadores[0]
        p.encerrar()
        primeiro = p.estado()["segundos"]
        time.sleep(0.05)
        self.assertEqual(p.estado()["segundos"], primeiro)


class TestAgenteAprendido(unittest.TestCase):
    """A rede treinada do plano 15 tem de estar jogável na mesa."""

    def test_rede_aparece_na_lista_de_oponentes(self):
        from src.humano.mesa import OPONENTES
        self.assertIn("rede_pontos", OPONENTES)
        self.assertEqual(OPONENTES["rede_pontos"]["nome"], "points-net")

    def test_partida_contra_a_rede_joga_e_termina(self):
        p = Partida(["rede_pontos"], semente=1)
        for _ in range(300):
            if p.jogo.terminou:
                break
            combinacoes = p.sugestao()
            r = p.jogar(combinacoes) if combinacoes else p.comprar()
            self.assertTrue(r["ok"], r.get("erro"))
        self.assertTrue(p.jogo.terminou)
        self.assertAlmostEqual(sum(p.estado()["pontos"]), 0.0)

    def test_rede_da_mesa_decide_igual_a_rede_avaliada(self):
        """A mesa trunca o vetor de atributos; se divergir, você joga contra outra política."""
        import os
        import random
        from pathlib import Path

        from src.humano.mesa import REDE_PONTOS, _politica, _rede_disponivel
        if not _rede_disponivel():
            self.skipTest("checkpoint do plano 15 não está no disco")
        if os.environ.get("RUMMIKUB_ENTRADAS") != "15":
            self.skipTest("comparação exige RUMMIKUB_ENTRADAS=15 na importação")
        from src.estrategias.arena import executar
        from src.estrategias.politicas import criar
        from src.estrategias.rede import PoliticaRede, Rede
        from src.jogo.jogo import Jogo

        referencia = PoliticaRede.da_rede(Rede.carregar(Path(REDE_PONTOS)),
                                          random.Random(0), guloso=True)
        da_mesa = _politica("rede_pontos", random.Random(0))
        divergencias = total = 0
        for s in range(8):
            jogo = Jogo(semente=s, quantidade_jogadores=2)
            saver = criar("poupar_coringa", random.Random(s))
            for _ in range(60):
                if jogo.terminou:
                    break
                acoes = jogo.listar_acoes_validas(True)
                if jogo.indice_jogador_atual == 0:
                    escolha = referencia(jogo, acoes)
                    total += 1
                    divergencias += escolha is not da_mesa(jogo, acoes)
                    executar(jogo, escolha)
                else:
                    executar(jogo, saver(jogo, acoes))
        self.assertGreater(total, 100)
        self.assertEqual(divergencias, 0, f"{divergencias} de {total} decisões divergem")

    def test_rede_e_opcional_e_nao_quebra_se_o_arquivo_faltar(self):
        """Quem clona o repo sem os checkpoints ainda deve poder jogar."""
        from src.humano.mesa import OPONENTES, _rede_disponivel
        if not _rede_disponivel():
            self.assertNotIn("rede_pontos", OPONENTES)
        else:
            self.assertIn("rede_pontos", OPONENTES)


class TestOponentes(unittest.TestCase):
    def test_lista_de_oponentes_tem_as_sete_heuristicas_com_nome_legivel(self):
        from src.humano.mesa import OPONENTES
        heuristicas = {"aleatorio", "max_pecas", "max_pontos", "minimo",
                       "poupar_coringa", "so_baixar", "cauteloso"}
        self.assertTrue(heuristicas <= set(OPONENTES))
        self.assertEqual(OPONENTES["poupar_coringa"]["nome"], "joker-saver")
        self.assertIn("vitorias", OPONENTES["poupar_coringa"])

    def test_saver_then_dump_esta_no_menu_com_a_taxa_do_torneio_original(self):
        """Plano 21: é a melhor estratégia por partidas ganhas com 2 e 3 jogadores.

        A taxa 0,732 é a média contra as mesmas 6 estratégias e com o mesmo protocolo que deu
        0,726 ao joker-saver (resultados/resumo.json); o cálculo reproduz o 0,7255 do joker-saver
        antes de ser usado, para a escala ser a mesma."""
        from src.humano.mesa import OPONENTES
        self.assertIn("poupar_descarta", OPONENTES)
        self.assertEqual(OPONENTES["poupar_descarta"]["nome"], "saver-then-dump")
        self.assertEqual(OPONENTES["poupar_descarta"]["vitorias"], 0.732)
        self.assertGreater(OPONENTES["poupar_descarta"]["vitorias"], OPONENTES["poupar_coringa"]["vitorias"])
        self.assertTrue(OPONENTES["poupar_descarta"]["sobre"])

    def test_partida_contra_o_saver_then_dump_joga_e_termina(self):
        p = Partida(["poupar_descarta"], semente=2)
        for _ in range(300):
            if p.jogo.terminou:
                break
            combinacoes = p.sugestao()
            r = p.jogar(combinacoes) if combinacoes else p.comprar()
            self.assertTrue(r["ok"], r.get("erro"))
        self.assertTrue(p.jogo.terminou)
        self.assertAlmostEqual(sum(p.estado()["pontos"]), 0.0)

    def test_o_oponente_do_menu_e_a_politica_da_arena(self):
        """A mesa não reimplementa a estratégia: usa a mesma que os torneios mediram."""
        import random
        from src.estrategias.arena import executar
        from src.estrategias.politicas import criar
        from src.humano.mesa import _politica
        from src.jogo.jogo import Jogo
        for semente in range(3):
            a, b = Jogo(semente=semente), Jogo(semente=semente)
            pa = _politica("poupar_descarta", random.Random(semente))
            pb = criar("poupar_descarta", random.Random(semente))
            for _ in range(60):
                if a.terminou:
                    break
                # Uma chamada por política por turno: cada uma consome o próprio gerador de desempates.
                acao_a, acao_b = pa(a, a.listar_acoes_validas()), pb(b, b.listar_acoes_validas())
                self.assertEqual(acao_a, acao_b)
                executar(a, acao_a)
                executar(b, acao_b)

    def test_oponente_desconhecido_e_recusado(self):
        with self.assertRaises(ValueError):
            Partida(["nao_existe"], semente=1)


if __name__ == "__main__":
    unittest.main()
