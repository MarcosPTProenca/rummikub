import unittest

from src.estrategias.sessao import sessao


class TestSessao(unittest.TestCase):
    def test_joga_o_numero_pedido_de_maos(self):
        r = sessao(["max_pecas", "poupar_coringa"], 240_000, maos=4)
        self.assertEqual(len(r["maos"]), 4)

    def test_pontos_da_sessao_somam_zero(self):
        r = sessao(["max_pecas", "poupar_coringa"], 240_001, maos=4)
        self.assertAlmostEqual(sum(r["pontos"]), 0.0)

    def test_total_e_a_soma_das_maos(self):
        r = sessao(["max_pecas", "poupar_coringa"], 240_002, maos=4)
        for assento in (0, 1):
            self.assertAlmostEqual(r["pontos"][assento], sum(m["pontos"][assento] for m in r["maos"]))

    def test_vencedor_da_sessao_e_quem_tem_mais_pontos(self):
        r = sessao(["max_pecas", "poupar_coringa"], 240_003, maos=4)
        if r["vencedor"] is not None:
            self.assertEqual(r["pontos"][r["vencedor"]], max(r["pontos"]))
            self.assertEqual(r["pontos"].count(max(r["pontos"])), 1)

    def test_assento_inicial_alterna_entre_as_maos(self):
        r = sessao(["max_pecas", "poupar_coringa"], 240_004, maos=4)
        self.assertEqual([m["primeiro"] for m in r["maos"]], [0, 1, 0, 1])

    def test_mesmas_maos_para_estrategias_diferentes(self):
        """O pareamento do plano 12 estendido: a mão k de uma sessão não depende de quem joga."""
        a = sessao(["max_pecas", "max_pecas"], 240_005, maos=3)
        b = sessao(["poupar_coringa", "minimo"], 240_005, maos=3)
        self.assertEqual([m["semente"] for m in a["maos"]], [m["semente"] for m in b["maos"]])

    def test_sessoes_diferentes_jogam_maos_diferentes(self):
        a = sessao(["max_pecas", "max_pecas"], 240_006, maos=3)
        b = sessao(["max_pecas", "max_pecas"], 240_007, maos=3)
        self.assertNotEqual([m["semente"] for m in a["maos"]], [m["semente"] for m in b["maos"]])

    def test_espelho_empata_a_sessao_em_pontos(self):
        """Mesma estratégia nos dois assentos, assento alternado: soma zero e K par cancela a vez."""
        r = sessao(["max_pecas", "max_pecas"], 240_008, maos=4)
        self.assertAlmostEqual(sum(r["pontos"]), 0.0)

    def test_empate_no_total_nao_tem_vencedor(self):
        r = sessao(["max_pecas", "max_pecas"], 240_009, maos=2)
        if r["pontos"][0] == r["pontos"][1]:
            self.assertIsNone(r["vencedor"])

    def test_registra_quantas_maos_cada_assento_ganhou(self):
        r = sessao(["poupar_coringa", "max_pecas"], 240_010, maos=4)
        ganhas = r["maos_ganhas"]
        self.assertEqual(len(ganhas), 2)
        vencedores = [m["vencedor"] for m in r["maos"] if m["vencedor"] is not None]
        self.assertEqual(sum(ganhas), len(vencedores))

    def test_saldo_da_sessao_e_visto_pela_politica_antes_de_cada_mao(self):
        """O espião vê os pontos acumulados; a política vê a diferença para o melhor oponente."""
        vistos = []
        r = sessao(["max_pecas", "max_pecas"], 240_011, maos=3, espiao=lambda saldos: vistos.append(list(saldos)))
        self.assertEqual(vistos[0], [0.0, 0.0])
        self.assertEqual(len(vistos), 3)
        # Antes da mão k, o acumulado é a soma das k-1 anteriores.
        for k in (1, 2):
            for assento in (0, 1):
                self.assertAlmostEqual(vistos[k][assento], sum(m["pontos"][assento] for m in r["maos"][:k]))

    def test_politica_recebe_a_diferenca_para_o_melhor_oponente(self):
        saldos_vistos = []

        def espiao_de_politica(jogo, acoes):
            saldos_vistos.append(jogo.saldo_sessao)
            from src.estrategias.politicas import criar
            return criar("max_pecas", __import__("random").Random(0))(jogo, acoes)

        from src.estrategias.sessao import _jogar_mao
        import src.estrategias.sessao as modulo
        original = modulo.criar
        modulo.criar = lambda nome, rng: espiao_de_politica if nome == "espiao" else original(nome, rng)
        try:
            _jogar_mao(["espiao", "max_pecas"], 240_012, 0, [-20.0, 50.0])
        finally:
            modulo.criar = original
        # O assento 0 está 70 pontos atrás do oponente.
        self.assertTrue(saldos_vistos)
        self.assertAlmostEqual(saldos_vistos[0], -70.0)


class TestSessaoAdaptativa(unittest.TestCase):
    """Atrás na sessão, maximiza a chance de ganhar a mão (poupa coringa);
    à frente ou em dia, protege pontos (solta o coringa cedo)."""

    def setUp(self):
        import random
        from src.estrategias.politicas import criar
        from tests.test_estrategias_pontos import cenario, coringa
        from src.jogo.peca import Peca
        self.criar = criar
        self.random = random
        self.mao = [Peca(id="a5", cor="azul", numero=5), Peca(id="a6", cor="azul", numero=6),
                    coringa(), Peca(id="v9", cor="vermelho", numero=9)]
        self.cenario = cenario

    def escolher(self, saldo):
        from src.jogo.peca import Peca
        jogo = self.cenario(self.mao, monte=[Peca(id=f"m{i}", cor="verde", numero=1) for i in range(5)],
                            maos_oponentes=[Peca(id=f"o{i}", cor="azul", numero=1) for i in range(3)])
        jogo.saldo_sessao = saldo
        politica = self.criar("sessao_adaptativa", self.random.Random(0))
        return politica(jogo, jogo.listar_acoes_validas(True))

    def test_existe_e_e_registrada(self):
        from src.estrategias.politicas import SESSAO
        self.assertIn("sessao_adaptativa", SESSAO)

    def test_muito_atras_poupa_o_coringa(self):
        self.assertEqual(sum(i == "C" for i in self.escolher(-60.0)["ids_pecas"]), 0)

    def test_em_dia_solta_o_coringa_porque_o_oponente_esta_perto(self):
        self.assertEqual(sum(i == "C" for i in self.escolher(0.0)["ids_pecas"]), 1)

    def test_o_limiar_e_um_coringa(self):
        self.assertEqual(sum(i == "C" for i in self.escolher(-31.0)["ids_pecas"]), 0)
        self.assertEqual(sum(i == "C" for i in self.escolher(-29.0)["ids_pecas"]), 1)

    def test_sem_saldo_de_sessao_age_como_na_mao_unica(self):
        """Fora de uma sessão o atributo não existe; a estratégia não pode quebrar."""
        from src.jogo.peca import Peca
        jogo = self.cenario(self.mao, monte=[Peca(id=f"m{i}", cor="verde", numero=1) for i in range(5)],
                            maos_oponentes=[Peca(id=f"o{i}", cor="azul", numero=1) for i in range(3)])
        politica = self.criar("sessao_adaptativa", self.random.Random(0))
        acao = politica(jogo, jogo.listar_acoes_validas(True))
        self.assertEqual(sum(i == "C" for i in acao["ids_pecas"]), 1)


if __name__ == "__main__":
    unittest.main()
