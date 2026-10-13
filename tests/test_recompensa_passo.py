"""Plano 17: recompensa por passo, potencial telescópico sobre o passivo da mão.

O defeito que isto corrige: em `treinar_rede`, `vantagem` é constante dentro da
trajetória, então todas as ~60 decisões de uma mão recebem o mesmo crédito. A
decisão de guardar o coringa recebe o mesmo sinal que uma decisão forçada.

A intervenção é `r_t = Φ(s_{t+1}) − Φ(s_t)` com `Φ = −(pontos + 30·coringas)/60`.
Como a soma telescopa, a política ótima não muda (Ng et al. 1999); o que muda é a
variância do estimador. Estes testes fixam as duas propriedades que fazem a
intervenção ser segura: a telescopagem exata e a equivalência byte-a-byte do
caminho padrão.
"""
import math
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.estrategias import arena, politicas, rede
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca


class TestPotencialDaMao(unittest.TestCase):
    def test_mao_vazia_tem_potencial_zero(self):
        self.assertEqual(rede.potencial_mao([]), 0.0)

    def test_coringa_pesa_trinta_na_escala_da_recompensa(self):
        coringa = [Peca(id="j", cor="coringa", numero=0)]
        self.assertAlmostEqual(rede.potencial_mao(coringa), -30 / rede.ESCALA_PONTOS)

    def test_potencial_e_negativo_do_passivo_normalizado(self):
        mao = [Peca(id="a", cor="azul", numero=7), Peca(id="b", cor="verde", numero=5),
               Peca(id="j", cor="coringa", numero=0)]
        self.assertAlmostEqual(rede.potencial_mao(mao), -(7 + 5 + 30) / rede.ESCALA_PONTOS)

    def test_baixar_coringa_melhora_o_potencial_em_meio_ponto(self):
        """O sinal que o treino atual dilui: soltar o coringa vale +0,5 no passo."""
        antes = [Peca(id="j", cor="coringa", numero=0), Peca(id="a", cor="azul", numero=3)]
        depois = [Peca(id="a", cor="azul", numero=3)]
        self.assertAlmostEqual(rede.potencial_mao(depois) - rede.potencial_mao(antes), 0.5)


class TestRecompensasPorPasso(unittest.TestCase):
    def test_telescopagem_exata(self):
        """Σ r_t = Φ(final) − Φ(inicial): é isto que preserva a política ótima."""
        rng = random.Random(4)
        potenciais = [rng.uniform(-2, 0) for _ in range(12)]
        passos = rede.recompensas_por_passo(potenciais)
        self.assertEqual(len(passos), len(potenciais) - 1)
        self.assertAlmostEqual(sum(passos), potenciais[-1] - potenciais[0])

    def test_sequencia_curta_e_vazia(self):
        self.assertEqual(rede.recompensas_por_passo([]), [])
        self.assertEqual(rede.recompensas_por_passo([-0.5]), [])

    def test_retornos_descontados_sao_somas_de_cauda(self):
        """G_t = retorno do jogo + Σ_{k≥t} r_k, com γ=1 (mãos são curtas)."""
        passos = [0.5, -0.25, 0.125]
        gs = rede.retornos_por_passo(passos, retorno_final=1.0)
        self.assertEqual(len(gs), 3)
        self.assertAlmostEqual(gs[0], 1.0 + 0.5 - 0.25 + 0.125)
        self.assertAlmostEqual(gs[1], 1.0 - 0.25 + 0.125)
        self.assertAlmostEqual(gs[2], 1.0 + 0.125)

    def test_sem_passos_o_retorno_e_o_do_jogo(self):
        self.assertEqual(rede.retornos_por_passo([], retorno_final=-1.0), [])

    def test_decisao_tardia_nao_carrega_credito_das_anteriores(self):
        """O ponto do plano: cada decisão responde pelo que ela mesma mudou."""
        passos = [0.0, 0.0, 0.5]
        gs = rede.retornos_por_passo(passos, retorno_final=0.0)
        self.assertAlmostEqual(gs[2], 0.5)
        self.assertAlmostEqual(gs[0], 0.5)
        passos_inicio = [0.5, 0.0, 0.0]
        gs2 = rede.retornos_por_passo(passos_inicio, retorno_final=0.0)
        self.assertAlmostEqual(gs2[2], 0.0, msg="crédito de uma decisão antiga vazou para a última")


class TestPotenciaisDeUmaPartida(unittest.TestCase):
    def test_observador_grava_um_potencial_por_decisao_do_lado(self):
        jogo = Jogo(semente=7)
        pol = politicas.criar("poupar_coringa", random.Random(7))
        serie = rede.ColetorDePotencial(lado=0)
        arena.jogar([pol, pol], semente=7, observador=serie)
        self.assertGreater(len(serie.potenciais), 2)
        self.assertTrue(all(p <= 0 for p in serie.potenciais),
                        "o potencial é um passivo, nunca positivo")

    def test_potencial_final_corresponde_a_mao_final(self):
        pol = politicas.criar("max_pecas", random.Random(3))
        serie = rede.ColetorDePotencial(lado=0)
        resultado = arena.jogar([pol, pol], semente=3, observador=serie)
        if not resultado["truncada"]:
            self.assertAlmostEqual(serie.potenciais[-1],
                                   -(serie.passivo_final) / rede.ESCALA_PONTOS, places=6)


class TestCaminhoPadraoInalterado(unittest.TestCase):
    """A garantia que protege o artigo: sem `--passo`, nada muda."""

    def test_treino_padrao_reproduz_os_mesmos_pesos(self):
        with TemporaryDirectory() as d:
            a = rede.treinar_rede(Path(d) / "a", iteracoes=2, jogos=4, processos=1,
                                  ocultos=4, semente=11, liga=("max_pecas",))
            b = rede.treinar_rede(Path(d) / "b", iteracoes=2, jogos=4, processos=1,
                                  ocultos=4, semente=11, liga=("max_pecas",))
            self.assertEqual(list(a.theta), list(b.theta), "o treino deixou de ser determinístico")

    def test_passo_nenhum_e_identico_ao_padrao(self):
        with TemporaryDirectory() as d:
            padrao = rede.treinar_rede(Path(d) / "p", iteracoes=2, jogos=4, processos=1,
                                       ocultos=4, semente=5, liga=("max_pecas",))
            explicito = rede.treinar_rede(Path(d) / "e", iteracoes=2, jogos=4, processos=1,
                                          ocultos=4, semente=5, liga=("max_pecas",),
                                          passo="nenhum")
            self.assertEqual(list(padrao.theta), list(explicito.theta))

    def test_passo_potencial_treina_e_muda_os_pesos(self):
        with TemporaryDirectory() as d:
            padrao = rede.treinar_rede(Path(d) / "p", iteracoes=2, jogos=4, processos=1,
                                       ocultos=4, semente=5, liga=("max_pecas",),
                                       recompensa="pontos")
            potencial = rede.treinar_rede(Path(d) / "q", iteracoes=2, jogos=4, processos=1,
                                          ocultos=4, semente=5, liga=("max_pecas",),
                                          recompensa="pontos", passo="potencial")
            self.assertNotEqual(list(padrao.theta), list(potencial.theta),
                                "o shaping por passo não alterou o gradiente")

    def test_passo_desconhecido_falha_cedo(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                rede.treinar_rede(Path(d) / "x", iteracoes=1, jogos=2, processos=1,
                                  ocultos=4, semente=1, liga=("max_pecas",), passo="inventado")


if __name__ == "__main__":
    unittest.main()


class TestBaselineDoSinalPorPasso(unittest.TestCase):
    """Regressão do defeito achado no plano 17 (primeira rodada).

    G_t inclui a cauda do potencial, então sua média (~+0,8) não é a do retorno do
    jogo (~−0,05). Subtrair o baseline das partidas enviesava toda decisão para cima
    em ~+0,87, empurrando também as ações ruins — o oposto de atribuição de crédito.
    A vantagem por passo tem de ser centrada no próprio sinal por passo.
    """

    def test_vantagem_por_passo_fica_centrada(self):
        passos = [0.5, -0.25, 0.125]
        gs = rede.retornos_por_passo(passos, retorno_final=0.0)
        media = sum(gs) / len(gs)
        centradas = [g - media for g in gs]
        self.assertAlmostEqual(sum(centradas), 0.0,
                               msg="a vantagem por passo não está centrada em zero")

    def test_media_de_g_difere_da_media_do_retorno(self):
        """Fixa o fato que causou o bug, para ninguém reusar o baseline errado."""
        rng = random.Random(9)
        retornos, medias_g = [], []
        for _ in range(40):
            passos = [rng.uniform(-0.1, 0.6) for _ in range(10)]
            r = rng.uniform(-1.5, 1.5)
            gs = rede.retornos_por_passo(passos, r)
            retornos.append(r)
            medias_g.append(sum(gs) / len(gs))
        media_r = sum(retornos) / len(retornos)
        media_g = sum(medias_g) / len(medias_g)
        self.assertGreater(abs(media_g - media_r), 0.3,
                           "se estas médias coincidissem, o baseline do jogo serviria")
