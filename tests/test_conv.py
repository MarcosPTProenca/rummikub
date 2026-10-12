"""Testes da representação estruturada (plano 16).

O vetor de atributos do artigo é escalar agregado: `n/14`, `soma/60`,
`min(maos)/14`. Isso apaga a adjacência entre números e a identidade de cor,
que são exatamente a estrutura de Rummikub. Aqui a entrada é um tensor
cor × número, e estes testes fixam o que ele tem de conter.
"""
import unittest

import numpy as np

from src.estrategias.tensor import CANAIS, CORES, ESCALARES, politica_conv, tensor_da_acao
from src.estrategias.conv import RedeConv
from src.jogo.jogo import Jogo


def um_jogo(semente=1, jogadores=2):
    return Jogo(semente=semente, quantidade_jogadores=jogadores)


class TestTensor(unittest.TestCase):
    def test_forma_e_canais_por_cor_e_numero(self):
        jogo = um_jogo()
        acao = jogo.listar_acoes_validas(True)[0]
        t, escalares = tensor_da_acao(jogo, acao)
        self.assertEqual(t.shape, (CANAIS, 4, 13))
        self.assertEqual(len(escalares), ESCALARES)

    def test_canal_da_mao_conta_as_pecas_que_sobram(self):
        jogo = um_jogo()
        acao = next(a for a in jogo.listar_acoes_validas(True) if a["tipo"] == "comprar")
        t, _ = tensor_da_acao(jogo, acao)
        # Comprar não baixa nada: a mão sobrante é a mão inteira.
        esperado = sum(1 for p in jogo.jogador_atual.mao if p.cor != "coringa")
        self.assertAlmostEqual(t[0].sum() * 2, esperado, places=5)

    def test_cada_peca_cai_na_celula_da_sua_cor_e_numero(self):
        jogo = um_jogo()
        acao = next(a for a in jogo.listar_acoes_validas(True) if a["tipo"] == "comprar")
        t, _ = tensor_da_acao(jogo, acao)
        for p in jogo.jogador_atual.mao:
            if p.cor == "coringa":
                continue
            self.assertGreater(t[0, CORES[p.cor], p.numero - 1], 0,
                               f"{p.cor} {p.numero} não aparece na própria célula")

    def test_canal_da_jogada_tem_so_as_pecas_baixadas(self):
        # Semente 1: a mão abre, então há jogada legal no primeiro turno.
        jogo = um_jogo(semente=1)
        jogada = next((a for a in jogo.listar_acoes_validas(True) if a["tipo"] != "comprar"), None)
        if jogada is None:
            self.skipTest("sem jogada legal nesta semente")
        t, _ = tensor_da_acao(jogo, jogada)
        por_id = {p.id: p for p in jogo.jogador_atual.mao}
        baixadas = [por_id[i] for i in jogada["ids_pecas"] if i in por_id]
        normais = sum(1 for p in baixadas if p.cor != "coringa")
        self.assertAlmostEqual(t[1].sum() * 2, normais, places=5)

    def test_canal_de_nao_vistas_complementa_mao_e_mesa(self):
        jogo = um_jogo()
        acao = next(a for a in jogo.listar_acoes_validas(True) if a["tipo"] == "comprar")
        t, _ = tensor_da_acao(jogo, acao)
        # Duas cópias de cada tipo; o que eu vejo sai da contagem de escondidas.
        for cor, i in CORES.items():
            for n in range(1, 14):
                vistas = sum(1 for p in jogo.jogador_atual.mao if p.cor == cor and p.numero == n)
                vistas += sum(1 for c in jogo.mesa for p in c.pecas if p.cor == cor and p.numero == n)
                self.assertAlmostEqual(t[3, i, n - 1] * 2, 2 - vistas, places=5)

    def test_comprar_e_jogar_produzem_tensores_diferentes(self):
        jogo = um_jogo(semente=1)
        acoes = jogo.listar_acoes_validas(True)
        jogada = next((a for a in acoes if a["tipo"] != "comprar"), None)
        if jogada is None:
            self.skipTest("sem jogada legal nesta semente")
        compra = next(a for a in acoes if a["tipo"] == "comprar")
        t1, _ = tensor_da_acao(jogo, jogada)
        t2, _ = tensor_da_acao(jogo, compra)
        self.assertFalse(np.allclose(t1, t2))

    def test_escalares_comecam_com_os_quinze_atributos_originais(self):
        """A comparação tem de ser adição de estrutura, não troca de informação.

        `atributos` devolve os 15 originais nas primeiras posições em qualquer
        `ENTRADAS`, então basta comparar o prefixo — mexer em
        RUMMIKUB_ENTRADAS aqui vazaria para o resto da suíte, já que o módulo
        lê a variável na importação.
        """
        from src.estrategias.rede import atributos
        jogo = um_jogo()
        acao = jogo.listar_acoes_validas(True)[0]
        _, escalares = tensor_da_acao(jogo, acao)
        originais = atributos(jogo, acao)
        self.assertGreaterEqual(len(originais), 15)
        self.assertEqual(list(escalares[:15]), list(originais[:15]))

    def test_valores_ficam_em_faixa_normalizada(self):
        jogo = um_jogo()
        for acao in jogo.listar_acoes_validas(True)[:5]:
            t, escalares = tensor_da_acao(jogo, acao)
            self.assertTrue(np.all(np.isfinite(t)))
            self.assertLessEqual(float(np.abs(t).max()), 1.5)
            self.assertTrue(all(abs(x) <= 2.0 for x in escalares))


class TestRedeConv(unittest.TestCase):
    def test_pontua_uma_acao_por_entrada(self):
        rede = RedeConv.nova(__import__("random").Random(0))
        jogo = um_jogo()
        acoes = jogo.listar_acoes_validas(True)
        X = [tensor_da_acao(jogo, a) for a in acoes]
        pontos = rede.pontuacoes(X)
        self.assertEqual(len(pontos), len(acoes))
        self.assertTrue(all(np.isfinite(p) for p in pontos))

    def test_theta_tem_o_tamanho_declarado_e_ida_e_volta_preserva(self):
        import random
        rede = RedeConv.nova(random.Random(0))
        self.assertEqual(len(rede.theta), rede.tamanho())
        copia = RedeConv(list(rede.theta))
        jogo = um_jogo()
        X = [tensor_da_acao(jogo, a) for a in jogo.listar_acoes_validas(True)]
        self.assertEqual(rede.pontuacoes(X), copia.pontuacoes(X))

    def test_salvar_e_carregar_mantem_a_politica(self):
        import random
        import tempfile
        from pathlib import Path
        rede = RedeConv.nova(random.Random(1))
        with tempfile.TemporaryDirectory() as d:
            caminho = Path(d) / "r.json"
            rede.salvar(caminho)
            outra = RedeConv.carregar(caminho)
        jogo = um_jogo()
        X = [tensor_da_acao(jogo, a) for a in jogo.listar_acoes_validas(True)]
        for a, b in zip(rede.pontuacoes(X), outra.pontuacoes(X)):
            self.assertAlmostEqual(a, b, places=9)

    def test_gradiente_tem_a_forma_de_theta_e_nao_e_nulo(self):
        import random
        rede = RedeConv.nova(random.Random(0))
        jogo = um_jogo()
        X = [tensor_da_acao(jogo, a) for a in jogo.listar_acoes_validas(True)]
        if len(X) < 2:
            self.skipTest("precisa de pelo menos duas ações")
        g = rede.gradiente(X, 0, 1.0)
        self.assertEqual(len(g), len(rede.theta))
        self.assertGreater(float(np.abs(g).max()), 0.0)

    def test_gradiente_confere_com_diferencas_finitas(self):
        """O gradiente analítico é o coração do treino; erro aqui passa silencioso."""
        import math
        import random
        rede = RedeConv.nova(random.Random(3))
        jogo = um_jogo()
        X = [tensor_da_acao(jogo, a) for a in jogo.listar_acoes_validas(True)]
        if len(X) < 2:
            self.skipTest("precisa de pelo menos duas ações")
        escolhida, vantagem = 0, 1.0
        g = np.asarray(rede.gradiente(X, escolhida, vantagem))

        def log_prob(theta):
            p = RedeConv(list(theta)).probabilidades(X)
            return math.log(p[escolhida])

        eps = 1e-5
        rng = random.Random(0)
        indices = rng.sample(range(len(rede.theta)), 12)
        for i in indices:
            mais = list(rede.theta)
            menos = list(rede.theta)
            mais[i] += eps
            menos[i] -= eps
            numerico = (log_prob(mais) - log_prob(menos)) / (2 * eps)
            self.assertAlmostEqual(g[i], numerico, delta=2e-3,
                                   msg=f"componente {i}: analítico {g[i]:.6f}, numérico {numerico:.6f}")


class TestPolitica(unittest.TestCase):
    def test_politica_escolhe_uma_acao_valida(self):
        import random
        rede = RedeConv.nova(random.Random(0))
        politica = politica_conv(rede, random.Random(0), guloso=True)
        jogo = um_jogo()
        acoes = jogo.listar_acoes_validas(True)
        self.assertIn(politica(jogo, acoes), acoes)

    def test_politica_joga_uma_partida_inteira(self):
        import random

        from src.estrategias.arena import jogar
        rede = RedeConv.nova(random.Random(0))
        politica = politica_conv(rede, random.Random(0), guloso=True)
        from src.estrategias.politicas import criar
        r = jogar([politica, criar("max_pecas", random.Random(0))], semente=7)
        self.assertEqual(r["pecas_total"], 106)
        self.assertIn(r["vencedor"], (0, 1, None))


if __name__ == "__main__":
    unittest.main()
