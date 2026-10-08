import unittest
from types import SimpleNamespace

from src.agentes.q_learning import AgenteQLearning
from src.rl.representacao import (
    observar,
    chave_observacao,
    chave_acao,
    agrupar_acoes,
)


def criar_peca(id: str, cor: str, numero: int):
    # Dublê para testar representação sem depender do Pydantic.
    return SimpleNamespace(id=id, cor=cor, numero=numero)


OBSERVACAO = ((1, 0), (2,), 3, 0, (False, False), ())
PROXIMA_OBSERVACAO = ((0, 0), (1,), 2, 0, (False, False), ())

COMPRAR = ("comprar", ())
BAIXAR = ("baixar", (("azul", 1),))


class TestRepresentacao(unittest.TestCase):

    def test_observacao_conta_copias(self):
        mao = [
            criar_peca("1", "azul", 1),
            criar_peca("2", "azul", 1),
            criar_peca("3", "vermelho", 3),
        ]

        jogo = SimpleNamespace(
            jogadores=[
                SimpleNamespace(id="0", mao=mao, abriu=False),
                SimpleNamespace(id="1", mao=[], abriu=False),
            ],
            monte=[],
            mesa=[],
            passagens_consecutivas=0,
        )

        resultado = observar(jogo, 0, numero_maximo=3)

        self.assertEqual(
            resultado["mao"],
            (2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        )
        self.assertEqual(resultado["pecas_adversarios"], (0,))
        self.assertEqual(resultado["pecas_monte"], 0)

    def test_chave_observacao(self):
        observacao = {
            "mao": (1, 0),
            "pecas_adversarios": (2,),
            "pecas_monte": 3,
            "passagens_consecutivas": 0,
            "aberturas": (False, False),
            "mesa": (),
        }

        self.assertEqual(
            chave_observacao(observacao),
            OBSERVACAO,
        )

    def test_chave_compra(self):
        acao = {"tipo": "comprar", "ids_pecas": []}

        self.assertEqual(chave_acao(acao, []), COMPRAR)

    def test_copias_equivalentes_produzem_mesma_chave(self):
        mao = [
            criar_peca("1", "azul", 3),
            criar_peca("2", "azul", 3),
            criar_peca("3", "azul", 4),
        ]

        acao_a = {"tipo": "baixar", "ids_pecas": ["1", "3"]}
        acao_b = {"tipo": "baixar", "ids_pecas": ["3", "2"]}

        self.assertEqual(
            chave_acao(acao_a, mao),
            chave_acao(acao_b, mao),
        )

    def test_ids_repetidos_sao_rejeitados(self):
        mao = [criar_peca("1", "azul", 3)]
        acao = {"tipo": "baixar", "ids_pecas": ["1", "1"]}

        with self.assertRaises(ValueError):
            chave_acao(acao, mao)

    def test_agrupar_mantem_primeira_acao(self):
        mao = [
            criar_peca("1", "azul", 3),
            criar_peca("2", "azul", 3),
        ]

        acoes = [
            {"tipo": "baixar", "ids_pecas": ["1"]},
            {"tipo": "baixar", "ids_pecas": ["2"]},
        ]

        resultado = agrupar_acoes(acoes, mao)

        self.assertEqual(len(resultado), 1)
        self.assertEqual(
            resultado[("baixar", (("azul", 3),))],
            acoes[0],
        )


class TestQLearning(unittest.TestCase):

    def test_valor_desconhecido_e_zero(self):
        agente = AgenteQLearning()

        self.assertEqual(
            agente.valor(OBSERVACAO, COMPRAR),
            0.0,
        )
        self.assertEqual(agente.tabela_q, {})

    def test_atualizacao_terminal(self):
        agente = AgenteQLearning(alpha=0.1)
        agente.tabela_q[(OBSERVACAO, COMPRAR)] = 0.2

        agente.atualizar(
            observacao=OBSERVACAO,
            acao=COMPRAR,
            recompensa=1.0,
            proxima_observacao=PROXIMA_OBSERVACAO,
            proximas_acoes=[],
            terminado=True,
        )

        # 0.2 + 0.1 * (1.0 - 0.2) = 0.28
        self.assertAlmostEqual(
            agente.valor(OBSERVACAO, COMPRAR),
            0.28,
        )

    def test_atualizacao_com_valor_futuro(self):
        agente = AgenteQLearning(alpha=0.1, gamma=0.9)

        agente.tabela_q[(OBSERVACAO, COMPRAR)] = 0.2
        agente.tabela_q[(PROXIMA_OBSERVACAO, BAIXAR)] = 0.5

        agente.atualizar(
            observacao=OBSERVACAO,
            acao=COMPRAR,
            recompensa=0.0,
            proxima_observacao=PROXIMA_OBSERVACAO,
            proximas_acoes=[BAIXAR],
            terminado=False,
        )

        # 0.2 + 0.1 * (0 + 0.9 * 0.5 - 0.2) = 0.225
        self.assertAlmostEqual(
            agente.valor(OBSERVACAO, COMPRAR),
            0.225,
        )

    def test_valor_de_acao_ilegal_e_ignorado(self):
        agente = AgenteQLearning(alpha=1.0, gamma=1.0)

        agente.tabela_q[(PROXIMA_OBSERVACAO, BAIXAR)] = 100.0

        agente.atualizar(
            observacao=OBSERVACAO,
            acao=COMPRAR,
            recompensa=0.0,
            proxima_observacao=PROXIMA_OBSERVACAO,
            proximas_acoes=[COMPRAR],
            terminado=False,
        )

        self.assertEqual(
            agente.valor(OBSERVACAO, COMPRAR),
            0.0,
        )

    def test_avaliacao_ignora_epsilon(self):
        agente = AgenteQLearning(epsilon=1.0)

        agente.tabela_q[(OBSERVACAO, COMPRAR)] = -1.0
        agente.tabela_q[(OBSERVACAO, BAIXAR)] = 1.0

        tabela_antes = agente.tabela_q.copy()

        acao = agente.escolher_acao(
            OBSERVACAO,
            [COMPRAR, BAIXAR],
            explorar=False,
        )

        self.assertEqual(acao, BAIXAR)
        self.assertEqual(agente.tabela_q, tabela_antes)
        self.assertEqual(agente.epsilon, 1.0)

    def test_escolha_sem_acoes_gera_erro(self):
        agente = AgenteQLearning()

        with self.assertRaises(ValueError):
            agente.escolher_acao(OBSERVACAO, [])


if __name__ == "__main__":
    unittest.main()