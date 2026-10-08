import unittest
from copy import deepcopy

from src.agentes.aleatorio import AgenteAleatorio
from src.agentes.q_learning import AgenteQLearning
from src.jogo.peca import Peca
from src.rl.representacao import agrupar_acoes, chave_acao


COMPRAR = {'tipo': 'comprar', 'ids_pecas': []}
OBSERVACAO = ((1,), (1,), 1, 0)
CHAVE = ('comprar', ())


class TestAgentes(unittest.TestCase):
    def test_aleatorio_reproduz_escolhas_e_rejeita_lista_vazia(self):
        acoes = [COMPRAR, {'tipo': 'baixar', 'ids_pecas': ['1']}]
        a = AgenteAleatorio(semente=42)
        b = AgenteAleatorio(semente=42)
        self.assertEqual([a.escolher_acao(acoes) for _ in range(20)],
                         [b.escolher_acao(acoes) for _ in range(20)])
        with self.assertRaises(ValueError):
            a.escolher_acao([])

    def test_parametros_invalidos(self):
        for nome, valores in {'alpha': [0, -1, 2, float('nan')],
                              'gamma': [-1, 2, float('nan')],
                              'epsilon': [-1, 2, float('nan')]}.items():
            for valor in valores:
                with self.subTest(nome=nome, valor=valor):
                    with self.assertRaises(ValueError):
                        AgenteQLearning(**{nome: valor})

    def test_exploracao_e_desempate(self):
        acoes = [CHAVE, ('baixar', (('azul', 1),))]
        agente = AgenteQLearning(epsilon=1, semente=42)
        agente.tabela_q[(OBSERVACAO, CHAVE)] = 10
        escolhas = {agente.escolher_acao(OBSERVACAO, acoes) for _ in range(30)}
        self.assertEqual(escolhas, set(acoes))
        agente.tabela_q.clear()
        escolhas = {agente.escolher_acao(OBSERVACAO, acoes, explorar=False) for _ in range(30)}
        self.assertEqual(escolhas, set(acoes))
        self.assertEqual(agente.tabela_q, {})

    def test_transicao_ativa_exige_acoes_futuras(self):
        agente = AgenteQLearning()
        with self.assertRaises(ValueError):
            agente.atualizar(OBSERVACAO, CHAVE, 1, OBSERVACAO, [], False)
        self.assertEqual(agente.tabela_q, {})

    def test_validacao_das_acoes(self):
        mao = [Peca(id='1', cor='azul', numero=3)]
        acoes_invalidas = [None, {}, {'tipo': 'desconhecido', 'ids_pecas': []},
                           {'tipo': 'comprar', 'ids_pecas': ['1']},
                           {'tipo': 'baixar', 'ids_pecas': []},
                           {'tipo': 'baixar', 'ids_pecas': ['ausente']},
                           {'tipo': 'baixar', 'ids_pecas': '1'},
                           {'tipo': 'baixar', 'ids_pecas': [1]},
                           {'tipo': 'baixar', 'ids_pecas': [['1']]}]
        for acao in acoes_invalidas:
            with self.subTest(acao=acao):
                with self.assertRaises(ValueError):
                    chave_acao(acao, mao)

    def test_representacao_preserva_multiplicidade_e_entradas(self):
        mao = [Peca(id='1', cor='azul', numero=3), Peca(id='2', cor='azul', numero=3)]
        acao = {'tipo': 'baixar', 'ids_pecas': ['2', '1']}
        anterior = deepcopy((mao, acao))
        self.assertEqual(chave_acao(acao, mao), ('baixar', (('azul', 3), ('azul', 3))))
        self.assertEqual((mao, acao), anterior)
        self.assertEqual(agrupar_acoes([], mao), {})


if __name__ == '__main__':
    unittest.main()
