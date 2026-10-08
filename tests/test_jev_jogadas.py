import random
import unittest

from src.estrategias.jev import Orcamento
from src.estrategias.jev_jogadas import JevJogadas
from tests.test_estrategias import grupo, partida, pecas


def mesa_com_encaixe():
    mesa = pecas([('vermelho', n) for n in (4, 5, 6)], 't')
    mao = pecas([('vermelho', 7), ('azul', 1), ('coringa', 0)], 'm') + grupo(9, 'g')
    return partida(mao, [mesa], abriu=True)


class TestJevJogadas(unittest.TestCase):
    def setUp(self):
        self.jogo = mesa_com_encaixe()
        self.acoes = self.jogo.listar_acoes_validas()

    def agente(self, escolher, orcamento=None, **opcoes):
        chamadas = []

        def transporte(corpo):
            chamadas.append(corpo)
            chave, confianca, tokens = escolher(corpo)
            return {"answers": {"jogada": {"type": "choice", "choice": chave, "confidence": confianca,
                                           "probabilities": {chave: 1.0}}},
                    "usage": {"input_tokens": tokens, "output_tokens": 3}}

        return JevJogadas(transporte, random.Random(0), orcamento or Orcamento(1.0), **opcoes), chamadas

    def criterios(self, corpo):
        return corpo['questions']['jogada']['criteria']

    def test_envia_todas_as_jogadas_legais_inclusive_comprar(self):
        agente, chamadas = self.agente(lambda c: (next(iter(self.criterios(c))), 0.9, 100))
        agente(self.jogo, self.acoes)
        criterios = self.criterios(chamadas[0])
        self.assertEqual(len(criterios), len(self.acoes))
        self.assertGreater(len(self.acoes), 3)
        self.assertEqual(sum('Draw a tile' in d for d in criterios.values()), 1)

    def test_escolha_vira_a_acao_legal_correspondente(self):
        def escolher(corpo):
            chave = next(k for k, d in self.criterios(corpo).items() if 'Draw a tile' in d)
            return chave, 0.9, 100
        agente, _ = self.agente(escolher)
        escolhida = agente(self.jogo, self.acoes)
        self.assertEqual(escolhida['tipo'], 'comprar')
        self.assertTrue(any(escolhida is a for a in self.acoes))
        self.assertEqual(agente.contagem['chamadas'], 1)

    def test_estado_traz_contexto_e_calculos_em_codigo(self):
        agente, chamadas = self.agente(lambda c: (next(iter(self.criterios(c))), 0.9, 100))
        agente(self.jogo, self.acoes)
        corpo = chamadas[0]
        estado = corpo['state']
        self.assertEqual(estado['my_hand'], sorted(estado['my_hand']))
        self.assertIn('red 7', estado['my_hand'])
        self.assertEqual(estado['table_sets'], [['red 4', 'red 5', 'red 6']])
        self.assertEqual(estado['hand_points_if_game_ends'], 7 + 1 + 30 + 27)
        self.assertEqual(estado['opponent_tiles_in_hand'], 14)
        self.assertIn('rules', estado)
        todas = ' '.join(self.criterios(corpo).values())
        self.assertIn('tiles from hand: red 7', todas)
        self.assertIn('hand tiles left', todas)

    def test_jogada_que_mexe_na_mesa_mostra_o_que_sai_e_o_que_entra(self):
        agente, chamadas = self.agente(lambda c: (next(iter(self.criterios(c))), 0.9, 100))
        agente(self.jogo, self.acoes)
        encaixe = [d for d in self.criterios(chamadas[0]).values() if 'tiles from hand: red 7' in d and 'table sets removed' in d]
        self.assertTrue(encaixe)
        self.assertIn('table sets removed: red 4-5-6', encaixe[0])
        self.assertIn('table sets added: red 4-5-6-7', encaixe[0])

    def test_ordem_das_opcoes_e_embaralhada_mas_reprodutivel(self):
        def ordem(semente):
            agente, chamadas = self.agente(lambda c: (next(iter(self.criterios(c))), 0.9, 1))
            agente.rng = random.Random(semente)
            agente(self.jogo, self.acoes)
            return list(self.criterios(chamadas[0]).values())
        self.assertEqual(ordem(1), ordem(1))
        self.assertNotEqual(ordem(1), ordem(2))
        self.assertEqual(sorted(ordem(1)), sorted(ordem(2)))

    def test_resposta_invalida_ou_pouco_confiante_cai_no_fallback(self):
        for resposta in [('inexistente', 0.9, 10), (None, 0.05, 10)]:
            agente, _ = self.agente(lambda c, r=resposta: (r[0] or next(iter(self.criterios(c))), r[1], r[2]),
                                    confianca_minima=0.3)
            acao = agente(self.jogo, self.acoes)
            self.assertTrue(any(acao is a for a in self.acoes))
            self.assertEqual(agente.contagem['fallbacks'], 1)

    def test_sem_escolha_real_ou_sem_orcamento_nao_chama_a_api(self):
        agente, chamadas = self.agente(lambda c: ('x', 0.9, 1), Orcamento(0.0))
        self.assertTrue(any(agente(self.jogo, self.acoes) is a for a in self.acoes))
        unica = [self.acoes[-1]]
        agente2, chamadas2 = self.agente(lambda c: ('x', 0.9, 1))
        self.assertIs(agente2(self.jogo, unica), unica[0])
        self.assertEqual(chamadas + chamadas2, [])

    def test_contabiliza_custo_e_telemetria_da_escolha(self):
        agente, _ = self.agente(lambda c: (next(iter(self.criterios(c))), 0.8, 1_000_000))
        agente(self.jogo, self.acoes)
        self.assertAlmostEqual(agente.orcamento.gasto, 0.042)
        self.assertEqual(agente.contagem['tokens'], 1_000_000)
        self.assertEqual(agente.confiancas, [0.8])
        self.assertEqual(sum(agente.concordancia.values()) >= 0, True)


if __name__ == '__main__':
    unittest.main()


class TestLigacaoNoExperimento(unittest.TestCase):
    def test_construir_jev_jogadas_usa_o_orcamento_compartilhado(self):
        from src import experimentos
        experimentos._chave, experimentos._orcamento = 'chave-falsa', Orcamento(0.5)
        try:
            agente = experimentos.construir('jev_jogadas', 7, 1)
        finally:
            experimentos._chave = experimentos._orcamento = None
        self.assertIsInstance(agente, JevJogadas)
        self.assertEqual(agente.orcamento.limite, 0.5)

    def test_registro_da_partida_traz_telemetria_do_jev(self):
        from src import experimentos
        class Falso(JevJogadas):
            def __init__(self):
                super().__init__(None, random.Random(0), Orcamento(0.0))
        original = experimentos.construir
        experimentos.construir = lambda nome, semente, assento, agente=None: (
            Falso() if nome == 'jev_jogadas' else original(nome, semente, assento, agente))
        try:
            registro = experimentos.partida('jev_jogadas', 'poupar_coringa', 4001, 0)
        finally:
            experimentos.construir = original
        self.assertEqual(registro['jev']['fallbacks'] > 0, True)
        self.assertIn('concordancia', registro['jev'])
        self.assertIn('confiancas', registro['jev'])
