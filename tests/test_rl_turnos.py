import unittest
from copy import deepcopy

from src.agentes.aleatorio import AgenteAleatorio
from src.agentes.q_learning import AgenteQLearning
from src.rl.ambiente import AmbienteRL
from src.rl.avaliacao import avaliar
from src.rl.representacao import agrupar_acoes, chave_acao, chave_observacao, observar
from src.rl.treinamento import treinar
from test_turnos import estado, grupo, ids, partida, pecas


class TestRLTurnos(unittest.TestCase):
    def test_observacao_distingue_abertura_e_mesa_sem_ids_fisicos(self):
        jogo = partida(grupo(10), [grupo(7, 't')])
        obs = observar(jogo, 0)
        self.assertEqual(obs['aberturas'], (False, False))
        self.assertEqual(obs['mesa'], ((('azul', 7), ('verde', 7), ('vermelho', 7)),))
        chave = chave_observacao(obs)
        jogo.jogador_atual.abriu = True
        self.assertNotEqual(chave, chave_observacao(observar(jogo, 0)))
        jogo.mesa = []
        self.assertNotEqual(chave_observacao(obs), chave_observacao(observar(jogo, 0)))

    def test_chave_de_mesa_ignora_ordem_e_distingue_particoes(self):
        mesa = pecas([('azul', n) for n in range(4, 9)], 't')
        mao = pecas([('azul', 6)])
        jogo = partida(mao, [mesa], abriu=True)
        a = {'tipo': 'mesa', 'ids_pecas': ids(mao),
             'combinacoes': [ids(mesa[:3]), ids(mao + mesa[3:])]}
        b = deepcopy(a)
        b['combinacoes'] = [list(reversed(c)) for c in reversed(b['combinacoes'])]
        self.assertEqual(chave_acao(a, mao, jogo.mesa), chave_acao(b, mao, jogo.mesa))
        self.assertEqual(len(agrupar_acoes([a, b], mao, jogo.mesa)), 1)
        c = {'tipo': 'mesa', 'ids_pecas': ids(mao),
             'combinacoes': [ids(mesa[:2] + mao), ids(mesa[2:])]}
        # As duas partições são semanticamente equivalentes (duas sequências 4–6 e 6–8).
        self.assertEqual(chave_acao(a, mao, jogo.mesa), chave_acao(c, mao, jogo.mesa))
        c['combinacoes'] = [ids(mesa + mao)]
        self.assertNotEqual(chave_acao(a, mao, jogo.mesa), chave_acao(c, mao, jogo.mesa))

    def test_acoes_de_mesa_malformadas_nao_alteram_o_jogo(self):
        mesa = pecas([('azul', n) for n in (3, 4, 5)], 't')
        mao = pecas([('azul', 6)])
        jogo = partida(mao, [mesa], abriu=True)
        ambiente = AmbienteRL(lambda s: deepcopy(jogo), AgenteAleatorio())
        ambiente.reiniciar(42)
        antes = estado(ambiente.jogo)
        for layout in (None, [], ['invalido'], [[[]]], [ids(mao)], [ids(mesa + mao), ids(mao)]):
            with self.subTest(layout=layout):
                acao = {'tipo': 'mesa', 'ids_pecas': ids(mao), 'combinacoes': layout}
                with self.assertRaises(ValueError):
                    ambiente.executar_acao(acao)
                self.assertEqual(estado(ambiente.jogo), antes)
        acao = {'tipo': 'mesa', 'ids_pecas': [], 'combinacoes': [ids(mesa + mao)]}
        with self.assertRaises(ValueError):
            chave_acao(acao, mao, jogo.mesa)

    def test_ambiente_executa_abertura_conjunta_e_encerra_apenas_um_turno(self):
        jogo = partida(grupo(7, 'a') + grupo(3, 'b'))
        ambiente = AmbienteRL(lambda s: deepcopy(jogo), AgenteAleatorio())
        _, acoes = ambiente.reiniciar(42)
        acao = next(a for a in acoes if a['tipo'] == 'mesa')
        resultado = ambiente.passo(acao)
        self.assertTrue(resultado['terminado'])
        self.assertEqual(resultado['recompensa'], 1)
        self.assertEqual(ambiente.passos, 1)
        self.assertEqual(resultado['observacao']['aberturas'], (True, False))
        self.assertEqual(len(ambiente.jogo.mesa), 2)

    def test_passo_aceita_plano_legal_fora_da_amostra_automatica(self):
        mesa = pecas([('azul', n) for n in (3, 4, 5)], 't')
        mao = pecas([('azul', 6)])
        jogo = partida(mao, [mesa], abriu=True)
        ambiente = AmbienteRL(lambda s: deepcopy(jogo), AgenteAleatorio())
        _, acoes = ambiente.reiniciar(42)
        acao = {'tipo': 'mesa', 'ids_pecas': ids(mao), 'combinacoes': [list(reversed(ids(mesa + mao)))]}
        self.assertNotIn(acao, acoes)
        resultado = ambiente.passo(acao)
        self.assertTrue(resultado['terminado'])
        self.assertEqual(resultado['recompensa'], 1)

    def test_treino_e_avaliacao_executam_encaixes_e_reorganizacoes(self):
        for mao, mesa in ((pecas([('azul', 6)]), [pecas([('azul', n) for n in (3, 4, 5)], 't')]),
                          (pecas([('vermelho', 6)]), [pecas([('vermelho', n) for n in range(4, 9)], 't')]),
                          (pecas([('azul', 3), ('azul', 5)]), [grupo(4, 't') + pecas([('amarelo', 4)], 'extra')])):
            with self.subTest(mao=mao, mesa=mesa):
                jogo = partida(mao, mesa, abriu=True)
                ambiente = AmbienteRL(lambda s: deepcopy(jogo), AgenteAleatorio())
                obs, acoes = ambiente.reiniciar(42)
                agrupadas = agrupar_acoes(acoes, ambiente.jogo.jogador_atual.mao, ambiente.jogo.mesa)
                escolhida = next(c for c in agrupadas if c[0] == 'mesa')
                agente = AgenteQLearning(alpha=1, epsilon=0, semente=42)
                chave = chave_observacao(obs)
                agente.tabela_q[(chave, escolhida)] = 0.1
                resumo = treinar(ambiente, agente, 1)
                self.assertEqual(resumo[0]['retorno'], 1)
                self.assertEqual(agente.tabela_q[(chave, escolhida)], 1)
                anterior = agente.tabela_q.copy()
                self.assertEqual(avaliar(ambiente, agente, 1)['vitorias'], 1)
                self.assertEqual(agente.tabela_q, anterior)


if __name__ == '__main__':
    unittest.main()
