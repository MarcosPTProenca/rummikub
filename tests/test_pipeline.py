import unittest
from copy import deepcopy

from src.agentes.q_learning import AgenteQLearning
from src.rl.avaliacao import avaliar
from src.rl.representacao import agrupar_acoes, chave_observacao
from src.rl.treinamento import treinar
from test_ambiente import criar_ambiente, fabrica_controlada, grupo, COMPRAR


class TestPipeline(unittest.TestCase):
    def test_numero_episodios_invalido(self):
        for funcao in (treinar, avaliar):
            for episodios in (0, -1, 1.5):
                with self.subTest(funcao=funcao, episodios=episodios):
                    with self.assertRaises(ValueError):
                        funcao(criar_ambiente(), AgenteQLearning(), episodios)

    def test_treinamento_terminal_atualiza_apenas_decisao_anterior(self):
        ambiente = criar_ambiente(fabrica_jogo=fabrica_controlada([grupo(), grupo('op')]))
        observacao, acoes = ambiente.reiniciar(0)
        chave = chave_observacao(observacao)
        baixar = next(c for c in agrupar_acoes(acoes, ambiente.jogo.jogador_atual.mao) if c[0] == 'baixar')
        agente = AgenteQLearning(alpha=1, epsilon=0, semente=0)
        agente.tabela_q[(chave, baixar)] = 0.5
        resumo = treinar(ambiente, agente, 1)
        self.assertEqual(resumo, [{'retorno': 1, 'passos': 1, 'terminado': True, 'truncado': False}])
        self.assertEqual(agente.tabela_q, {(chave, baixar): 1})

    def test_truncamento_bootstrap_usa_novas_acoes(self):
        ambiente = criar_ambiente(limite_passos=1)
        obs, _ = ambiente.reiniciar(42)
        chave = chave_observacao(obs)
        futuro = ambiente.passo(COMPRAR)
        futuras = agrupar_acoes(futuro['acoes_validas'], ambiente.jogo.jogadores[0].mao, ambiente.jogo.mesa)
        proxima = chave_observacao(futuro['observacao'])
        agente = AgenteQLearning(alpha=1, gamma=0.5, epsilon=0)
        compra = ('comprar', ())
        agente.tabela_q[(chave, compra)] = 1
        agente.tabela_q[(proxima, next(iter(futuras)))] = 4
        resumo = treinar(ambiente, agente, 1, semente_inicial=42)
        self.assertTrue(resumo[0]['truncado'])
        self.assertFalse(resumo[0]['terminado'])
        self.assertEqual(agente.tabela_q[(chave, compra)], 2)

    def test_avaliacao_nao_altera_aprendizado_e_ignora_exploracao(self):
        ambiente = criar_ambiente(fabrica_jogo=fabrica_controlada([grupo(), grupo('op')]))
        obs, acoes = ambiente.reiniciar(0)
        chave = chave_observacao(obs)
        baixar = next(c for c in agrupar_acoes(acoes, ambiente.jogo.jogador_atual.mao) if c[0] == 'baixar')
        agente = AgenteQLearning(epsilon=1)
        agente.tabela_q[(chave, baixar)] = 5
        anterior = deepcopy(agente.tabela_q)
        metricas = avaliar(ambiente, agente, 4)
        self.assertEqual(metricas, {'vitorias': 4, 'derrotas': 0, 'empates': 0,
                                   'truncados': 0, 'taxa_vitoria': 1, 'media_passos': 1})
        self.assertEqual(agente.tabela_q, anterior)
        self.assertEqual(agente.epsilon, 1)

    def test_avaliacao_contabiliza_derrota_empate_e_truncamento(self):
        derrota = criar_ambiente(fabrica_jogo=fabrica_controlada([grupo()[:1], grupo('op')]))
        # O adversário sorteia baixar para a semente 1, após a passagem do agente.
        metricas = avaliar(derrota, AgenteQLearning(), 1, semente_inicial=1)
        self.assertEqual(metricas['derrotas'], 1)
        self.assertEqual(metricas['empates'], 0)
        empate = criar_ambiente(fabrica_jogo=fabrica_controlada([grupo()[:1], grupo('op')[:1]]))
        self.assertEqual(avaliar(empate, AgenteQLearning(), 1)['empates'], 1)
        truncado = criar_ambiente(limite_passos=1)
        metricas = avaliar(truncado, AgenteQLearning(), 3)
        self.assertEqual(metricas['truncados'], 3)
        self.assertEqual(metricas['empates'], 0)
        self.assertEqual(metricas['media_passos'], 1)

    def test_partidas_reais_treino_avaliacao_e_conservacao_das_pecas(self):
        agente = AgenteQLearning(semente=42)
        ambiente = criar_ambiente()
        resumos = treinar(ambiente, agente, 5, semente_inicial=42)
        self.assertEqual(len(resumos), 5)
        self.assertTrue(all(r['terminado'] != r['truncado'] for r in resumos))
        self.assertTrue(agente.tabela_q)
        anterior = deepcopy(agente.tabela_q)
        metricas = avaliar(ambiente, agente, 5)
        self.assertEqual(sum(metricas[c] for c in ('vitorias', 'derrotas', 'empates', 'truncados')), 5)
        self.assertEqual(agente.tabela_q, anterior)
        jogo = ambiente.jogo
        pecas = jogo.monte + [p for j in jogo.jogadores for p in j.mao] + [p for c in jogo.mesa for p in c.pecas]
        self.assertEqual(len(pecas), 106)
        self.assertEqual(len({p.id for p in pecas}), 106)
        self.assertTrue(all(c.eh_valido() for c in jogo.mesa))

    def test_treinamento_e_reproduzivel(self):
        a = AgenteQLearning(semente=42)
        b = AgenteQLearning(semente=42)
        self.assertEqual(treinar(criar_ambiente(), a, 3), treinar(criar_ambiente(), b, 3))
        self.assertEqual(a.tabela_q, b.tabela_q)


if __name__ == '__main__':
    unittest.main()
