import random
import unittest

from src.estrategias.arena import jogar
from src.estrategias.meta import OPCOES, MetaQ, estado_meta, treinar_meta
from src.estrategias.politicas import ESTRATEGIAS, criar
from src.agentes.q_learning import AgenteQLearning
from tests.test_turnos import grupo, partida, pecas


def conserva_pecas(jogo):
    return (sum(len(j.mao) for j in jogo.jogadores) + len(jogo.monte)
            + sum(len(c.pecas) for c in jogo.mesa))


class TestPoliticas(unittest.TestCase):
    def test_cada_estrategia_joga_partida_completa_e_conserva_pecas(self):
        for nome in ESTRATEGIAS:
            with self.subTest(nome=nome):
                politicas = [criar(nome, random.Random(1)), criar('max_pecas', random.Random(2))]
                resultado = jogar(politicas, semente=3)
                self.assertEqual(resultado['pecas_total'], 106)
                self.assertIn(resultado['vencedor'], (0, 1, None))
                self.assertGreater(resultado['decisoes'], 0)

    def test_partida_e_reprodutivel(self):
        def rodar():
            return jogar([criar('max_pecas', random.Random(1)),
                          criar('aleatorio', random.Random(2))], semente=5)
        self.assertEqual(rodar(), rodar())

    def test_max_pecas_prefere_mais_pecas_e_max_pontos_prefere_mais_pontos(self):
        jogo = partida(pecas([('azul', 1), ('azul', 2), ('azul', 3), ('azul', 4)]) + pecas(
            [('verde', 13), ('vermelho', 13), ('azul', 13)], 'q'), abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['p0', 'p1', 'p2', 'p3']},
                 {'tipo': 'baixar', 'ids_pecas': ['q0', 'q1', 'q2']},
                 {'tipo': 'comprar', 'ids_pecas': []}]
        mais_pecas = criar('max_pecas', random.Random(0))(jogo, acoes)
        mais_pontos = criar('max_pontos', random.Random(0))(jogo, acoes)
        minimo = criar('minimo', random.Random(0))(jogo, acoes)
        self.assertEqual(len(mais_pecas['ids_pecas']), 4)
        self.assertEqual(mais_pontos['ids_pecas'], ['q0', 'q1', 'q2'])
        self.assertEqual(minimo['ids_pecas'], ['q0', 'q1', 'q2'])

    def test_poupar_coringa_evita_coringa_e_compra_quando_so_ele_joga(self):
        mao = pecas([('azul', 5), ('azul', 6), ('coringa', 0)])
        jogo = partida(mao, abriu=True)
        acao = criar('poupar_coringa', random.Random(0))(jogo, jogo.listar_acoes_validas())
        self.assertEqual(acao['tipo'], 'baixar')  # baixar tudo vence o jogo
        jogo = partida(mao + pecas([('verde', 1)], 'z'), abriu=True)
        acao = criar('poupar_coringa', random.Random(0))(jogo, jogo.listar_acoes_validas())
        self.assertEqual(acao['tipo'], 'comprar')

    def test_so_baixar_ignora_manipulacao_da_mesa(self):
        jogo = partida(pecas([('azul', 4), ('verde', 9)]),
                       mesa=[pecas([('azul', 1), ('azul', 2), ('azul', 3)], 'm')], abriu=True)
        acao = criar('so_baixar', random.Random(0))(jogo, jogo.listar_acoes_validas())
        self.assertEqual(acao['tipo'], 'comprar')
        acao = criar('max_pecas', random.Random(0))(jogo, jogo.listar_acoes_validas())
        self.assertEqual(acao['tipo'], 'mesa')

    def test_cauteloso_compra_com_jogadas_pequenas(self):
        jogo = partida(pecas([('azul', 5), ('azul', 6), ('azul', 7), ('verde', 1)]), abriu=True)
        acao = criar('cauteloso', random.Random(0))(jogo, jogo.listar_acoes_validas())
        self.assertEqual(acao['tipo'], 'comprar')


class TestMeta(unittest.TestCase):
    def test_estado_meta_e_pequeno_e_estavel(self):
        jogo = partida(grupo(7) + pecas([('verde', 1)], 'z'), abriu=True)
        estado = estado_meta(jogo, jogo.listar_acoes_validas())
        self.assertEqual(estado, estado_meta(jogo, jogo.listar_acoes_validas()))
        hash(estado)

    def test_treino_atualiza_q_e_e_reprodutivel(self):
        def rodar():
            agente = AgenteQLearning(gamma=1.0, semente=1)
            resultados = treinar_meta(agente, 4, ['max_pecas', 'aleatorio'], semente_inicial=10)
            return dict(agente.tabela_q), resultados
        q1, r1 = rodar()
        q2, r2 = rodar()
        self.assertEqual((q1, r1), (q2, r2))
        self.assertTrue(q1)
        self.assertEqual(len(r1), 4)
        self.assertTrue(all(valor in (1.0, 0.0, -1.0) for valor in r1))

    def test_meta_q_gulosa_nao_altera_tabela(self):
        agente = AgenteQLearning(gamma=1.0, semente=1)
        treinar_meta(agente, 2, ['max_pecas'], semente_inicial=1)
        antes = dict(agente.tabela_q)
        jogar([MetaQ(agente, random.Random(1)), criar('aleatorio', random.Random(2))], semente=9)
        self.assertEqual(antes, agente.tabela_q)
        self.assertIn('max_pecas', OPCOES)


if __name__ == '__main__':
    unittest.main()


class TestJev(unittest.TestCase):
    def setUp(self):
        from src.estrategias.jev import Orcamento
        self.Orcamento = Orcamento
        self.jogo = partida(grupo(7) + pecas([('verde', 1)], 'z'), abriu=True)
        self.acoes = self.jogo.listar_acoes_validas()

    def agente(self, resposta, orcamento=None):
        from src.estrategias.jev import JevMeta
        chamadas = []

        def transporte(corpo):
            chamadas.append(corpo)
            return resposta

        return JevMeta(transporte, random.Random(0), orcamento or self.Orcamento(1.0)), chamadas

    def resposta(self, escolha, confianca=0.9, tokens=1000):
        return {"answers": {"estrategia": {"type": "choice", "choice": escolha,
                                           "probabilities": {escolha: 1.0}, "confidence": confianca}},
                "usage": {"input_tokens": tokens, "output_tokens": 5}}

    def test_executa_a_estrategia_escolhida_e_contabiliza_custo(self):
        agente, chamadas = self.agente(self.resposta('comprar', tokens=1_000_000))
        self.assertEqual(agente(self.jogo, self.acoes)['tipo'], 'comprar')
        self.assertEqual(set(chamadas[0]['questions']['estrategia']['criteria']), set(OPCOES))
        self.assertEqual(chamadas[0]['model'], 'jev-1.13.0')
        self.assertAlmostEqual(agente.orcamento.gasto, 0.042)
        self.assertEqual(agente.contagem['chamadas'], 1)

    def test_confianca_baixa_usa_fallback(self):
        agente, _ = self.agente(self.resposta('comprar', confianca=0.1))
        self.assertEqual(agente(self.jogo, self.acoes)['tipo'], 'baixar')
        self.assertEqual(agente.contagem['fallbacks'], 1)

    def test_orcamento_esgotado_nao_chama_a_api(self):
        agente, chamadas = self.agente(self.resposta('comprar'), self.Orcamento(0.0))
        self.assertEqual(agente(self.jogo, self.acoes)['tipo'], 'baixar')
        self.assertEqual(chamadas, [])


class TestExperimentos(unittest.TestCase):
    def test_cli_torneio_grava_partidas_pareadas_e_retoma(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as pasta:
            saida = Path(pasta) / 't.jsonl'
            comando = [sys.executable, '-m', 'src.experimentos', 'torneio', '--sementes', '1',
                       '--processos', '2', '--saida', str(saida)]
            primeira = subprocess.run(comando, cwd=raiz, capture_output=True, text=True, check=True, timeout=300)
            registros = [json.loads(l) for l in saida.read_text().splitlines()]
            self.assertEqual(len(registros), 42)  # 21 pares x 1 semente x 2 assentos
            self.assertEqual({r['assento_a'] for r in registros}, {0, 1})
            self.assertTrue(all(r['pontos_a'] in (0.0, 0.5, 1.0) for r in registros))
            segunda = subprocess.run(comando, cwd=raiz, capture_output=True, text=True, check=True, timeout=300)
            self.assertIn('0 partidas pendentes', segunda.stdout)
            self.assertEqual(len(saida.read_text().splitlines()), 42)
            self.assertEqual({r['desempate'] for r in registros}, {'pontos'})


class TestMecanismo(unittest.TestCase):
    def cenario(self, abriu):
        mao = pecas([('azul', 1), ('azul', 2), ('azul', 3), ('coringa', 0)]) + pecas(
            [('verde', 5), ('vermelho', 5), ('azul', 5)], 'q')
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['p0', 'p1', 'p2', 'p3']},
                 {'tipo': 'baixar', 'ids_pecas': ['q0', 'q1', 'q2']},
                 {'tipo': 'comprar', 'ids_pecas': []}]
        return partida(mao, abriu=abriu), acoes

    def escolhe(self, nome, abriu):
        jogo, acoes = self.cenario(abriu)
        return criar(nome, random.Random(0))(jogo, acoes)['ids_pecas']

    def test_penalidade_por_coringa_pesa_em_pecas(self):
        self.assertEqual(self.escolhe('penal_0.5', True), ['p0', 'p1', 'p2', 'p3'])
        self.assertEqual(self.escolhe('penal_2', True), ['q0', 'q1', 'q2'])
        self.assertEqual(self.escolhe('bonus_2', True), ['p0', 'p1', 'p2', 'p3'])

    def test_poupar_coringa_so_antes_ou_so_depois_da_abertura(self):
        poupa = ['q0', 'q1', 'q2']
        gasta = ['p0', 'p1', 'p2', 'p3']
        self.assertEqual(self.escolhe('poupar_antes', False), poupa)
        self.assertEqual(self.escolhe('poupar_antes', True), gasta)
        self.assertEqual(self.escolhe('poupar_depois', False), gasta)
        self.assertEqual(self.escolhe('poupar_depois', True), poupa)

    def test_sem_roubo_remove_jogadas_que_tiram_coringa_do_conjunto_da_mesa(self):
        from src.estrategias.politicas import sem_roubo
        mesa = [pecas([('azul', 1), ('azul', 2), ('coringa', 0)], 'm')]
        jogo = partida(pecas([('azul', 4), ('vermelho', 7), ('verde', 7), ('azul', 7)]), mesa, abriu=True)
        estende = {'tipo': 'mesa', 'ids_pecas': ['p0'], 'combinacoes': [['m0', 'm1', 'm2', 'p0']]}
        rouba = {'tipo': 'mesa', 'ids_pecas': ['p1', 'p2'], 'combinacoes': [['m0', 'm1', 'p0'], ['m2', 'p1', 'p2']]}
        comprar = {'tipo': 'comprar', 'ids_pecas': []}
        recebidas = []
        sem_roubo(lambda j, a: recebidas.append(a) or a[0])(jogo, [estende, rouba, comprar])
        self.assertEqual(recebidas, [[estende, comprar]])

    def test_partida_informa_telemetria_de_coringas(self):
        r = jogar([criar('max_pecas', random.Random(1)), criar('poupar_coringa', random.Random(2))], semente=4)
        t = r['telemetria']
        self.assertEqual(sum(x['jogadas'] + x['compras'] for x in t), r['decisoes'])
        self.assertEqual(len(r['coringas_mao']), 2)
        self.assertIn(r['saida_com_coringa'], (True, False))

    def test_poupar_sem_final_nunca_gasta_coringa_nem_para_sair(self):
        mao = pecas([('azul', 1), ('azul', 2), ('azul', 3), ('coringa', 0)])
        jogo = partida(mao, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['p0', 'p1', 'p2', 'p3']},
                 {'tipo': 'comprar', 'ids_pecas': []}]
        self.assertEqual(criar('poupar_coringa', random.Random(0))(jogo, acoes)['ids_pecas'],
                         ['p0', 'p1', 'p2', 'p3'])
        self.assertEqual(criar('poupar_sem_final', random.Random(0))(jogo, acoes)['tipo'], 'comprar')


class TestDesempateNaArena(unittest.TestCase):
    def jogar(self, desempate):
        politicas = [criar('cauteloso', random.Random(1)), criar('cauteloso', random.Random(2))]
        return jogar(politicas, 1000, desempate=desempate)

    def test_monte_esgotado_vira_vitoria_da_menor_mao_ou_empate_conforme_a_regra(self):
        self.assertIsNone(self.jogar('empate')['vencedor'])
        resultado = self.jogar('pontos')
        self.assertEqual(resultado['vencedor'], 1)
        self.assertEqual(resultado['maos'], [9, 6])

    def test_experimento_registra_a_regra_usada(self):
        from src.experimentos import partida
        registro = partida('cauteloso', 'cauteloso', 1001, 0, desempate='empate')
        self.assertEqual((registro['desempate'], registro['pontos_a']), ('empate', 0.5))
        self.assertEqual(partida('cauteloso', 'cauteloso', 1001, 0)['pontos_a'], 0.0)

    def test_recompensa_rl_deixa_de_ser_zero_no_desempate(self):
        from tests.test_ambiente import COMPRAR, criar_ambiente, fabrica_controlada
        ambiente = criar_ambiente(fabrica_jogo=fabrica_controlada([grupo(3)[:1], grupo(9, 'op')[:1]]))
        ambiente.reiniciar(0)
        self.assertEqual(ambiente.passo(COMPRAR)['recompensa'], 1)


class TestCoringasComprados(unittest.TestCase):
    def test_arena_conta_coringas_comprados_por_jogador(self):
        r = jogar([criar('max_pecas', random.Random(1)), criar('poupar_coringa', random.Random(2))], semente=4)
        comprados = [t['coringas_comprados'] for t in r['telemetria']]
        from src.jogo.jogo import Jogo
        iniciais = [sum(p.cor == 'coringa' for p in j.mao) for j in Jogo(semente=4).jogadores]
        self.assertLessEqual(sum(comprados) + sum(iniciais), 2)
        self.assertEqual(r['compras_total'], sum(t['compras'] for t in r['telemetria']))

    def test_compra_do_topo_com_coringa_e_registrada(self):
        from src.jogo.peca import Peca
        import src.estrategias.arena as arena
        jogo = partida(pecas([('azul', 1)]))
        jogo.monte = jogo.monte[:3] + [Peca(id='j', cor='coringa', numero=0)]
        telemetria = {'jogadas': 0, 'compras': 0, 'coringas_jogados': 0, 'coringas_comprados': 0}
        arena.registrar_compra(jogo, {'tipo': 'comprar', 'ids_pecas': []}, telemetria)
        self.assertEqual(telemetria['coringas_comprados'], 1)


class TestDesempateNoTreino(unittest.TestCase):
    def test_treinar_meta_repassa_o_desempate_a_arena(self):
        from unittest import mock
        from src.estrategias import meta
        agente = AgenteQLearning(gamma=1.0, semente=1)
        with mock.patch.object(meta, 'jogar', wraps=meta.jogar) as espiao:
            treinar_meta(agente, 2, ['max_pecas'], semente_inicial=1, desempate='pecas')
        self.assertEqual({c.kwargs['desempate'] for c in espiao.call_args_list}, {'pecas'})

    def test_cli_treinar_registra_a_regra_e_usa_sufixo(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as pasta:
            saida = Path(pasta) / 'treino.jsonl'
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'treinar', '--replicas', '1',
                            '--episodios', '2', '--avaliacao', '1', '--processos', '1', '--desempate', 'pontos',
                            '--saida', str(saida)], cwd=raiz, capture_output=True, text=True, check=True, timeout=300)
            linha = json.loads(saida.read_text().splitlines()[0])
            self.assertEqual({r['desempate'] for r in linha['avaliacao']}, {'pontos'})
            self.assertEqual(linha['desempate'], 'pontos')

