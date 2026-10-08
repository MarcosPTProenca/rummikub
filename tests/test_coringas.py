import unittest
from collections import Counter
from itertools import combinations

from src.agentes.aleatorio import AgenteAleatorio
from src.agentes.q_learning import AgenteQLearning
from src.jogo.combinacao import Combinacao
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from src.rl.ambiente import AmbienteRL
from src.rl.avaliacao import avaliar
from src.rl.representacao import agrupar_acoes, chave_acao, chave_observacao, observar
from src.rl.treinamento import treinar


class TestPecasCoringas(unittest.TestCase):
    def test_baralho_tem_104_pecas_normais_e_dois_coringas(self):
        jogo = Jogo(semente=42)
        pecas = jogo.monte + [p for j in jogo.jogadores for p in j.mao]
        self.assertEqual(len(pecas), 106)
        self.assertEqual(len({p.id for p in pecas}), 106)
        self.assertEqual(len(jogo.monte), 78)
        self.assertEqual(sum(p.cor == 'coringa' for p in pecas), 2)
        normais = Counter((p.cor, p.numero) for p in pecas if p.cor != 'coringa')
        self.assertEqual(len(normais), 52)
        self.assertTrue(all(copias == 2 for copias in normais.values()))

    def test_coringa_tem_representacao_unica_e_imutavel(self):
        coringa = Peca(id='j', cor='coringa', numero=0)
        self.assertEqual(coringa.numero, 0)
        with self.assertRaises(ValueError):
            coringa.numero = 7
        for cor, numero in (('azul', 0), ('coringa', 7), ('coringa', -1), ('azul', 14)):
            with self.subTest(cor=cor, numero=numero):
                with self.assertRaises(ValueError):
                    Peca(id='invalida', cor=cor, numero=numero)

    def test_limite_de_distribuicao_considera_106_pecas(self):
        jogo = Jogo(pecas_por_jogador=53, semente=42)
        self.assertEqual([len(j.mao) for j in jogo.jogadores], [53, 53])
        self.assertEqual(jogo.monte, [])
        with self.assertRaises(ValueError):
            Jogo(pecas_por_jogador=54)


CORINGA = ('coringa', 0)


def combinacao(tipos):
    return Combinacao(pecas=[Peca(id=str(i), cor=cor, numero=numero)
                             for i, (cor, numero) in enumerate(tipos)])


class TestCombinacoesCoringas(unittest.TestCase):
    def test_grupos_com_um_ou_dois_coringas(self):
        for tipos in ([('azul', 7), ('vermelho', 7), CORINGA],
                      [('azul', 7), CORINGA, CORINGA],
                      [('azul', 7), ('verde', 7), CORINGA, CORINGA]):
            with self.subTest(tipos=tipos):
                self.assertTrue(combinacao(tipos).eh_grupo())
                self.assertTrue(combinacao(tipos).eh_valido())

    def test_grupos_invalidos_nao_sao_consertados_por_coringas(self):
        for tipos in ([('azul', 7), ('azul', 7), CORINGA],
                      [('azul', 7), ('verde', 8), CORINGA],
                      [CORINGA, CORINGA],
                      [('azul', 7), ('verde', 7), ('amarelo', 7), ('vermelho', 7), CORINGA]):
            with self.subTest(tipos=tipos):
                self.assertFalse(combinacao(tipos).eh_grupo())

    def test_sequencias_preenchem_lacunas_e_extremidades(self):
        for tipos in ([('azul', 3), ('azul', 5), CORINGA],
                      [('azul', 2), ('azul', 3), CORINGA],
                      [('azul', 12), ('azul', 13), CORINGA],
                      [('azul', 1), CORINGA, CORINGA],
                      [('azul', 13), CORINGA, CORINGA],
                      [('azul', 4), ('azul', 7), CORINGA, CORINGA],
                      [('azul', n) for n in range(1, 13)] + [CORINGA]):
            with self.subTest(tipos=tipos):
                self.assertTrue(combinacao(tipos).eh_sequencia())

    def test_sequencias_invalidas_continuam_invalidas(self):
        for tipos in ([('azul', 3), ('verde', 4), CORINGA],
                      [('azul', 3), ('azul', 3), CORINGA],
                      [('azul', 3), ('azul', 6), CORINGA],
                      [('azul', 4), ('azul', 8), CORINGA, CORINGA],
                      [('azul', 13), ('azul', 1), CORINGA],
                      [CORINGA, CORINGA],
                      [('azul', n) for n in range(1, 14)] + [CORINGA]):
            with self.subTest(tipos=tipos):
                self.assertFalse(combinacao(tipos).eh_sequencia())

    def test_um_mesmo_coringa_nao_pode_ser_usado_duas_vezes(self):
        jogo = combinacao([('azul', 7), CORINGA])
        jogo.pecas.append(jogo.pecas[-1])
        self.assertFalse(jogo.eh_grupo())
        self.assertFalse(jogo.eh_sequencia())
        self.assertFalse(jogo.eh_valido())

    def test_coringa_pontua_como_substituto_sem_alterar_a_peca(self):
        for tipos, pontos in (([('azul', 7), ('verde', 7), CORINGA], 21),
                              ([('azul', 3), ('azul', 5), CORINGA], 12),
                              ([('azul', 1), ('azul', 2), CORINGA], 6),
                              ([('azul', 12), ('azul', 13), CORINGA], 36),
                              ([('azul', 5), ('azul', 6), CORINGA], 18)):
            with self.subTest(tipos=tipos):
                meld = combinacao(tipos)
                antes = [p.model_dump() for p in meld.pecas]
                self.assertEqual(meld.calcula_pontos(), pontos)
                self.assertEqual([p.model_dump() for p in meld.pecas], antes)


class TestAcoesCoringas(unittest.TestCase):
    def test_geracao_encontra_todas_as_combinacoes_sem_repetir_acoes(self):
        for tipos in ([('azul', 1), ('azul', 2), ('azul', 2), ('azul', 4),
                       ('verde', 2), ('vermelho', 2), CORINGA, CORINGA],
                      [('azul', 12), ('azul', 13), ('amarelo', 13), CORINGA, CORINGA],
                      [('azul', 1), CORINGA, CORINGA]):
            with self.subTest(tipos=tipos):
                jogo = Jogo(semente=42)
                jogo.jogador_atual.abriu = True
                mao = combinacao(tipos).pecas
                jogo.jogador_atual.mao = mao
                acoes = jogo.listar_acoes_validas()
                obtidas = [frozenset(a['ids_pecas']) for a in acoes if a['tipo'] == 'baixar']
                esperadas = {frozenset(p.id for p in pecas)
                             for tamanho in range(3, len(mao) + 1)
                             for pecas in combinations(mao, tamanho)
                             if Combinacao(pecas=list(pecas)).eh_valido()}
                self.assertEqual(set(obtidas), esperadas)
                self.assertEqual(len(obtidas), len(set(obtidas)))
                self.assertEqual(acoes[-1], {'tipo': 'comprar', 'ids_pecas': []})
                self.assertEqual(jogo.jogador_atual.mao, mao)

    def test_coringa_pode_substituir_peca_que_tambem_esta_na_mao(self):
        jogo = Jogo(semente=42)
        jogo.jogador_atual.abriu = True
        jogo.jogador_atual.mao = combinacao([('azul', 1), ('azul', 2), ('azul', 3), CORINGA]).pecas
        obtidas = {frozenset(a['ids_pecas']) for a in jogo.listar_acoes_validas() if a['tipo'] == 'baixar'}
        self.assertIn(frozenset(['0', '1', '2']), obtidas)
        self.assertIn(frozenset(['0', '1', '3']), obtidas)
        self.assertIn(frozenset(['0', '1', '2', '3']), obtidas)

    def test_baixar_coringa_remove_ids_reais_e_pode_ganhar(self):
        jogo = Jogo(semente=42)
        mao = combinacao([('azul', 3), ('azul', 5), CORINGA]).pecas
        jogo.jogador_atual.abriu = True
        jogo.jogador_atual.mao = mao.copy()
        jogador = jogo.jogador_atual
        acao = next(a for a in jogo.listar_acoes_validas() if a['tipo'] == 'baixar')
        jogo.baixar_combinacao(acao['ids_pecas'])
        self.assertTrue(jogo.terminou)
        self.assertIs(jogo.vencedor, jogador)
        self.assertEqual(jogador.mao, [])
        self.assertEqual({p.id for p in jogo.mesa[0].pecas}, {p.id for p in mao})
        self.assertTrue(jogo.mesa[0].eh_valido())
        self.assertEqual(jogo.mesa[0].pecas[-1].cor, 'coringa')


class TestCoringasRL(unittest.TestCase):
    def test_observacao_conta_coringas_separadamente(self):
        jogo = Jogo(semente=42)
        jogo.jogador_atual.mao = combinacao([('azul', 1), CORINGA, CORINGA]).pecas
        resultado = observar(jogo, 0, numero_maximo=3)
        self.assertEqual(resultado['mao'], (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2))
        self.assertEqual(len(observar(jogo, 0)['mao']), 53)
        chave = chave_observacao(resultado)
        jogo.jogador_atual.mao.pop()
        self.assertNotEqual(chave, chave_observacao(observar(jogo, 0, numero_maximo=3)))

    def test_comprar_coringa_libera_combinacao_no_proximo_turno(self):
        def fabrica(semente):
            jogo = Jogo(semente=semente)
            todas = jogo.monte + [p for j in jogo.jogadores for p in j.mao]
            por_id = {p.id: p for p in todas}
            jogo.jogadores[0].abriu = True
            jogo.jogadores[0].mao = [por_id['0'], por_id['4']]
            jogo.jogadores[1].mao = [por_id['26']]
            jogo.monte = [p for p in todas if p.id not in ('0', '4', '26', '104')] + [por_id['104']]
            return jogo
        ambiente = AmbienteRL(fabrica, AgenteAleatorio())
        ambiente.reiniciar(42)
        resultado = ambiente.passo({'tipo': 'comprar', 'ids_pecas': []})
        self.assertEqual(resultado['observacao']['mao'][-1], 1)
        baixar = next(a for a in resultado['acoes_validas'] if a['tipo'] == 'baixar')
        resultado = ambiente.passo(baixar)
        self.assertTrue(resultado['terminado'])
        self.assertEqual(resultado['recompensa'], 1)

    def test_chaves_equivalentes_preservam_multiplicidade_dos_coringas(self):
        mao = combinacao([('azul', 7), CORINGA, CORINGA]).pecas
        a = {'tipo': 'baixar', 'ids_pecas': ['0', '1']}
        b = {'tipo': 'baixar', 'ids_pecas': ['2', '0']}
        self.assertEqual(chave_acao(a, mao), chave_acao(b, mao))
        self.assertEqual(len(agrupar_acoes([a, b], mao)), 1)
        dupla = {'tipo': 'baixar', 'ids_pecas': ['0', '1', '2']}
        self.assertEqual(chave_acao(dupla, mao), ('baixar', (('azul', 7), CORINGA, CORINGA)))

    def test_treino_e_avaliacao_jogam_coringas_reais_e_conservam_106_pecas(self):
        for ids in (['0', '4', '104'], ['24', '104', '105']):
            with self.subTest(ids=ids):
                def fabrica(semente):
                    jogo = Jogo(semente=semente)
                    todas = jogo.monte + [p for j in jogo.jogadores for p in j.mao]
                    por_id = {p.id: p for p in todas}
                    jogo.jogadores[0].abriu = True
                    jogo.jogadores[0].mao = [por_id[i] for i in ids]
                    jogo.jogadores[1].mao = [por_id['26']]
                    jogo.monte = [p for p in todas if p.id not in ids and p.id != '26']
                    return jogo
                ambiente = AmbienteRL(fabrica, AgenteAleatorio())
                obs, acoes = ambiente.reiniciar(42)
                agente = AgenteQLearning(alpha=1, epsilon=0, semente=42)
                mao = ambiente.jogo.jogador_atual.mao
                baixar = next(c for c in agrupar_acoes(acoes, mao) if c[0] == 'baixar')
                chave = chave_observacao(obs)
                agente.tabela_q[(chave, baixar)] = 0.1
                resumos = treinar(ambiente, agente, 1, semente_inicial=42)
                self.assertEqual(resumos[0]['retorno'], 1)
                self.assertEqual(agente.tabela_q[(chave, baixar)], 1)
                anterior = agente.tabela_q.copy()
                self.assertEqual(avaliar(ambiente, agente, 1)['vitorias'], 1)
                self.assertEqual(agente.tabela_q, anterior)
                jogo = ambiente.jogo
                todas = (jogo.monte + [p for j in jogo.jogadores for p in j.mao]
                         + [p for c in jogo.mesa for p in c.pecas])
                self.assertEqual(len(todas), 106)
                self.assertEqual(len({p.id for p in todas}), 106)
                self.assertEqual(sum(p.cor == 'coringa' for p in todas), 2)
                self.assertTrue(jogo.mesa[0].eh_valido())


if __name__ == '__main__':
    unittest.main()
