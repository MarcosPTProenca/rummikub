import unittest
from copy import deepcopy

from src.jogo.combinacao import Combinacao
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca


def pecas(tipos, prefixo='p'):
    return [Peca(id=f'{prefixo}{i}', cor=cor, numero=n) for i, (cor, n) in enumerate(tipos)]


def grupo(n, prefixo='g'):
    return pecas([('azul', n), ('verde', n), ('vermelho', n)], prefixo)


def ids(pecas):
    return [p.id for p in pecas]


def estado(jogo):
    return deepcopy(([j.model_dump() for j in jogo.jogadores],
                     [c.model_dump() for c in jogo.mesa], ids(jogo.monte),
                     jogo.indice_jogador_atual, jogo.passagens_consecutivas,
                     jogo.terminou, jogo.vencedor.id if jogo.vencedor else None))


def partida(mao, mesa=(), abriu=False):
    jogo = Jogo(semente=42)
    jogo.jogador_atual.mao = mao
    # A flag é configurada só para cenários posteriores à primeira baixa.
    if abriu:
        jogo.jogador_atual.abriu = True
    jogo.mesa = [Combinacao(pecas=list(c)) for c in mesa]
    return jogo


class TestTurnos(unittest.TestCase):
    def test_inicio_tem_14_pecas_e_ninguem_fez_abertura(self):
        jogo = Jogo(semente=42)
        self.assertEqual([len(j.mao) for j in jogo.jogadores], [14, 14])
        self.assertTrue(all(not j.abriu for j in jogo.jogadores))

    def test_30_pontos_sao_da_baixa_nao_da_mao_inteira(self):
        baixa = pecas([('azul', 3), ('azul', 4), ('azul', 5)])
        extras = pecas([('verde', 13), ('vermelho', 13)], 'extra')
        jogo = partida(baixa + extras)
        antes = estado(jogo)
        with self.assertRaises(ValueError):
            jogo.baixar_combinacao(ids(baixa))
        self.assertEqual(estado(jogo), antes)

    def test_abertura_exata_de_30_pontos(self):
        baixa = pecas([('azul', 9), ('azul', 10), ('azul', 11)])
        jogo = partida(baixa + grupo(1, 'extra'))
        jogador = jogo.jogador_atual
        jogo.baixar_combinacao(ids(baixa))
        self.assertTrue(jogador.abriu)
        self.assertEqual(jogo.indice_jogador_atual, 1)
        self.assertEqual(len(jogador.mao), 3)
        self.assertFalse(jogo.terminou)

    def test_abertura_soma_varias_combinacoes_em_um_turno(self):
        a, b = grupo(7, 'a'), grupo(3, 'b')
        jogo = partida(a + b + grupo(1, 'extra'))
        jogador = jogo.jogador_atual
        jogo.baixar_combinacoes([ids(a), ids(b)])
        self.assertTrue(jogador.abriu)
        self.assertEqual(len(jogo.mesa), 2)
        self.assertEqual(len(jogador.mao), 3)
        self.assertEqual(jogo.indice_jogador_atual, 1)

    def test_coringas_podem_completar_30_na_abertura(self):
        mao = pecas([('azul', 9), ('coringa', 0), ('coringa', 0)])
        self.assertEqual(Combinacao(pecas=mao).calcula_pontos(), 30)
        jogo = partida(mao)
        jogo.baixar_combinacao(ids(mao))
        self.assertTrue(jogo.terminou)
        self.assertEqual(jogo.vencedor.id, '0')

    def test_depois_da_abertura_pode_baixar_menos_de_30_e_varias(self):
        a, b = grupo(1, 'a'), grupo(2, 'b')
        jogo = partida(a + b + grupo(1, 'extra'), abriu=True)
        jogo.baixar_combinacoes([ids(a), ids(b)])
        self.assertEqual(len(jogo.mesa), 2)
        self.assertEqual(jogo.indice_jogador_atual, 1)
        self.assertEqual(len(jogo.jogadores[0].mao), 3)

    def test_acrescentar_pecas_pode_ganhar(self):
        mesa = pecas([('azul', 3), ('azul', 4), ('azul', 5)], 't')
        mao = pecas([('azul', 6), ('azul', 7)])
        jogo = partida(mao, [mesa], abriu=True)
        jogo.acrescentar_pecas(0, ids(mao))
        self.assertTrue(jogo.terminou)
        self.assertEqual(len(jogo.mesa[0].pecas), 5)
        self.assertEqual(jogo.jogadores[0].mao, [])

    def test_acrescentar_antes_da_abertura_e_proibido(self):
        mesa = pecas([('azul', 3), ('azul', 4), ('azul', 5)], 't')
        mao = pecas([('azul', 6)])
        jogo = partida(mao, [mesa])
        antes = estado(jogo)
        with self.assertRaises(ValueError):
            jogo.acrescentar_pecas(0, ids(mao))
        self.assertEqual(estado(jogo), antes)

    def test_reorganizar_separando_sequencia(self):
        mesa = pecas([('vermelho', n) for n in range(4, 9)], 't')
        mao = pecas([('vermelho', 6)])
        jogo = partida(mao, [mesa], abriu=True)
        jogo.jogar_turno([ids(mesa[:3]), ids(mao + mesa[3:])])
        self.assertTrue(jogo.terminou)
        self.assertEqual([len(c.pecas) for c in jogo.mesa], [3, 3])
        self.assertTrue(all(c.eh_valido() for c in jogo.mesa))

    def test_reorganizar_libera_peca_de_grupo_para_nova_sequencia(self):
        mesa = grupo(4, 't') + pecas([('amarelo', 4)], 't-extra')
        mao = pecas([('azul', 3), ('azul', 5)])
        jogo = partida(mao, [mesa], abriu=True)
        jogo.jogar_turno([ids(mesa[1:]), ids(mao + mesa[:1])])
        self.assertTrue(jogo.terminou)
        self.assertTrue(all(c.eh_valido() for c in jogo.mesa))

    def test_reorganizar_junta_duas_sequencias(self):
        a = pecas([('azul', n) for n in (3, 4, 5)], 'a')
        b = pecas([('azul', n) for n in (7, 8, 9)], 'b')
        mao = pecas([('azul', 6)])
        jogo = partida(mao, [a, b], abriu=True)
        jogo.jogar_turno([ids(a + mao + b)])
        self.assertEqual(len(jogo.mesa), 1)
        self.assertTrue(jogo.terminou)

    def test_mesa_invalida_nao_altera_estado_nem_perde_pecas(self):
        mesa = pecas([('azul', n) for n in (3, 4, 5)], 't')
        mao = pecas([('azul', 6)])
        jogo = partida(mao, [mesa], abriu=True)
        antes = estado(jogo)
        propostas = [None, [], [ids(mesa)], [ids(mao)],
                     [ids(mesa + mao), ids(mao)], [ids(mesa[:2] + mao)],
                     [ids(mesa) + ['ausente']], [ids(mesa) + [jogo.monte[0].id]],
                     [ids(mesa) + [jogo.jogadores[1].mao[0].id]], [ids(mesa), []],
                     ['invalida'], [[[]]]]
        for proposta in propostas:
            with self.subTest(proposta=proposta):
                with self.assertRaises(ValueError):
                    jogo.jogar_turno(proposta)
                self.assertEqual(estado(jogo), antes)

    def test_abertura_nao_pode_usar_pecas_da_mesa(self):
        mesa = grupo(10, 't') + pecas([('amarelo', 10)], 'outro')
        mao = pecas([('azul', 9), ('azul', 11)])
        jogo = partida(mao, [mesa])
        antes = estado(jogo)
        with self.assertRaises(ValueError):
            jogo.jogar_turno([ids(mesa[1:]), ids(mao + mesa[:1])])
        self.assertEqual(estado(jogo), antes)

    def test_combinacao_invalida_no_lote_nao_baixa_as_outras(self):
        mao = grupo(10) + pecas([('azul', 1)], 'extra')
        jogo = partida(mao)
        antes = estado(jogo)
        with self.assertRaises(ValueError):
            jogo.baixar_combinacoes([ids(mao[:3]), ids(mao[3:])])
        self.assertEqual(estado(jogo), antes)


class TestAcoesTurno(unittest.TestCase):
    def verificar_acoes(self, jogo):
        acoes = jogo.listar_acoes_validas()
        self.assertEqual(acoes[-1], {'tipo': 'comprar', 'ids_pecas': []})
        for acao in acoes:
            copia = deepcopy(jogo)
            if acao['tipo'] == 'baixar':
                copia.baixar_combinacao(acao['ids_pecas'])
            elif acao['tipo'] == 'mesa':
                copia.jogar_turno(acao['combinacoes'])
            else:
                copia.comprar_peca()
            self.assertTrue(copia.terminou or copia.indice_jogador_atual == 1)
        return acoes

    def test_sem_30_pontos_so_pode_comprar(self):
        jogo = partida(grupo(7))
        self.assertEqual(jogo.listar_acoes_validas(), [{'tipo': 'comprar', 'ids_pecas': []}])

    def test_gerador_encontra_abertura_somando_duas_combinacoes(self):
        jogo = partida(grupo(7, 'a') + grupo(3, 'b'))
        acoes = self.verificar_acoes(jogo)
        self.assertTrue(any(a['tipo'] == 'mesa' and len(a['ids_pecas']) == 6 for a in acoes))

    def test_gerador_encontra_varias_baixas_apos_abertura(self):
        jogo = partida(grupo(1, 'a') + grupo(2, 'b'), abriu=True)
        acoes = self.verificar_acoes(jogo)
        self.assertTrue(any(a['tipo'] == 'mesa' and len(a['ids_pecas']) == 6 for a in acoes))

    def test_gerador_encontra_encaixe_separacao_e_juncao(self):
        cenarios = [
            (pecas([('azul', 6)]), [pecas([('azul', n) for n in (3, 4, 5)], 't')]),
            (pecas([('vermelho', 6)]), [pecas([('vermelho', n) for n in range(4, 9)], 't')]),
            (pecas([('azul', 3), ('azul', 5)]),
             [grupo(4, 't') + pecas([('amarelo', 4)], 'extra')]),
            (pecas([('azul', 6)]), [pecas([('azul', n) for n in (3, 4, 5)], 'a'),
                                   pecas([('azul', n) for n in (7, 8, 9)], 'b')]),
        ]
        for mao, mesa in cenarios:
            with self.subTest(mao=mao, mesa=mesa):
                jogo = partida(mao, mesa, abriu=True)
                acoes = self.verificar_acoes(jogo)
                self.assertTrue(any(a['tipo'] == 'mesa' and set(a['ids_pecas']) == set(ids(mao))
                                    for a in acoes))

    def test_gerador_nao_manipula_mesa_antes_da_abertura(self):
        jogo = partida(pecas([('azul', 6)]), [pecas([('azul', n) for n in (3, 4, 5)], 't')])
        self.assertEqual(jogo.listar_acoes_validas(), [{'tipo': 'comprar', 'ids_pecas': []}])


if __name__ == '__main__':
    unittest.main()


class TestModoRapido(unittest.TestCase):
    def cenario(self):
        mesa = grupo(4, 't') + pecas([('amarelo', 4)], 't-extra')
        mao = pecas([('azul', 3), ('azul', 5)])
        return partida(mao, [mesa], abriu=True), mao

    def test_completo_gera_o_rearranjo_e_rapido_nao(self):
        jogo, mao = self.cenario()
        usa_as_duas = lambda a: a['tipo'] == 'mesa' and set(ids(mao)) <= set(a['ids_pecas'])
        self.assertTrue(any(usa_as_duas(a) for a in jogo.listar_acoes_validas()))
        rapidas = jogo.listar_acoes_validas(rearranjar=False)
        self.assertFalse(any(usa_as_duas(a) for a in rapidas))
        self.assertEqual(rapidas[-1]['tipo'], 'comprar')

    def test_rapido_so_devolve_acoes_aceitas_pelo_jogo(self):
        for semente in range(5):
            jogo = Jogo(semente=semente)
            for _ in range(12):
                acoes = jogo.listar_acoes_validas(rearranjar=False)
                for acao in acoes:
                    copia = deepcopy(jogo)
                    if acao['tipo'] == 'comprar':
                        copia.comprar_peca()
                    elif acao['tipo'] == 'baixar':
                        copia.baixar_combinacao(acao['ids_pecas'])
                    else:
                        copia.jogar_turno(acao['combinacoes'])
                if jogo.terminou:
                    break
                jogo.comprar_peca()

