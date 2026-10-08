import unittest


class TestJogo(unittest.TestCase):
    def test_partida_e_acoes(self):
        from src.jogo.jogo import Jogo
        from src.jogo.combinacao import Combinacao

        jogo = Jogo(semente=42)
        self.assertEqual([len(j.mao) for j in jogo.jogadores], [14, 14])
        pecas = jogo.monte + [p for j in jogo.jogadores for p in j.mao]
        self.assertEqual(len({p.id for p in pecas}), 106)
        por_id = {p.id: p for p in jogo.jogador_atual.mao}
        for acao in jogo.listar_acoes_validas():
            if acao['tipo'] == 'baixar':
                self.assertTrue(Combinacao(pecas=[por_id[i] for i in acao['ids_pecas']]).eh_valido())
        jogador = jogo.jogador_atual
        jogo.comprar_peca()
        self.assertEqual(len(jogador.mao), 15)
        self.assertEqual(jogo.indice_jogador_atual, 1)

    def test_baixa_invalida_nao_altera_partida(self):
        from src.jogo.jogo import Jogo

        jogo = Jogo(semente=42)
        mao = jogo.jogador_atual.mao.copy()
        for ids in ([mao[0].id] * 3, ['inexistente'], []):
            with self.assertRaises(ValueError):
                jogo.baixar_combinacao(ids)
            self.assertEqual(jogo.jogador_atual.mao, mao)
            self.assertEqual(jogo.mesa, [])
            self.assertEqual(jogo.indice_jogador_atual, 0)


if __name__ == '__main__':
    unittest.main()


class TestDesempate(unittest.TestCase):
    def fim_por_passagens(self, desempate, mao_a, mao_b):
        from src.jogo.jogo import Jogo
        from src.jogo.peca import Peca

        jogo = Jogo(semente=1, desempate=desempate) if desempate else Jogo(semente=1)
        for jogador, tipos in zip(jogo.jogadores, (mao_a, mao_b)):
            jogador.mao = [Peca(id=f'{jogador.id}{i}', cor=c, numero=n) for i, (c, n) in enumerate(tipos)]
        jogo.monte = []
        jogo.comprar_peca()
        jogo.comprar_peca()
        return jogo

    def test_padrao_e_pontos_menor_mao_vence(self):
        jogo = self.fim_por_passagens(None, [('azul', 3), ('azul', 4)], [('azul', 13)])
        self.assertTrue(jogo.terminou)
        self.assertEqual(jogo.vencedor.id, '0')

    def test_pontos_contam_coringa_como_30(self):
        jogo = self.fim_por_passagens('pontos', [('coringa', 0)], [('azul', 13), ('azul', 12)])
        self.assertEqual(jogo.vencedor.id, '1')

    def test_pecas_menos_pecas_vence_mesmo_com_mais_pontos(self):
        jogo = self.fim_por_passagens('pecas', [('azul', 13)], [('azul', 1), ('azul', 2)])
        self.assertEqual(jogo.vencedor.id, '0')

    def test_modo_empate_e_igualdade_real_nao_tem_vencedor(self):
        self.assertIsNone(self.fim_por_passagens('empate', [('azul', 1)], [('azul', 13)]).vencedor)
        self.assertIsNone(self.fim_por_passagens('pontos', [('azul', 5)], [('verde', 5)]).vencedor)

    def test_desempate_invalido_e_rejeitado(self):
        from src.jogo.jogo import Jogo
        with self.assertRaises(ValueError):
            Jogo(desempate='sorte')


class TestMuitosJogadores(unittest.TestCase):
    def test_ate_quatro_jogadores_usam_um_conjunto_de_106(self):
        from src.jogo.jogo import Jogo
        for n in (2, 3, 4):
            jogo = Jogo(quantidade_jogadores=n, semente=1)
            self.assertEqual(sum(len(j.mao) for j in jogo.jogadores) + len(jogo.monte), 106)

    def test_dez_jogadores_usam_tres_conjuntos_com_pecas_unicas(self):
        from src.jogo.jogo import Jogo
        jogo = Jogo(quantidade_jogadores=10, semente=1)
        todas = [p for j in jogo.jogadores for p in j.mao] + jogo.monte
        self.assertEqual((len(todas), len({p.id for p in todas})), (318, 318))
        self.assertEqual(sum(p.cor == 'coringa' for p in todas), 6)
        self.assertTrue(all(len(j.mao) == 14 for j in jogo.jogadores))

    def test_conjuntos_crescem_de_quatro_em_quatro_jogadores(self):
        from src.jogo.jogo import Jogo
        total = lambda n: (lambda j: sum(len(x.mao) for x in j.jogadores) + len(j.monte))(Jogo(quantidade_jogadores=n))
        self.assertEqual([total(n) for n in (5, 8, 9)], [212, 212, 318])

    def test_limites_de_jogadores(self):
        from src.jogo.jogo import Jogo
        for n in (1, 11):
            with self.assertRaises(ValueError):
                Jogo(quantidade_jogadores=n)
