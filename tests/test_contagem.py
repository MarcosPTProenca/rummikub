import random
import unittest

from src.estrategias.contagem import (contagem_escondidas, copias_por_tipo, desbloqueios_por_numero,
                                      extensoes_da_mesa, potencial_ponderado, raridade, travantes_gastos)
from src.estrategias.politicas import criar
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from tests.test_turnos import grupo, partida, pecas


class TestContagem(unittest.TestCase):
    def test_mesa_e_mao_descontam_das_copias(self):
        jogo = partida(pecas([('azul', 3)], 'm'), [grupo(3)])
        escondidas = contagem_escondidas(jogo)
        self.assertEqual(escondidas[('azul', 3)], 0)
        self.assertEqual(escondidas[('verde', 3)], 1)
        self.assertEqual(escondidas[('amarelo', 3)], 2)

    def test_so_pecas_normais_em_52_tipos(self):
        escondidas = contagem_escondidas(Jogo(semente=1))
        self.assertEqual(len(escondidas), 52)
        self.assertTrue(all(c >= 0 for c in escondidas.values()))

    def test_escondidas_sao_as_pecas_dos_oponentes_e_do_monte(self):
        jogo = Jogo(semente=7)
        oculto = [p for o in jogo.jogadores[1:] for p in o.mao] + jogo.monte
        esperado = sum(p.cor != 'coringa' for p in oculto)
        self.assertEqual(sum(contagem_escondidas(jogo).values()), esperado)

    def test_copias_crescem_com_os_conjuntos(self):
        self.assertEqual(copias_por_tipo(Jogo(semente=1)), 2)
        self.assertEqual(copias_por_tipo(Jogo(quantidade_jogadores=5, semente=1)), 4)

    def test_raridade_vai_de_0_a_1(self):
        jogo = partida(pecas([('azul', 3)], 'm'), [grupo(3)])
        escondidas = contagem_escondidas(jogo)
        self.assertEqual(raridade(jogo, escondidas, ('azul', 3)), 1.0)
        self.assertEqual(raridade(jogo, escondidas, ('verde', 3)), 0.5)
        self.assertEqual(raridade(jogo, escondidas, ('amarelo', 3)), 0.0)


if __name__ == '__main__':
    unittest.main()


def mesa_sem(tipos_esgotados):
    """Mesa com as duas cópias de cada tipo dado, em grupos de 3 (cada grupo traz 1 cópia dele)."""
    grupos = []
    for k, (cor, n) in enumerate(tipos_esgotados):
        outras = [c for c in ('azul', 'vermelho', 'verde', 'amarelo') if c != cor][:2]
        for r in range(2):
            grupos.append(pecas([(cor, n), (outras[0], n), (outras[1], n)], f'x{k}{r}'))
    return grupos


class TestPotencialPonderado(unittest.TestCase):
    def test_par_so_vale_se_ainda_da_para_completar(self):
        mao = pecas([('azul', 5), ('azul', 6)])
        cheio = {t: 2 for t in contagem_escondidas(partida([])).keys()}
        sem_completar = {**cheio, ('azul', 4): 0, ('azul', 7): 0}
        self.assertEqual(potencial_ponderado(mao, cheio, 2), 1.0)
        self.assertEqual(potencial_ponderado(mao, sem_completar, 2), 0.0)

    def test_grupo_usa_as_outras_cores(self):
        mao = pecas([('azul', 3), ('verde', 3)])
        cheio = dict.fromkeys(contagem_escondidas(partida([])), 2)
        parcial = {**cheio, ('vermelho', 3): 0, ('amarelo', 3): 1}
        self.assertEqual(potencial_ponderado(mao, cheio, 2), 1.0)
        self.assertEqual(potencial_ponderado(mao, parcial, 2), 0.5)

    def test_ignora_coringa(self):
        mao = pecas([('azul', 3), ('coringa', 0)])
        self.assertEqual(potencial_ponderado(mao, dict.fromkeys(contagem_escondidas(partida([])), 2), 2), 0.0)


class TestContaRara(unittest.TestCase):
    def test_prefere_manter_o_par_que_ainda_completa(self):
        mao = pecas([('azul', 5), ('azul', 6), ('azul', 7), ('verde', 9), ('verde', 10), ('verde', 11)], 'm')
        # azul 4 e azul 8 esgotados: os pares azuis não completam mais; os verdes sim
        jogo = partida(mao, mesa_sem([('azul', 4), ('azul', 8)]), abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['m0', 'm1', 'm2']},
                 {'tipo': 'baixar', 'ids_pecas': ['m3', 'm4', 'm5']}]
        for semente in range(8):
            escolhida = criar('conta_rara', random.Random(semente))(jogo, acoes)
            self.assertEqual(escolhida['ids_pecas'], ['m0', 'm1', 'm2'])

    def test_nao_troca_tamanho_por_potencial(self):
        mao = pecas([('azul', 5), ('azul', 6), ('azul', 7), ('verde', 9), ('verde', 10), ('verde', 11)], 'm')
        jogo = partida(mao, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['m0', 'm1']}, {'tipo': 'baixar', 'ids_pecas': ['m3', 'm4', 'm5']}]
        self.assertEqual(criar('conta_rara', random.Random(0))(jogo, acoes)['ids_pecas'], ['m3', 'm4', 'm5'])

    def test_compra_quando_nao_ha_jogada(self):
        jogo = partida(pecas([('azul', 5)], 'm'), abriu=True)
        compra = {'tipo': 'comprar', 'ids_pecas': []}
        self.assertEqual(criar('conta_rara', random.Random(0))(jogo, [compra]), compra)

    def test_joga_partida_completa_conservando_pecas(self):
        from src.estrategias.arena import jogar
        resultado = jogar([criar('conta_rara', random.Random(1)), criar('max_pecas', random.Random(2))], semente=3)
        self.assertEqual(resultado['pecas_total'], 106)
        self.assertGreater(resultado['decisoes'], 0)


def sequencia(cor, numeros, prefixo='s'):
    return pecas([(cor, n) for n in numeros], prefixo)


class TestExtensoes(unittest.TestCase):
    def test_sequencia_estende_pelas_pontas(self):
        jogo = partida([], [sequencia('azul', [5, 6, 7])])
        self.assertEqual(extensoes_da_mesa(jogo), {('azul', 4), ('azul', 8)})

    def test_grupo_de_3_aceita_a_cor_que_falta_e_de_4_nao(self):
        jogo = partida([], [grupo(7)])
        self.assertEqual(extensoes_da_mesa(jogo), {('amarelo', 7)})
        cheio = pecas([(c, 7) for c in ('azul', 'verde', 'vermelho', 'amarelo')])
        self.assertEqual(extensoes_da_mesa(partida([], [cheio])), set())

    def test_sequencia_com_coringa_e_ignorada(self):
        com_coringa = pecas([('azul', 5), ('coringa', 0), ('azul', 7)])
        self.assertEqual(extensoes_da_mesa(partida([], [com_coringa])), set())

    def test_sequencia_nas_bordas_do_baralho(self):
        jogo = partida([], [sequencia('verde', [1, 2, 3])])
        self.assertEqual(extensoes_da_mesa(jogo), {('verde', 4)})


class TestDesbloqueios(unittest.TestCase):
    def mao_de_3(self):
        return pecas([('azul', 3), ('azul', 3), ('verde', 3), ('verde', 3), ('verde', 9)], 'm')

    def test_gastar_peca_de_numero_que_eu_travo_desbloqueia(self):
        jogo = partida(self.mao_de_3())
        self.assertEqual(desbloqueios_por_numero(jogo, ['m0']), 1)

    def test_gastar_outro_numero_nao_desbloqueia(self):
        jogo = partida(self.mao_de_3())
        self.assertEqual(desbloqueios_por_numero(jogo, ['m4']), 0)

    def test_sem_travar_nada_nao_ha_desbloqueio(self):
        jogo = partida(pecas([('azul', 3), ('verde', 3)], 'm'))
        self.assertEqual(desbloqueios_por_numero(jogo, ['m0']), 0)


class TestTravantesGastos(unittest.TestCase):
    def cenario(self):
        mesa = [sequencia('azul', [5, 6, 7]), pecas([('azul', 8), ('verde', 8), ('vermelho', 8)], 'g')]
        return partida(pecas([('azul', 8), ('verde', 1), ('verde', 2), ('amarelo', 4), ('amarelo', 5)], 'm'), mesa)

    def test_so_conta_tipo_sem_copia_escondida_que_estende_a_mesa(self):
        jogo = self.cenario()
        esc, ext = contagem_escondidas(jogo), extensoes_da_mesa(jogo)
        self.assertEqual(esc[('azul', 8)], 0)
        self.assertEqual(travantes_gastos(jogo, ['m0'], esc, ext), 1)
        self.assertEqual(travantes_gastos(jogo, ['m1', 'm2'], esc, ext), 0)


class TestDefensivas(unittest.TestCase):
    def test_numero_evita_desbloquear_mesmo_baixando_menos(self):
        mao = pecas([('azul', 3), ('azul', 3), ('verde', 3), ('verde', 3), ('verde', 9), ('verde', 10), ('verde', 11)], 'm')
        jogo = partida(mao, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['m0', 'm1', 'm2', 'm3']},
                 {'tipo': 'baixar', 'ids_pecas': ['m4', 'm5', 'm6']}]
        self.assertEqual(criar('defensiva_numero', random.Random(0))(jogo, acoes)['ids_pecas'], ['m4', 'm5', 'm6'])
        self.assertEqual(criar('max_pecas', random.Random(0))(jogo, acoes)['ids_pecas'], ['m0', 'm1', 'm2', 'm3'])

    def test_numero_gasta_se_a_jogada_esvazia_a_mao(self):
        mao = pecas([('azul', 3), ('azul', 3), ('verde', 3), ('verde', 3)], 'm')
        jogo = partida(mao, abriu=True)
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['m0', 'm1', 'm2', 'm3']}, {'tipo': 'baixar', 'ids_pecas': ['m0']}]
        self.assertEqual(criar('defensiva_numero', random.Random(0))(jogo, acoes)['ids_pecas'], ['m0', 'm1', 'm2', 'm3'])

    def test_copia_evita_gastar_o_tipo_que_so_eu_tenho(self):
        jogo = TestTravantesGastos().cenario()
        jogo.jogador_atual.abriu = True
        acoes = [{'tipo': 'baixar', 'ids_pecas': ['m0', 'm1', 'm2']}, {'tipo': 'baixar', 'ids_pecas': ['m3', 'm4']}]
        self.assertEqual(criar('defensiva_copia', random.Random(0))(jogo, acoes)['ids_pecas'], ['m3', 'm4'])
        self.assertEqual(criar('max_pecas', random.Random(0))(jogo, acoes)['ids_pecas'], ['m0', 'm1', 'm2'])

    def test_compram_quando_nao_ha_jogada(self):
        jogo = partida(pecas([('azul', 5)], 'm'), abriu=True)
        compra = {'tipo': 'comprar', 'ids_pecas': []}
        for nome in ('defensiva_numero', 'defensiva_copia'):
            self.assertEqual(criar(nome, random.Random(0))(jogo, [compra]), compra)


class TestAtributosDeContagem(unittest.TestCase):
    """Atributos 20–23 da rede: raridade gasta/retida, travantes gastos, extensões travadas por mim."""

    def setUp(self):
        import src.estrategias.rede as rede
        self.rede = rede
        mesa = [sequencia('azul', [5, 6, 7]), pecas([('azul', 8), ('verde', 8), ('vermelho', 8)], 'g')]
        self.jogo = partida(pecas([('azul', 8), ('verde', 1), ('verde', 2), ('amarelo', 4), ('amarelo', 5)], 'm'),
                            mesa, abriu=True)

    def vetor(self, ids, tipo='baixar'):
        return self.rede.contagem_atributos(self.jogo, {'tipo': tipo, 'ids_pecas': ids})

    def test_quatro_atributos_novos_no_fim_do_vetor(self):
        self.assertEqual(self.rede.ENTRADAS, 23)
        self.assertEqual(len(self.rede.atributos(self.jogo, {'tipo': 'comprar', 'ids_pecas': []})), 23)

    def test_comprar_nao_gasta_nada(self):
        gasta, retida, travantes, travaveis = self.vetor([], 'comprar')
        self.assertEqual((gasta, travantes), (0.0, 0.0))
        self.assertGreater(retida, 0.0)
        self.assertEqual(travaveis, 1 / 6)  # só azul 8 trava a mesa

    def test_gastar_o_travante_marca_fracao_total(self):
        _, _, travantes, _ = self.vetor(['m0'])
        self.assertEqual(travantes, 1.0)
        _, _, travantes, _ = self.vetor(['m1'])
        self.assertEqual(travantes, 0.0)

    def test_raridade_gasta_e_a_media_das_pecas_gastas(self):
        esc = contagem_escondidas(self.jogo)
        esperado = raridade(self.jogo, esc, ('azul', 8))
        self.assertEqual(self.vetor(['m0'])[0], esperado)

    def test_valores_ficam_entre_0_e_1(self):
        jogo = Jogo(semente=11)
        for a in jogo.listar_acoes_validas()[:20]:
            self.assertTrue(all(0.0 <= x <= 1.0 for x in self.rede.atributos(jogo, a)[19:]))
