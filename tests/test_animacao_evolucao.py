import random
import tempfile
import unittest
from pathlib import Path

from src.animacao.evolucao import (MARCOS_BUSCA, MARCO_FINAL, agregar_comportamento, arquivo_da_rede, cartela_fim,
                                   comando_concatenar, comportamento, cortar, dados_curva, escolher_semente, ilustra, legenda_do_marco,
                                   pontos_de_virada)
from src.estrategias.rede import Rede


class TestRedeInicial(unittest.TestCase):
    def test_marco_zero_e_a_rede_sorteada_antes_do_treino(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = arquivo_da_rede(0, Path('resultados/rede'), Path(pasta))
            self.assertEqual(Rede.carregar(arquivo).theta, Rede.nova(random.Random(0), 16).theta)

    def test_marcos_seguintes_usam_o_checkpoint_gravado(self):
        with tempfile.TemporaryDirectory() as pasta:
            self.assertEqual(arquivo_da_rede(300, Path('resultados/rede'), Path(pasta)),
                             Path('resultados/rede/iter_000300.json'))

    def test_busca_comeca_sem_treino_e_termina_antes_do_marco_final(self):
        self.assertEqual(MARCOS_BUSCA[0], 0)
        self.assertEqual(list(MARCOS_BUSCA), sorted(MARCOS_BUSCA))
        self.assertLess(MARCOS_BUSCA[-1], MARCO_FINAL)

    def test_snapshots_da_busca_estao_gravados(self):
        for marco in MARCOS_BUSCA[1:]:
            self.assertTrue(arquivo_da_rede(marco, Path('resultados/rede'), Path('/nao-usado')).exists(), marco)


class TestCortar(unittest.TestCase):
    def replay(self):
        return {'passos': [{'i': i} for i in range(50)], 'vencedor': 1, 'motivo': 'mao_vazia'}

    def test_corta_nos_primeiros_passos_sem_mexer_no_original(self):
        original = self.replay()
        cortado = cortar(original, 30, 'Title', 'Sub', ['a', 'b'])
        self.assertEqual(len(cortado['passos']), 30)
        self.assertEqual(len(original['passos']), 50)
        self.assertEqual((cortado['titulo'], cortado['subtitulo'], cortado['fim']), ('Title', 'Sub', ['a', 'b']))
        self.assertEqual(cortado['cortado'], 30)

    def test_partida_mais_curta_que_o_corte_fica_inteira(self):
        cortado = cortar({'passos': [{'i': 0}], 'vencedor': 0, 'motivo': 'mao_vazia'}, 30, 'T', 'S', ['a', 'b'])
        self.assertEqual(len(cortado['passos']), 1)
        self.assertNotIn('cortado', cortado)


def resumo(vitorias=0.5, compra=0.1, mesa=0.5):
    return {'vitorias': vitorias, 'partidas': 200, 'ic': 0.07, 'compra_com_jogada': compra, 'mesa': mesa}


class TestLegenda(unittest.TestCase):
    def test_cada_etapa_diz_o_que_mudou_com_a_taxa_medida(self):
        for tag in ('inicio', 'mesa', 'joga', 'plato'):
            titulo, sub, fim = legenda_do_marco(25, tag, resumo(0.456), 12)
            self.assertIn('25', titulo)
            self.assertIn('46%', fim[0])
            self.assertIn('200', fim[1])
            self.assertTrue(all(x.isascii() for x in (titulo, sub, *fim)))

    def test_etapas_mostram_o_numero_que_sustenta_a_frase(self):
        self.assertIn('75%', legenda_do_marco(0, 'inicio', resumo(compra=0.75), 12)[1])
        self.assertIn('32%', legenda_do_marco(15, 'mesa', resumo(mesa=0.32), 12)[1])
        self.assertIn('Never', legenda_do_marco(25, 'joga', resumo(compra=0.0), 12)[1])
        self.assertIn('4%', legenda_do_marco(25, 'joga', resumo(compra=0.04), 12)[1])

    def test_inicio_diz_que_nao_treinou(self):
        self.assertIn('untrained', legenda_do_marco(0, 'inicio', resumo(), 12)[0].lower())


class TestConcatenar(unittest.TestCase):
    def test_usa_o_demuxer_concat_sem_reencodar(self):
        c = comando_concatenar(Path('lista.txt'), Path('saida.mp4'))
        self.assertIn('concat', c)
        self.assertIn('copy', c)
        self.assertEqual(c[-1], 'saida.mp4')


class TestCartelaFinal(unittest.TestCase):
    def etapas(self, final, ic=0.07):
        return [(0, 'inicio', resumo(0.01)), (25, 'joga', resumo(0.47)), (1500, 'plato', {**resumo(final), 'ic': ic})]

    def textos(self, etapas):
        return ' '.join(l.get('t', '') for l in cartela_fim(etapas)['linhas'])

    def test_empate_estatistico_com_o_oponente_e_dito_como_equilibrio(self):
        self.assertIn('about even', self.textos(self.etapas(0.49)))

    def test_vencer_e_perder_com_certeza_sao_ditos_assim(self):
        self.assertIn('beats', self.textos(self.etapas(0.7, 0.06)))
        self.assertIn('still loses', self.textos(self.etapas(0.3, 0.06)))

    def test_uma_linha_por_etapa_na_tabela(self):
        cols = [l for l in cartela_fim(self.etapas(0.5))['linhas'] if 'cols' in l]
        self.assertEqual(len(cols), 3 + 1)


class TestComportamento(unittest.TestCase):
    def test_quem_maximiza_pecas_nunca_compra_tendo_jogada(self):
        c = comportamento('max_pecas', 'poupar_coringa', 110_000, 0)
        self.assertEqual(c['comprou_com_jogada'], 0)
        self.assertGreater(c['jogadas'], 0)
        self.assertGreaterEqual(c['vez_com_jogada'], c['jogadas'])

    def test_aleatorio_compra_mesmo_tendo_jogada(self):
        total = sum(comportamento('aleatorio', 'max_pecas', 110_000 + s, 0)['comprou_com_jogada'] for s in range(5))
        self.assertGreater(total, 0)

    def test_registra_quando_abriu_ou_none(self):
        c = comportamento('max_pecas', 'max_pecas', 110_000, 0)
        self.assertTrue(c['turno_abertura'] is None or c['turno_abertura'] >= 1)


class TestAgregarComportamento(unittest.TestCase):
    def test_taxas_somando_contagens(self):
        a = {'vez_com_jogada': 10, 'comprou_com_jogada': 5, 'jogadas': 5, 'pecas': 15, 'mesa': 1, 'turno_abertura': 4, 'vitoria': 1.0}
        b = {'vez_com_jogada': 10, 'comprou_com_jogada': 0, 'jogadas': 10, 'pecas': 20, 'mesa': 3, 'turno_abertura': None, 'vitoria': 0.0}
        r = agregar_comportamento([a, b])
        self.assertAlmostEqual(r['compra_com_jogada'], 0.25)
        self.assertAlmostEqual(r['pecas_por_jogada'], 35 / 15)
        self.assertAlmostEqual(r['mesa'], 4 / 15)
        self.assertAlmostEqual(r['abriu'], 0.5)
        self.assertAlmostEqual(r['turno_abertura'], 4.0)
        self.assertAlmostEqual(r['vitorias'], 0.5)
        self.assertGreater(r['ic'], 0)


def amostra(compra, mesa, vitorias=0.3):
    return {'compra_com_jogada': compra, 'mesa': mesa, 'vitorias': vitorias, 'partidas': 80}


class TestPontosDeVirada(unittest.TestCase):
    serie = {0: amostra(0.75, 0.01, 0.01), 10: amostra(0.74, 0.10), 15: amostra(0.72, 0.32),
             20: amostra(0.19, 0.85), 25: amostra(0.0, 0.88, 0.47), 40: amostra(0.0, 0.92, 0.54)}

    def test_primeiro_marco_em_que_cada_aprendizado_aparece(self):
        r = pontos_de_virada(self.serie, final=1500)
        self.assertEqual([m for m, _ in r], [0, 15, 25, 1500])
        self.assertEqual([t for _, t in r], ['inicio', 'mesa', 'joga', 'plato'])

    def test_aprendizado_que_nunca_aparece_nao_vira_marco(self):
        sem_mesa = {m: amostra(a['compra_com_jogada'], 0.0) for m, a in self.serie.items()}
        self.assertEqual([t for _, t in pontos_de_virada(sem_mesa, final=1500)], ['inicio', 'joga', 'plato'])

    def test_marcos_em_ordem_crescente(self):
        r = [m for m, _ in pontos_de_virada(self.serie, final=1500)]
        self.assertEqual(r, sorted(r))


def passo(tipo, jogador=0, tinha=None):
    return {'tipo': tipo, 'jogador': jogador, 'tinha_jogada': tipo != 'comprar' if tinha is None else tinha}


def replay_de(*tipos):
    return {'passos': [passo(t) for t in tipos]}


COM_JOGADA = replay_de('baixar', 'baixar', 'mesa')
COMPRA_SEM_PRECISAR = {'passos': [passo('baixar'), passo('comprar', tinha=True), passo('comprar', tinha=False)]}


class TestIlustra(unittest.TestCase):
    def test_aceita_quando_cada_clipe_mostra_o_comportamento_da_etapa(self):
        replays = {'inicio': COMPRA_SEM_PRECISAR, 'mesa': replay_de('baixar', 'mesa', 'comprar'),
                   'joga': COM_JOGADA, 'plato': replay_de('baixar', 'baixar', 'baixar', 'mesa')}
        self.assertTrue(ilustra(replays))

    def test_inicio_precisa_comprar_tendo_jogada(self):
        sem_jogada = {'passos': [passo('baixar'), passo('comprar', tinha=False), passo('comprar', tinha=False)]}
        self.assertFalse(ilustra({'inicio': sem_jogada}))
        self.assertTrue(ilustra({'inicio': COMPRA_SEM_PRECISAR}))

    def test_treinado_nao_compra_tendo_jogada_e_joga_ao_menos_duas_vezes(self):
        self.assertFalse(ilustra({'joga': COMPRA_SEM_PRECISAR}))
        self.assertFalse(ilustra({'joga': replay_de('baixar', 'comprar')}))
        sem_jogada_disponivel = {'passos': [passo('baixar'), passo('baixar'), passo('comprar', tinha=False)]}
        self.assertTrue(ilustra({'joga': sem_jogada_disponivel}))

    def test_recusa_etapa_mesa_sem_jogada_na_mesa(self):
        self.assertFalse(ilustra({'mesa': replay_de('baixar', 'baixar', 'comprar')}))

    def test_so_conta_os_passos_da_rede(self):
        r = {'passos': [passo('comprar', 1, tinha=True)] * 3 + [passo('baixar', 0)]}
        self.assertFalse(ilustra({'inicio': r}))


class TestEscolherSemente(unittest.TestCase):
    def test_semente_escolhida_ilustra_as_etapas_com_a_rede_real(self):
        arquivos = {m: arquivo_da_rede(m, Path('resultados/rede'), Path('/tmp')) for m in (15, 25, 1500)}
        with tempfile.TemporaryDirectory() as pasta:
            arquivos[0] = arquivo_da_rede(0, Path('resultados/rede'), Path(pasta))
            etapas = [(0, 'inicio'), (15, 'mesa'), (25, 'joga'), (1500, 'plato')]
            self.assertEqual(escolher_semente(etapas, arquivos, passos=12, inicio=110_036, limite=1), 110_036)
            with self.assertRaises(RuntimeError):
                escolher_semente(etapas, arquivos, passos=12, inicio=110_000, limite=1)


if __name__ == '__main__':
    unittest.main()


class TestRitmo(unittest.TestCase):
    def test_clipe_leva_o_fator_de_ritmo(self):
        r = {'passos': [{'i': 0}], 'vencedor': 0, 'motivo': 'mao_vazia'}
        self.assertEqual(cortar(r, 5, 'T', 'S', ['a', 'b'], ritmo=2.5)['ritmo'], 2.5)
        self.assertEqual(cortar(r, 5, 'T', 'S', ['a', 'b'])['ritmo'], 1.0)


class TestDadosCurva(unittest.TestCase):
    serie = {10: {'partidas': 80, 'vitorias': 0.15, 'compra_com_jogada': 0.7, 'mesa': 0.1},
             0: {'partidas': 80, 'vitorias': 0.01, 'compra_com_jogada': 0.75, 'mesa': 0.01}}
    medidas = {10: {'partidas': 200, 'vitorias': 0.2, 'compra_com_jogada': 0.6, 'mesa': 0.2},
               1500: {'partidas': 200, 'vitorias': 0.485, 'compra_com_jogada': 0.0, 'mesa': 0.87}}

    def test_junta_serie_e_medidas_em_ordem_e_prefere_a_medida_maior(self):
        d = dados_curva(self.serie, self.medidas)
        self.assertEqual(d['x'], [0, 10, 1500])
        self.assertEqual([s['nome'] for s in d['series']], ['win', 'draw', 'table'])
        self.assertEqual(d['series'][0]['y'], [0.01, 0.2, 0.485])
        self.assertEqual(d['series'][1]['y'], [0.75, 0.6, 0.0])
        self.assertEqual(d['series'][2]['y'], [0.01, 0.2, 0.87])

    def test_so_a_taxa_de_vitoria_tem_intervalo_e_ele_tem_a_forma_de_wilson(self):
        d = dados_curva({5: {'partidas': 100, 'vitorias': 0.5, 'compra_com_jogada': 0, 'mesa': 0}}, {})
        self.assertAlmostEqual(d['series'][0]['ic'][0], 0.0962, places=3)
        self.assertIsNone(d['series'][1]['ic'])
        self.assertEqual(len(dados_curva(self.serie, self.medidas)['series'][0]['ic']), 3)

    def test_texto_ascii(self):
        d = dados_curva(self.serie, self.medidas)
        self.assertTrue(all(s['rotulo'].isascii() for s in d['series']))
