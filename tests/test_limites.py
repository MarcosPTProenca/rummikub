import random
import unittest

from src.estrategias.arena import jogar
from src.estrategias.politicas import criar
from src.experimentos import PERFIS_LIMITES, partida as partida_torneio
from tests.test_turnos import grupo, partida, pecas


def cenario():
    mesa = [grupo(5, 'a'), grupo(6, 'b'), grupo(7, 'c')]
    mao = pecas([('amarelo', 5), ('amarelo', 6), ('amarelo', 7), ('azul', 9), ('verde', 9), ('vermelho', 9)])
    return partida(mao, mesa, abriu=True)


def planos(acoes):
    return [a for a in acoes if a['tipo'] == 'mesa']


class TestLimites(unittest.TestCase):
    def test_padrao_nao_muda_as_acoes(self):
        jogo = cenario()
        self.assertEqual(jogo.listar_acoes_validas(), jogo.listar_acoes_validas(limites={}))

    def test_limite_de_planos_corta_e_registra_saturacao(self):
        jogo = cenario()
        livre = planos(jogo.listar_acoes_validas())
        self.assertGreater(len(livre), 1)
        sat = {}
        cortado = planos(jogo.listar_acoes_validas(limites={'planos': 1}, saturacao=sat))
        self.assertEqual(len(cortado), 1)
        self.assertTrue(sat['planos'])

    def test_sem_corte_nao_registra_saturacao(self):
        sat = {}
        cenario().listar_acoes_validas(saturacao=sat)
        self.assertFalse(any(sat.values()))

    def test_limite_maior_inclui_as_acoes_do_padrao(self):
        jogo = cenario()
        base = jogo.listar_acoes_validas(limites={'planos': 3})
        mais = jogo.listar_acoes_validas(limites={'planos': 500})
        chave = lambda a: (a['tipo'], tuple(a['ids_pecas']), repr(a.get('combinacoes')))
        self.assertTrue({chave(a) for a in base} <= {chave(a) for a in mais})


class TestArena(unittest.TestCase):
    def rodar(self, **kw):
        return jogar([criar('max_pecas', random.Random(1)), criar('poupar_coringa', random.Random(2))],
                     semente=300_003, **kw)

    def test_resultado_traz_contagem_de_saturacao(self):
        r = self.rodar()
        self.assertEqual(set(r['saturacao']), {'planos', 'visitas', 'pecas', 'nos'})
        self.assertTrue(all(0 <= v <= r['decisoes'] for v in r['saturacao'].values()))

    def test_limite_baixo_satura_mais_e_conserva_pecas(self):
        base = self.rodar()
        baixo = self.rodar(limites={'planos': 1})
        self.assertNotEqual(baixo['saturacao'], base['saturacao'])
        self.assertEqual(baixo['saturacao']['nos'], 0)
        self.assertGreater(baixo['saturacao']['planos'], 0)
        self.assertEqual(baixo['pecas_total'], 106)


class TestPerfis(unittest.TestCase):
    def test_registro_traz_perfil_e_saturacao(self):
        r = partida_torneio('max_pecas', 'poupar_coringa', 150_000, 0, perfil='padrao')
        self.assertEqual(r['perfil'], 'padrao')
        self.assertEqual(set(r['saturacao']), {'planos', 'visitas', 'pecas', 'nos'})

    def test_perfil_padrao_nao_muda_o_resultado(self):
        sem = partida_torneio('max_pecas', 'poupar_coringa', 150_000, 0)
        com = partida_torneio('max_pecas', 'poupar_coringa', 150_000, 0, perfil='padrao')
        self.assertEqual(sem, com)

    def test_perfis_maiores_estao_definidos(self):
        self.assertEqual(PERFIS_LIMITES['padrao'], {})
        for nome in ('x4', 'x16'):
            self.assertGreater(PERFIS_LIMITES[nome]['planos'], 128)

    def test_perfil_desconhecido_falha(self):
        with self.assertRaises(KeyError):
            partida_torneio('max_pecas', 'max_pecas', 150_000, 0, perfil='inexistente')


if __name__ == '__main__':
    unittest.main()
