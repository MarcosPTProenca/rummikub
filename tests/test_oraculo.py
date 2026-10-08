import unittest

from src.oraculo import amostrar_estados, avaliar_estado
from tests.test_limites import cenario


class TestOraculo(unittest.TestCase):
    def test_avaliar_estado_devolve_maximo_de_pecas_por_perfil(self):
        r = avaliar_estado(cenario(), {'a': {'planos': 1}, 'b': {'planos': 500}})
        self.assertEqual(set(r), {'a', 'b'})
        self.assertGreaterEqual(r['b']['max_pecas'], r['a']['max_pecas'])
        self.assertGreater(r['b']['acoes'], r['a']['acoes'])
        self.assertTrue(r['a']['saturou'])
        self.assertGreaterEqual(r['a']['segundos'], 0)

    def test_estouro_de_tempo_e_registrado(self):
        r = avaliar_estado(cenario(), {'a': {}}, tempo_max=0)
        self.assertTrue(r['a']['estourou'])

    def test_amostrar_estados_so_pega_decisoes_saturadas_e_e_deterministico(self):
        partidas = [('max_pecas', 'poupar_coringa', 300_003, 0)]
        e1 = amostrar_estados(partidas, por_partida=3)
        e2 = amostrar_estados(partidas, por_partida=3)
        self.assertTrue(0 < len(e1) <= 3)
        self.assertEqual([e['semente'] for e in e1], [e['semente'] for e in e2])
        for e in e1:
            sat = {}
            e['jogo'].listar_acoes_validas(saturacao=sat)
            self.assertTrue(sat['planos'] or sat['nos'])


if __name__ == '__main__':
    unittest.main()
