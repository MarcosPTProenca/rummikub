import unittest

from src.coringas import resumir


class TestResumir(unittest.TestCase):
    def test_media_e_erro_por_agrupamento_de_sementes(self):
        obs = [(1, 'x', 1.0), (1, 'x', 0.0), (2, 'x', 1.0), (2, 'x', 1.0), (3, 'y', 0.5)]
        r = resumir(obs)
        self.assertEqual(r['x']['maos'], 4)
        self.assertEqual(r['x']['sementes'], 2)
        self.assertAlmostEqual(r['x']['pontuacao'], 0.75)
        self.assertGreater(r['x']['ic95'], 0)
        self.assertEqual(r['y']['pontuacao'], 0.5)


class TestEfeitoEstratificado(unittest.TestCase):
    def dados(self, ganho):
        obs = []
        for semente in range(200):
            for estrato in (0, 1):
                obs.append((semente, estrato, True, min(1.0, 0.5 + ganho)))
                obs.append((semente, estrato, False, 0.5))
        return obs

    def test_recupera_efeito_conhecido(self):
        from src.coringas import efeito_estratificado
        e = efeito_estratificado(self.dados(0.25), reps=50)
        self.assertAlmostEqual(e['efeito'], 0.25)
        self.assertEqual((e['tratados'], e['controles']), (400, 400))

    def test_sem_efeito_da_zero_e_ic_contem_zero(self):
        from src.coringas import efeito_estratificado
        e = efeito_estratificado(self.dados(0.0), reps=50)
        self.assertEqual(e['efeito'], 0.0)
        self.assertLessEqual(e['ic95'][0], 0.0)
        self.assertGreaterEqual(e['ic95'][1], 0.0)

    def test_estrato_sem_tratados_ou_sem_controles_e_ignorado(self):
        from src.coringas import efeito_estratificado
        obs = [(1, 0, True, 1.0), (2, 0, False, 0.0), (3, 1, True, 1.0)]
        self.assertAlmostEqual(efeito_estratificado(obs, reps=10)['efeito'], 1.0)


class TestAnaliseCampo(unittest.TestCase):
    def mao(self, semente, estrategia, ini, comprados, compras, ponto):
        return {'semente': semente, 'estrategia': estrategia, 'iniciais': ini, 'iniciais_oponente': 0,
                'comprados': comprados, 'compras': compras, 'ponto': ponto}

    def test_estratificar_por_compras_remove_a_confusao(self):
        from src.coringas import analisar_campo
        maos = [self.mao(s, 'x', 0, 1, 5, 0.6) for s in range(100)]
        maos += [self.mao(s + 1000, 'x', 0, 0, 5, 0.6) for s in range(300)]
        maos += [self.mao(s + 2000, 'x', 0, 1, 20, 0.2) for s in range(300)]
        maos += [self.mao(s + 3000, 'x', 0, 0, 20, 0.2) for s in range(100)]
        r = analisar_campo(maos, reps=30)
        self.assertEqual(r['maos_com_coringa_comprado'], 400)
        self.assertAlmostEqual(r['comprado_ingenuo']['efeito'], -0.2)
        self.assertAlmostEqual(r['comprado_estratificado']['efeito'], 0.0)
        self.assertIn('x', r['comprado_por_estrategia'])
