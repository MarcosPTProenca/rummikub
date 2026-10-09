import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.estrategias.rede import ENTRADAS, PoliticaRede, Rede, atributos, retorno_formado
from tests.test_estrategias import grupo, partida, pecas


def rede_aleatoria(semente=0, ocultos=5):
    return Rede.nova(random.Random(semente), ocultos)


def frente_em_python_puro(rede, X):
    """Referência independente (laços e math.tanh) para conferir a versão com numpy."""
    f, h, t = ENTRADAS, rede.ocultos, rede.theta
    w1 = [t[i * f:(i + 1) * f] for i in range(h)]
    b1, w2, b2 = t[h * f:h * f + h], t[h * f + h:h * f + 2 * h], t[-1]
    ocultas = [[math.tanh(b + sum(w * x for w, x in zip(linha, v))) for linha, b in zip(w1, b1)] for v in X]
    return [b2 + sum(a * o for a, o in zip(w2, oc)) for oc in ocultas]


class TestNumpy(unittest.TestCase):
    def test_pontuacoes_equivalem_a_referencia_em_python_puro(self):
        rng = random.Random(5)
        rede = rede_aleatoria(2, ocultos=7)
        X = [[rng.uniform(-2, 2) for _ in range(ENTRADAS)] for _ in range(9)]
        for obtido, esperado in zip(rede.pontuacoes(X), frente_em_python_puro(rede, X)):
            self.assertAlmostEqual(obtido, esperado, places=12)

    def test_passo_adam_equivale_a_formula_em_python_puro(self):
        rng = random.Random(6)
        rede = rede_aleatoria(3, ocultos=3)
        theta, m, v = list(rede.theta), [0.0] * len(rede.theta), [0.0] * len(rede.theta)
        for t in range(1, 4):
            grad = [rng.uniform(-1, 1) for _ in theta]
            rede.passo(grad, t, lr=0.01)
            for i, g in enumerate(grad):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                theta[i] += 0.01 * (m[i] / (1 - 0.9 ** t)) / (math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        for obtido, esperado in zip(rede.theta, theta):
            self.assertAlmostEqual(obtido, esperado, places=12)


class TestAtributos(unittest.TestCase):
    def setUp(self):
        self.jogo = partida(grupo(7) + pecas([('verde', 1), ('coringa', 0)], 'z'), abriu=True)
        self.acoes = self.jogo.listar_acoes_validas()

    def test_um_vetor_por_acao_com_tamanho_fixo(self):
        for acao in self.acoes:
            self.assertEqual(len(atributos(self.jogo, acao)), ENTRADAS)

    def test_distingue_comprar_de_baixar_e_marca_o_que_esvazia_a_mao(self):
        comprar = next(a for a in self.acoes if a['tipo'] == 'comprar')
        baixar = max((a for a in self.acoes if a['tipo'] != 'comprar'), key=lambda a: len(a['ids_pecas']))
        fc, fb = atributos(self.jogo, comprar), atributos(self.jogo, baixar)
        self.assertEqual((fc[0], fc[2]), (1.0, 0.0))
        self.assertEqual(fb[0], 0.0)
        self.assertGreater(fb[2], 0.0)

    def test_nao_usa_a_mao_do_oponente_so_o_tamanho(self):
        antes = atributos(self.jogo, self.acoes[0])
        self.jogo.jogadores[1 - self.jogo.indice_jogador_atual].mao.reverse()
        self.assertEqual(atributos(self.jogo, self.acoes[0]), antes)


class TestGradiente(unittest.TestCase):
    def objetivo(self, rede, X, escolhida, vantagem, beta):
        p = rede.probabilidades(X)
        import math
        entropia = -sum(q * math.log(q) for q in p if q > 0)
        return vantagem * math.log(p[escolhida]) + beta * entropia

    def test_gradiente_analitico_confere_com_diferencas_finitas(self):
        rng = random.Random(3)
        rede = rede_aleatoria(1, ocultos=4)
        X = [[rng.uniform(-1, 1) for _ in range(ENTRADAS)] for _ in range(5)]
        grad = rede.gradiente(X, 2, vantagem=1.3, beta=0.05)
        self.assertEqual(len(grad), len(rede.theta))
        for i in rng.sample(range(len(rede.theta)), 12):
            h = 1e-6
            base = rede.theta[i]
            rede.theta[i] = base + h
            mais = self.objetivo(rede, X, 2, 1.3, 0.05)
            rede.theta[i] = base - h
            menos = self.objetivo(rede, X, 2, 1.3, 0.05)
            rede.theta[i] = base
            self.assertAlmostEqual(grad[i], (mais - menos) / (2 * h), places=5)

    def test_reforco_aprende_um_bandit_sintetico(self):
        rng = random.Random(0)
        rede = rede_aleatoria(2, ocultos=6)
        decisoes = []
        for _ in range(30):
            X = [[rng.uniform(0, 1) for _ in range(ENTRADAS)] for _ in range(4)]
            decisoes.append((X, max(range(4), key=lambda k: X[k][2])))
        def acerto():
            return sum(rede.probabilidades(X)[alvo] for X, alvo in decisoes) / len(decisoes)
        antes = acerto()
        for passo in range(400):
            grad = [0.0] * len(rede.theta)
            for X, alvo in decisoes:
                p = rede.probabilidades(X)
                escolhida = rng.choices(range(4), p)[0]
                g = rede.gradiente(X, escolhida, vantagem=1.0 if escolhida == alvo else -1.0, beta=0.0)
                grad = [a + b for a, b in zip(grad, g)]
            rede.passo(grad, passo + 1, lr=0.02)
        self.assertGreater(acerto(), max(0.8, antes + 0.3))


class TestPolitica(unittest.TestCase):
    def setUp(self):
        self.jogo = partida(grupo(7) + pecas([('verde', 1)], 'z'), abriu=True)
        self.acoes = self.jogo.listar_acoes_validas()

    def test_devolve_acao_legal_e_grava_a_trajetoria(self):
        politica = PoliticaRede(rede_aleatoria(), random.Random(0), gravar=True)
        escolhida = politica(self.jogo, self.acoes)
        self.assertTrue(any(escolhida is a for a in self.acoes))
        X, indice = politica.trajetoria[0]
        self.assertEqual(len(X), len(self.acoes))
        self.assertIs(self.acoes[indice], escolhida)

    def test_guloso_e_deterministico_e_amostrado_varia_com_a_semente(self):
        rede = rede_aleatoria(4)
        gulosas = {id(PoliticaRede(rede, random.Random(s), guloso=True)(self.jogo, self.acoes)) for s in range(5)}
        self.assertEqual(len(gulosas), 1)
        amostradas = {id(PoliticaRede(rede, random.Random(s))(self.jogo, self.acoes)) for s in range(40)}
        self.assertGreater(len(amostradas), 1)

    def test_serializa_e_recarrega_sem_perder_nada(self):
        rede = rede_aleatoria(5)
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'r.json'
            rede.salvar(arquivo, iteracao=7)
            recarregada = Rede.carregar(arquivo)
        self.assertEqual(recarregada.theta, rede.theta)
        self.assertEqual(recarregada.ocultos, rede.ocultos)


class TestTreinoPeloCLI(unittest.TestCase):
    def test_treina_grava_checkpoints_e_o_agente_joga_pelo_experimento(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '3', '--jogos', '4',
                            '--processos', '2', '--ocultos', '4', '--checkpoint-a-cada', '2', '--saida', str(pasta)],
                           check=True, capture_output=True)
            self.assertTrue((pasta / 'final.json').exists())
            self.assertTrue((pasta / 'iter_000002.json').exists())
            log = [json.loads(l) for l in (pasta / 'treino.jsonl').read_text().splitlines()]
            self.assertEqual([r['iteracao'] for r in log], [1, 2, 3])
            self.assertIn('vitorias', log[0])
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'busca', '--jogador', f'rede:{pasta / "final.json"}',
                            '--oponentes', 'max_pecas', '--sementes', '1', '--processos', '1',
                            '--saida', str(pasta / 'busca.jsonl')], check=True, capture_output=True)
            partidas = [json.loads(l) for l in (pasta / 'busca.jsonl').read_text().splitlines()]
            self.assertEqual(len(partidas), 2)
            self.assertTrue(all(p['a'].startswith('rede:') for p in partidas))


class TestForma(unittest.TestCase):
    def test_forma_zero_deixa_o_retorno_intacto(self):
        self.assertEqual(retorno_formado(1.0, 0, 0.0), 1.0)
        self.assertEqual(retorno_formado(-1.0, 9, 0.0), -1.0)

    def test_soma_a_queda_da_mao_em_fracao_das_14_pecas_iniciais(self):
        self.assertAlmostEqual(retorno_formado(1.0, 0, 0.5), 1.5)
        self.assertAlmostEqual(retorno_formado(-1.0, 7, 0.5), -0.75)

    def test_mao_que_cresceu_pune(self):
        self.assertLess(retorno_formado(-1.0, 20, 0.5), -1.0)

    def test_empate_sem_vencedor_so_recebe_a_forma(self):
        self.assertAlmostEqual(retorno_formado(0.0, 7, 0.5), 0.25)

    def test_cli_aceita_oponentes_extras_na_liga_de_treino_e_grava_no_log(self):
        with tempfile.TemporaryDirectory() as pasta:
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '2',
                            '--processos', '1', '--ocultos', '4', '--liga', 'flexivel', 'poupar_ate_2',
                            '--saida', pasta], check=True, capture_output=True)
            log = json.loads((Path(pasta) / 'treino.jsonl').read_text().splitlines()[0])
            self.assertEqual(log['liga'], ['flexivel', 'poupar_ate_2'])

    def test_cli_mesas_sorteia_o_tamanho_da_mesa_e_grava_no_log(self):
        with tempfile.TemporaryDirectory() as pasta:
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '6',
                            '--processos', '1', '--ocultos', '4', '--mesas', '3', '10', '--saida', pasta],
                           check=True, capture_output=True)
            log = json.loads((Path(pasta) / 'treino.jsonl').read_text().splitlines()[0])
            self.assertEqual(log['mesas'], [3, 10])
            self.assertEqual(set(log['tamanhos']), {3, 10})

    def test_cli_aceita_mesa_fixa_de_10_jogadores(self):
        with tempfile.TemporaryDirectory() as pasta:
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '2',
                            '--processos', '1', '--ocultos', '4', '--jogadores', '10', '--saida', pasta],
                           check=True, capture_output=True)
            log = json.loads((Path(pasta) / 'treino.jsonl').read_text().splitlines()[0])
            self.assertEqual(log['tamanhos'], [10, 10])

    def test_liga_com_nome_desconhecido_falha(self):
        with tempfile.TemporaryDirectory() as pasta:
            r = subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '2',
                                '--processos', '1', '--liga', 'inexistente', '--saida', pasta], capture_output=True)
            self.assertNotEqual(r.returncode, 0)

    def test_cli_grava_a_forma_no_log_de_treino(self):
        with tempfile.TemporaryDirectory() as pasta:
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '2',
                            '--processos', '1', '--ocultos', '4', '--forma', '0.5', '--saida', pasta],
                           check=True, capture_output=True)
            log = json.loads((Path(pasta) / 'treino.jsonl').read_text().splitlines()[0])
            self.assertEqual(log['forma'], 0.5)


class TestResidual(unittest.TestCase):
    def test_gradiente_com_deslocamento_confere_com_diferencas_finitas(self):
        import math
        rng = random.Random(4)
        rede = rede_aleatoria(2, ocultos=4)
        X = [[rng.uniform(-1, 1) for _ in range(ENTRADAS)] for _ in range(5)]
        desloc = [0.0, 3.0, 0.0, 0.0, 0.0]

        def objetivo():
            p = rede.probabilidades(X, desloc)
            return 1.3 * math.log(p[2])
        grad = rede.gradiente(X, 2, vantagem=1.3, deslocamento=desloc)
        for i in rng.sample(range(len(rede.theta)), 12):
            h, base = 1e-6, rede.theta[i]
            rede.theta[i] = base + h
            mais = objetivo()
            rede.theta[i] = base - h
            menos = objetivo()
            rede.theta[i] = base
            self.assertAlmostEqual(grad[i], (mais - menos) / (2 * h), places=5)

    def test_rede_zerada_com_base_joga_exatamente_como_o_saver_a_partida_inteira(self):
        from src.estrategias.arena import jogar
        from src.estrategias.politicas import ESTRATEGIAS, criar
        base = ESTRATEGIAS['poupar_coringa']
        rede = Rede.nova(random.Random(0), 4, zero=True)
        politica = PoliticaRede(rede, random.Random(0), guloso=True, base=base, peso_base=4.0)
        conferidas = []

        def espiao(jogo, acoes):
            escolhida = politica(jogo, acoes)
            conferidas.append(escolhida is base(jogo, acoes, random.Random(0)))
            return escolhida
        jogar([espiao, criar('max_pecas', random.Random(1))], semente=3)
        self.assertGreater(len(conferidas), 10)
        self.assertTrue(all(conferidas))

    def test_cli_residual_grava_no_checkpoint_e_o_agente_carregado_usa_a_base(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--iteracoes', '1', '--jogos', '2',
                            '--processos', '1', '--ocultos', '4', '--residual', '4', '--saida', str(pasta)],
                           check=True, capture_output=True)
            dados = json.loads((pasta / 'final.json').read_text())
            self.assertEqual(dados['residual'], 4.0)
            log = json.loads((pasta / 'treino.jsonl').read_text().splitlines()[0])
            self.assertEqual(log['residual'], 4.0)
            from src.experimentos import construir
            politica = construir(f'rede:{pasta / "final.json"}', 1, 0)
            self.assertIsNotNone(politica.base)
            self.assertEqual(politica.peso_base, 4.0)


class TestDMCePPO(unittest.TestCase):
    def test_gradiente_q_e_o_da_perda_quadratica_por_diferencas_finitas(self):
        rng = random.Random(6)
        rede = rede_aleatoria(3, ocultos=4)
        x = [rng.uniform(-1, 1) for _ in range(ENTRADAS)]
        alvo = 0.7
        grad = rede.gradiente_q(x, alvo)
        perda = lambda: -0.5 * (rede.pontuacoes([x])[0] - alvo) ** 2
        for i in rng.sample(range(len(rede.theta)), 12):
            h, base = 1e-6, rede.theta[i]
            rede.theta[i] = base + h
            mais = perda()
            rede.theta[i] = base - h
            menos = perda()
            rede.theta[i] = base
            self.assertAlmostEqual(grad[i], (mais - menos) / (2 * h), places=5)

    def test_epsilon_zero_e_guloso_e_epsilon_um_explora(self):
        jogo = partida(grupo(7) + pecas([('verde', 1), ('coringa', 0)], 'z'), abriu=True)
        acoes = jogo.listar_acoes_validas()
        rede = rede_aleatoria(1)
        gulosa = {id(PoliticaRede(rede, random.Random(s), epsilon=0.0)(jogo, acoes)) for s in range(20)}
        self.assertEqual(len(gulosa), 1)
        livre = {id(PoliticaRede(rede, random.Random(s), epsilon=1.0)(jogo, acoes)) for s in range(60)}
        self.assertGreater(len(livre), 2)

    def test_gradiente_com_kl_ao_modelo_de_referencia_confere_com_diferencas_finitas(self):
        import math
        rng = random.Random(8)
        rede, ref = rede_aleatoria(2, ocultos=4), rede_aleatoria(9, ocultos=4)
        X = [[rng.uniform(-1, 1) for _ in range(ENTRADAS)] for _ in range(5)]
        log_ref = [math.log(q) for q in ref.probabilidades(X)]

        def objetivo():
            p = rede.probabilidades(X)
            return -sum(q * (math.log(q) - lr) for q, lr in zip(p, log_ref))
        grad = rede.gradiente(X, 1, vantagem=0.0, kl=1.0, log_ref=log_ref)
        for i in rng.sample(range(len(rede.theta)), 12):
            h, base = 1e-6, rede.theta[i]
            rede.theta[i] = base + h
            mais = objetivo()
            rede.theta[i] = base - h
            menos = objetivo()
            rede.theta[i] = base
            self.assertAlmostEqual(grad[i], (mais - menos) / (2 * h), places=5)

    def test_ppo_zera_o_gradiente_fora_da_faixa_de_clip_e_escala_pela_razao_dentro(self):
        import math
        from src.estrategias.rede import gradiente_ppo
        rng = random.Random(10)
        rede = rede_aleatoria(4, ocultos=4)
        X = [[rng.uniform(-1, 1) for _ in range(ENTRADAS)] for _ in range(4)]
        p = rede.probabilidades(X)[2]
        dentro = gradiente_ppo(rede, X, 2, 1.0, math.log(p / 1.1), 0.2)
        esperado = [1.1 * g for g in rede.gradiente(X, 2, 1.0)]
        for a, b in zip(dentro, esperado):
            self.assertAlmostEqual(a, b, places=9)
        fora = gradiente_ppo(rede, X, 2, 1.0, math.log(p / 1.5), 0.2)
        self.assertTrue(all(g == 0.0 for g in fora))
        fora_neg = gradiente_ppo(rede, X, 2, -1.0, math.log(p / 0.5), 0.2)
        self.assertTrue(all(g == 0.0 for g in fora_neg))
        ainda_ativa = gradiente_ppo(rede, X, 2, -1.0, math.log(p / 1.5), 0.2)
        self.assertTrue(any(g != 0.0 for g in ainda_ativa))

    def test_cli_treina_dmc_e_ppo_e_grava_o_algoritmo_no_log(self):
        for algoritmo in ('dmc', 'ppo'):
            with tempfile.TemporaryDirectory() as pasta:
                subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--algoritmo', algoritmo,
                                '--iteracoes', '2', '--jogos', '3', '--processos', '1', '--ocultos', '4',
                                '--saida', pasta], check=True, capture_output=True)
                self.assertTrue((Path(pasta) / 'final.json').exists())
                log = [json.loads(l) for l in (Path(pasta) / 'treino.jsonl').read_text().splitlines()]
                self.assertEqual([r['iteracao'] for r in log], [1, 2])
                self.assertEqual(log[0]['algoritmo'], algoritmo)
                self.assertIn('perda_valor', log[0])


if __name__ == '__main__':
    unittest.main()
