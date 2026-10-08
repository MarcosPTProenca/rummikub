import random
import unittest

from src.jogo.combinacao import Combinacao
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from src.estrategias.pimc import PIMC, clonar, determinizar


def cenario():
    jogo = Jogo(semente=5)
    mao = [Peca(id=f'm{i}', cor=c, numero=n) for i, (c, n) in enumerate(
        [('azul', 10), ('azul', 11), ('azul', 12), ('verde', 10), ('vermelho', 10), ('amarelo', 5), ('verde', 6), ('coringa', 0)])]
    jogo.jogador_atual.mao = mao
    jogo.jogador_atual.abriu = True
    return jogo


def chave(jogo):
    return ([sorted(p.id for p in j.mao) for j in jogo.jogadores], sorted(p.id for p in jogo.monte),
            [sorted(p.id for p in c.pecas) for c in jogo.mesa], jogo.indice_jogador_atual)


class TestClone(unittest.TestCase):
    def test_clone_e_independente(self):
        jogo = cenario()
        antes = chave(jogo)
        copia = clonar(jogo)
        copia.comprar_peca()
        copia.jogador_atual.mao.clear()
        self.assertEqual(chave(jogo), antes)
        self.assertNotEqual(chave(copia), antes)


class TestDeterminizar(unittest.TestCase):
    def test_preserva_o_publico_e_so_embaralha_o_oculto(self):
        jogo = cenario()
        mundo = determinizar(jogo, random.Random(1))
        eu, outro = jogo.indice_jogador_atual, 1 - jogo.indice_jogador_atual
        self.assertEqual(mundo.jogadores[eu].mao, jogo.jogadores[eu].mao)
        self.assertEqual(len(mundo.jogadores[outro].mao), len(jogo.jogadores[outro].mao))
        self.assertEqual(len(mundo.monte), len(jogo.monte))
        oculto = lambda j: sorted(p.id for p in j.jogadores[outro].mao + j.monte)
        self.assertEqual(oculto(mundo), oculto(jogo))
        outros = {tuple(sorted(p.id for p in determinizar(jogo, random.Random(s)).jogadores[outro].mao)) for s in range(5)}
        self.assertGreater(len(outros), 1)
        todas = lambda j: sorted(p.id for x in j.jogadores for p in x.mao) + sorted(p.id for p in j.monte)
        self.assertEqual(sorted(todas(mundo)), sorted(todas(jogo)))
        self.assertEqual(len(set(todas(mundo))), len(todas(mundo)))


class TestPIMC(unittest.TestCase):
    def politica(self):
        return PIMC(random.Random(7), determinizacoes=2, candidatas=3)

    def test_devolve_uma_acao_legal(self):
        jogo = cenario()
        acoes = jogo.listar_acoes_validas()
        escolhida = self.politica()(jogo, acoes)
        self.assertTrue(any(escolhida is a for a in acoes))

    def test_nao_espia_a_mao_do_oponente(self):
        a, b = cenario(), cenario()
        outro = 1 - b.indice_jogador_atual
        oculta, topo = b.jogadores[outro].mao, b.monte
        oculta[:5], topo[:5] = topo[:5], oculta[:5]
        self.assertNotEqual(chave(a), chave(b))
        ea = self.politica()(a, a.listar_acoes_validas())
        eb = self.politica()(b, b.listar_acoes_validas())
        self.assertEqual(ea, eb)

    def test_trapaceiro_ve_o_mundo_verdadeiro(self):
        jogo = cenario()
        outro = 1 - jogo.indice_jogador_atual
        ids = lambda j: ([p.id for p in j.jogadores[outro].mao], [p.id for p in j.monte])
        mundo = PIMC(random.Random(7), 2, 3, ve_tudo=True).mundo(jogo, random.Random(1))
        self.assertEqual(ids(mundo), ids(jogo))
        self.assertIsNot(mundo, jogo)
        honesto = PIMC(random.Random(7), 2, 3).mundo(jogo, random.Random(1))
        self.assertNotEqual(ids(honesto), ids(jogo))

    def test_partida_completa_pelo_cli_termina(self):
        from src.experimentos import partida
        r = partida('pimc:1:2', 'poupar_coringa', 1001, 0)
        self.assertIn(r['pontos_a'], (0.0, 0.5, 1.0))
        self.assertEqual(r['a'], 'pimc:1:2')


class TestCLIBusca(unittest.TestCase):
    def test_busca_grava_jogos_pareados_com_a_regra(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as pasta:
            saida = Path(pasta) / 'busca.jsonl'
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'busca', '--jogador', 'pimc:1:2', '--oponentes',
                            'poupar_coringa', '--sementes', '1', '--processos', '1', '--saida', str(saida)],
                           cwd=raiz, capture_output=True, text=True, check=True, timeout=300)
            linhas = [json.loads(l) for l in saida.read_text().splitlines()]
            self.assertEqual(sorted(r['assento_a'] for r in linhas), [0, 1])
            self.assertEqual({(r['a'], r['b'], r['desempate']) for r in linhas}, {('pimc:1:2', 'poupar_coringa', 'pontos')})
            self.assertEqual({r['semente'] for r in linhas}, {100_000})

    def test_busca_aceita_semente_inicial_e_jogador_trapaceiro(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as pasta:
            saida = Path(pasta) / 'busca.jsonl'
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'busca', '--jogador', 'pimcv:1:2', '--oponentes',
                            'poupar_coringa', '--sementes', '1', '--semente-inicial', '176000', '--processos', '1',
                            '--saida', str(saida)], cwd=raiz, capture_output=True, text=True, check=True, timeout=300)
            linhas = [json.loads(l) for l in saida.read_text().splitlines()]
            self.assertEqual({(r['a'], r['semente']) for r in linhas}, {('pimcv:1:2', 176_000)})


if __name__ == '__main__':
    unittest.main()
