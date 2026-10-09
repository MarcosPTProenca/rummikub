import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.agentes.q_learning import AgenteQLearning

import random as _r
from src.estrategias.jev import JevMeta
from src.estrategias.jev_jogadas import JevJogadas, REGRAS, regras
from src.estrategias.jev import Orcamento
from src.estrategias.meta import estado_meta, treinar_meta
from src.estrategias.pimc import determinizar
from src.estrategias.politicas import oponentes
from src.estrategias.rede import atributos
from src.jogo.jogo import Jogo
from tests.test_turnos import pecas

COMPRAR = {'tipo': 'comprar', 'ids_pecas': []}


def mesa_n(n, maos_oponentes, abriram):
    jogo = Jogo(semente=7, quantidade_jogadores=n)
    for o, tamanho, abriu in zip(oponentes(jogo), maos_oponentes, abriram):
        o.mao = o.mao[:tamanho]
        o.abriu = abriu
    return jogo


class TestEntradasParaNJogadores(unittest.TestCase):
    def test_oponentes_exclui_o_jogador_atual(self):
        for n in (2, 3, 4):
            jogo = Jogo(semente=1, quantidade_jogadores=n)
            self.assertEqual(len(oponentes(jogo)), n - 1)
            self.assertNotIn(jogo.jogador_atual, oponentes(jogo))

    def test_estado_meta_usa_o_oponente_mais_proximo_de_sair_e_se_todos_abriram(self):
        jogo = mesa_n(3, [12, 3], [True, True])
        faixa_mao_oponente = estado_meta(jogo, [COMPRAR])[3]
        self.assertEqual(faixa_mao_oponente, 0)  # min(12, 3) = 3 peças: primeira faixa
        self.assertTrue(estado_meta(jogo, [COMPRAR])[1])
        jogo = mesa_n(3, [12, 3], [True, False])
        self.assertFalse(estado_meta(jogo, [COMPRAR])[1])

    def test_estado_com_dois_jogadores_nao_muda(self):
        jogo = Jogo(semente=3)
        jogo.jogadores[1].mao = jogo.jogadores[1].mao[:7]
        self.assertEqual(estado_meta(jogo, [COMPRAR])[3], 1)

    def test_atributos_da_rede_resumem_os_oponentes(self):
        jogo = mesa_n(4, [10, 4, 8], [True, False, False])
        x = atributos(jogo, COMPRAR)
        self.assertEqual(len(x), 19)
        self.assertAlmostEqual(x[10], 4 / 14)
        self.assertAlmostEqual(x[11], 1 / 3)
        self.assertAlmostEqual(x[15], 3 / 9)   # 3 oponentes
        self.assertAlmostEqual(x[16], 10 / 14)  # maior mão
        self.assertAlmostEqual(x[18], 0.0)      # nenhum com <= 1 peça

    def test_determinizar_preserva_tamanho_das_maos_e_o_total_de_pecas(self):
        jogo = mesa_n(4, [10, 4, 8], [False] * 3)
        antes = [len(o.mao) for o in oponentes(jogo)]
        mundo = determinizar(jogo, random.Random(0))
        self.assertEqual([len(o.mao) for o in oponentes(mundo)], antes)
        ids = [p.id for j in mundo.jogadores for p in j.mao] + [p.id for p in mundo.monte]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len([p.id for j in jogo.jogadores for p in j.mao] + [p.id for p in jogo.monte]))


class TestTreinoParaNJogadores(unittest.TestCase):
    def test_treinar_meta_numa_mesa_de_tres(self):
        agente = AgenteQLearning(gamma=1.0, semente=0)
        retornos = treinar_meta(agente, 3, ['max_pecas', 'aleatorio'], semente_inicial=5, jogadores=3)
        self.assertEqual(len(retornos), 3)
        self.assertTrue(all(r in (-1.0, 0.0, 1.0) for r in retornos))
        self.assertTrue(agente.tabela_q)

    def test_cli_treina_a_rede_para_tres_jogadores_e_ela_joga_a_mesa(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'rede', '--jogadores', '3', '--iteracoes', '2',
                            '--jogos', '3', '--processos', '1', '--ocultos', '4', '--saida', str(pasta)],
                           check=True, capture_output=True)
            self.assertTrue((pasta / 'final.json').exists())
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'mesa', '--jogadores', '3',
                            '--focos', f'rede:{pasta / "final.json"}', '--campos', 'iguais', '--sementes', '1',
                            '--processos', '1', '--saida', str(pasta / 'mesa.jsonl')],
                           check=True, capture_output=True)
            registros = [json.loads(l) for l in (pasta / 'mesa.jsonl').read_text().splitlines()]
            self.assertEqual(len(registros), 3)
            self.assertTrue(all(r['n'] == 3 for r in registros))

    def test_cli_treina_o_q_meta_para_tres_jogadores_e_avalia_na_mesa(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = Path(pasta) / 'treino.jsonl'
            subprocess.run([sys.executable, '-m', 'src.experimentos', 'treinar', '--jogadores', '3', '--replicas', '1',
                            '--episodios', '3', '--avaliacao', '1', '--processos', '1', '--saida', str(saida)],
                           check=True, capture_output=True)
            replica = json.loads(saida.read_text().splitlines()[0])
            self.assertEqual(replica['jogadores'], 3)
            self.assertTrue(replica['avaliacao'])
            self.assertTrue(all(r['n'] == 3 and r['foco'] == 'rl_meta' for r in replica['avaliacao']))


class TestJevParaNJogadores(unittest.TestCase):
    def estados(self, n):
        jogo = mesa_n(n, [10, 4, 8][:n - 1], [True, False, False][:n - 1])
        acoes = jogo.listar_acoes_validas()
        orc = Orcamento(1.0)
        meta = JevMeta(None, _r.Random(0), orc)._corpo(jogo, acoes)["state"]
        jogadas = JevJogadas(None, _r.Random(0), orc)._corpo(jogo, acoes)[0]["state"]
        return meta, jogadas

    def test_com_dois_jogadores_o_payload_nao_muda(self):
        for estado in self.estados(2):
            self.assertIn('opponent_tiles_in_hand', estado)
            self.assertNotIn('opponents', estado)
        self.assertIn('both players pass', regras(2))
        self.assertIn('2 players', self.estados(2)[0]['game'])

    def test_com_tres_ou_quatro_o_estado_lista_cada_oponente(self):
        for n in (3, 4):
            meta, jogadas = self.estados(n)
            for estado in (meta, jogadas):
                self.assertNotIn('opponent_tiles_in_hand', estado)
                self.assertEqual(len(estado['opponents']), n - 1)
                self.assertEqual(estado['opponents'][0], {'tiles_in_hand': 10, 'has_made_initial_meld': True})
            self.assertIn(f'{n} players', meta['game'])
            self.assertIn(f'{n} players', jogadas['rules'])
            self.assertIn('all players pass', jogadas['rules'])


if __name__ == '__main__':
    unittest.main()
