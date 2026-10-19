"""Plano 23 (continuação): reduzir a fração de self-play no treino.

A hipótese testada no plano 23 ("o treino não vê o adversário forte") caiu — incluir o campo
forte na liga não mudou o resultado. Metade dos oponentes de treino continua sendo a própria
rede (25%) ou um checkpoint antigo (25%), inalterado. Este parâmetro deixa essa fração
configurável, para testar se é o self-play, não o adversário, que domina o sinal de treino.

Garantia: o valor padrão (0.5) deve reproduzir byte a byte o comportamento anterior à mudança
(mesmo hash de pesos do `test_base_semente`, que já é a referência congelada do caminho
padrão).
"""
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.estrategias import rede

HASH_ANTERIOR = {
    15: "ba45d6acc17f2e64973fd1ea9ea2a3c6383ac81f4367321ac2a0fb23183a4959",
    23: "d25bc3b512992389e7080f22d71c046e2097c4c6f066b85260ecf6e1e706d2a6",
}


def _treinar(d, nome, iteracoes=3, **extra):
    return rede.treinar_rede(Path(d) / nome, iteracoes=iteracoes, jogos=8, processos=1, ocultos=4, semente=11,
                             liga=("max_pecas",), grupo=4, recompensa="pontos", **extra)


def _hash(r):
    return hashlib.sha256(json.dumps([round(x, 12) for x in r.theta]).encode()).hexdigest()


class TestFracaoSelfPlay(unittest.TestCase):
    def test_padrao_reproduz_o_hash_congelado(self):
        """fracao_self_play=0.5 (padrão) não pode mudar nada do que já foi medido."""
        esperado = HASH_ANTERIOR.get(rede.ENTRADAS)
        if esperado is None:
            self.skipTest(f"sem hash congelado para ENTRADAS={rede.ENTRADAS}")
        with TemporaryDirectory() as d:
            self.assertEqual(_hash(_treinar(d, "a")), esperado)

    def test_fracao_explicita_de_0_5_e_igual_ao_padrao(self):
        with TemporaryDirectory() as d:
            self.assertEqual(list(_treinar(d, "a").theta),
                             list(_treinar(d, "b", fracao_self_play=0.5).theta))

    def test_fracao_zero_nunca_usa_a_propria_rede_nem_checkpoints_antigos(self):
        """Com fracao_self_play=0, todo oponente vem da liga (heurísticas), nunca self-play."""
        from multiprocessing.pool import Pool
        with TemporaryDirectory() as d, mock.patch.object(Pool, "map") as mapa:
            mapa.side_effect = lambda fn, tarefas: [fn(t) for t in tarefas]
            _treinar(d, "a", fracao_self_play=0.0)
            vistos = [[isinstance(e, str) for e in t[2]] for chamada in mapa.call_args_list for t in chamada.args[1]]
            self.assertTrue(vistos)
            self.assertTrue(all(all(e) for e in vistos), "self-play apareceu com fracao_self_play=0")

    def test_fracao_um_nunca_usa_heuristica(self):
        """Com fracao_self_play=1, todo oponente é a própria rede ou checkpoint antigo (depois que o
        primeiro checkpoint antigo existe, na iteração 10; `jogos=8` gera 8 chamadas por iteração)."""
        from multiprocessing.pool import Pool
        with TemporaryDirectory() as d, mock.patch.object(Pool, "map") as mapa:
            mapa.side_effect = lambda fn, tarefas: [fn(t) for t in tarefas]
            _treinar(d, "a", fracao_self_play=1.0, iteracoes=15)
            vistos = [[isinstance(e, str) for e in t[2]] for chamada in mapa.call_args_list for t in chamada.args[1]]
            depois_do_checkpoint = vistos[8 * 10:]
            self.assertTrue(depois_do_checkpoint)
            self.assertTrue(all(not any(e) for e in depois_do_checkpoint),
                            "com fracao_self_play=1, esperava nunca usar heurística depois da iteração 10")

    def test_fracao_diferente_de_0_5_muda_o_treino_e_e_registrada(self):
        with TemporaryDirectory() as d:
            self.assertNotEqual(list(_treinar(d, "a").theta),
                                list(_treinar(d, "b", fracao_self_play=0.2).theta))
            r = json.loads((Path(d) / "b" / "treino.jsonl").read_text().splitlines()[0])
            self.assertEqual(r["fracao_self_play"], 0.2)

    def test_fracao_fora_de_0_1_e_rejeitada(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                _treinar(d, "a", fracao_self_play=1.5)


if __name__ == "__main__":
    unittest.main()
