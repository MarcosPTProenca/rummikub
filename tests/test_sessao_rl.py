"""Plano 19: treinar para ganhar a sessão.

Quatro garantias, nesta ordem de importância:

1. O motor de sessão não mudou para quem não usa o recurso novo: os valores abaixo
   foram congelados ANTES da mudança, com o código do commit anterior.
2. A rede estendida com as entradas de sessão, com pesos zero nelas, decide
   exatamente como a rede original (é o que torna o ponto de partida do treino a
   `grande` e não uma rede nova).
3. A recompensa de sessão telescopa para vitória − 0,5: o objetivo é ganhar a
   sessão, não uma aproximação dele.
4. O treino é determinístico e T ≠ R (a transformação da recompensa muda algo).
"""
import json
import math
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.estrategias import arena, politicas, rede
from src.estrategias.sessao import sessao
from src.estrategias.treino_sessao import (
    SIGMA, recompensa_mao, treinar_sessao, valor_sessao)
from src.jogo.jogo import Jogo

# Congelado com o motor anterior à mudança (K=3).
SNAPSHOT = {
    ("max_pecas", "poupar_coringa", 3): {
        "pontos": [213.0, -213.0], "vencedor": 0, "maos_ganhas": [3, 0],
        "coringas": [0.0, 1.0], "decisoes": [89, 74, 91]},
    ("max_pecas", "poupar_coringa", 11): {
        "pontos": [4.0, -4.0], "vencedor": 0, "maos_ganhas": [1, 2],
        "coringas": [0.0, 0.0], "decisoes": [93, 127, 82]},
    ("poupar_descarta_perto", "sessao_adaptativa", 3): {
        "pontos": [213.0, -213.0], "vencedor": 0, "maos_ganhas": [3, 0],
        "coringas": [0.0, 1.0], "decisoes": [89, 74, 91]},
    ("poupar_descarta_perto", "sessao_adaptativa", 11): {
        "pontos": [5.0, -5.0], "vencedor": 0, "maos_ganhas": [2, 1],
        "coringas": [0.0, 0.0], "decisoes": [91, 112, 120]},
}


class TestMotorDeSessaoInalterado(unittest.TestCase):
    def test_resultados_identicos_aos_congelados(self):
        for (a, b, semente), esperado in SNAPSHOT.items():
            r = sessao([a, b], semente, maos=3)
            self.assertEqual(r["pontos"], esperado["pontos"], f"{a}|{b}|{semente}")
            self.assertEqual(r["vencedor"], esperado["vencedor"])
            self.assertEqual(r["maos_ganhas"], esperado["maos_ganhas"])
            self.assertEqual(r["coringas_mao"], esperado["coringas"])
            self.assertEqual([m["decisoes"] for m in r["maos"]], esperado["decisoes"])


class TestValorDeSessao(unittest.TestCase):
    def test_vazio_vale_meio_e_e_simetrico(self):
        for h in (1, 3, 10):
            self.assertAlmostEqual(valor_sessao(0.0, h), 0.5)
            for s in (-80.0, -7.0, 13.0, 200.0):
                self.assertAlmostEqual(valor_sessao(s, h) + valor_sessao(-s, h), 1.0)

    def test_monotono_no_saldo(self):
        valores = [valor_sessao(s, 5) for s in range(-200, 201, 20)]
        self.assertEqual(valores, sorted(valores))

    def test_degrau_quando_nao_restam_maos(self):
        self.assertEqual(valor_sessao(0.5, 0), 1.0)
        self.assertEqual(valor_sessao(-0.5, 0), 0.0)
        self.assertEqual(valor_sessao(0.0, 0), 0.5)

    def test_menos_maos_restantes_torna_o_saldo_mais_decisivo(self):
        self.assertGreater(valor_sessao(40.0, 1), valor_sessao(40.0, 10))
        self.assertLess(valor_sessao(-40.0, 1), valor_sessao(-40.0, 10))

    def test_sigma_declarado(self):
        self.assertEqual(SIGMA, 30.0)


class TestRecompensaTelescopa(unittest.TestCase):
    def test_soma_das_recompensas_e_vitoria_menos_meio(self):
        rng = random.Random(5)
        for _ in range(200):
            k = rng.randint(1, 10)
            pontos = [rng.choice([-1, 1]) * rng.randint(0, 90) for _ in range(k)]
            s, total = 0.0, 0.0
            for h, p in enumerate(pontos):
                total += recompensa_mao("sessao", s, p, k - h)
                s += p
            final = 1.0 if s > 0 else 0.5 if s == 0 else 0.0
            self.assertAlmostEqual(total, final - 0.5, places=12)

    def test_zero_soma_entre_os_dois_lados(self):
        rng = random.Random(8)
        s_me, s_op = 0.0, 0.0
        for h in range(6):
            p = rng.randint(-60, 60)
            me = recompensa_mao("sessao", s_me, p, 6 - h)
            op = recompensa_mao("sessao", s_op, -p, 6 - h)
            self.assertAlmostEqual(me + op, 0.0, places=12)
            s_me += p
            s_op -= p

    def test_pontos_e_o_proprio_ponto_normalizado(self):
        self.assertAlmostEqual(recompensa_mao("pontos", 123.0, 30.0, 4), 30.0 / rede.ESCALA_PONTOS)

    def test_modo_desconhecido_falha_cedo(self):
        with self.assertRaises(ValueError):
            recompensa_mao("inventada", 0.0, 1.0, 3)

    def test_vencer_uma_mao_decisiva_vale_mais_que_a_primeira(self):
        """A não linearidade que muda a política: o mesmo ponto vale mais no fim."""
        cedo = recompensa_mao("sessao", 0.0, 30.0, 10)
        tarde_equilibrado = recompensa_mao("sessao", 0.0, 30.0, 1)
        self.assertGreater(tarde_equilibrado, cedo)


class TestRedeSessao(unittest.TestCase):
    def _base(self):
        return rede.Rede.nova(random.Random(3), 16)

    def test_extras_sem_sessao_sao_neutros(self):
        base = self._base()
        sessao_rede = rede.RedeSessao.de_rede(base)
        jogo = Jogo(semente=1)
        self.assertEqual(base.extras(jogo), [])
        self.assertEqual(sessao_rede.extras(jogo), [0.0, 1.0])

    def test_extras_leem_saldo_e_maos_restantes(self):
        sessao_rede = rede.RedeSessao.de_rede(self._base())
        jogo = Jogo(semente=1)
        jogo.saldo_sessao = 100.0
        jogo.maos_restantes, jogo.maos_total = 4, 10
        self.assertAlmostEqual(sessao_rede.extras(jogo)[0], math.tanh(1.0))
        self.assertAlmostEqual(sessao_rede.extras(jogo)[1], 0.4)

    def test_pesos_zero_decidem_exatamente_como_a_original(self):
        """Ponto de partida do treino = `grande`: 1.000+ estados reais, zero divergências."""
        base = self._base()
        estendida = rede.RedeSessao.de_rede(base)
        comparados = 0
        for semente in range(6):
            jogo = Jogo(semente=semente)
            jogo.saldo_sessao, jogo.maos_restantes, jogo.maos_total = 55.0, 3, 10
            guia = politicas.criar("max_pecas", random.Random(semente))
            p0 = rede.PoliticaRede.da_rede(base, random.Random(1), guloso=True)
            p1 = rede.PoliticaRede.da_rede(estendida, random.Random(1), guloso=True)
            for _ in range(160):
                if jogo.terminou:
                    break
                acoes = jogo.listar_acoes_validas()
                self.assertIs(p0(jogo, acoes), p1(jogo, acoes))
                X0 = [rede.atributos(jogo, a) for a in acoes]
                X1 = [rede.atributos(jogo, a) + estendida.extras(jogo) for a in acoes]
                for a, b in zip(base.probabilidades(X0), estendida.probabilidades(X1)):
                    self.assertAlmostEqual(a, b, places=12)
                comparados += 1
                arena.executar(jogo, guia(jogo, acoes))
        self.assertGreater(comparados, 300)

    def test_peso_nas_entradas_novas_muda_a_decisao(self):
        """Se os pesos de sessão fossem ignorados, o treino não teria o que aprender."""
        estendida = rede.RedeSessao.de_rede(self._base())
        h, f = estendida.ocultos, estendida.entradas
        theta = list(estendida.theta)
        for j in range(h):
            theta[j * f + (f - 2)] = 3.0
            theta[j * f + (f - 1)] = -3.0
        mexida = rede.RedeSessao(theta, h, entradas=f)
        jogo = Jogo(semente=2)
        jogo.saldo_sessao, jogo.maos_restantes, jogo.maos_total = 150.0, 1, 10
        acoes = jogo.listar_acoes_validas()
        X = [rede.atributos(jogo, a) + mexida.extras(jogo) for a in acoes]
        Y = [rede.atributos(jogo, a) + estendida.extras(jogo) for a in acoes]
        self.assertNotEqual(mexida.pontuacoes(X), estendida.pontuacoes(Y))

    def test_salvar_e_carregar_preserva_a_marca_de_sessao(self):
        estendida = rede.RedeSessao.de_rede(self._base())
        with TemporaryDirectory() as d:
            caminho = Path(d) / "r.json"
            estendida.salvar(caminho)
            dados = json.loads(caminho.read_text())
            self.assertTrue(dados["sessao"])
            self.assertEqual(dados["entradas"], rede.ENTRADAS + 2)
            volta = rede.carregar_rede(caminho)
            self.assertIsInstance(volta, rede.RedeSessao)
            self.assertEqual(list(volta.theta), list(estendida.theta))

    def test_rede_comum_continua_carregando_como_rede_comum(self):
        base = self._base()
        with TemporaryDirectory() as d:
            caminho = Path(d) / "b.json"
            base.salvar(caminho)
            volta = rede.carregar_rede(caminho)
            self.assertNotIsInstance(volta, rede.RedeSessao)
            self.assertEqual(list(volta.theta), list(base.theta))

    def test_gradiente_tem_a_dimensao_da_rede_estendida(self):
        estendida = rede.RedeSessao.de_rede(self._base())
        jogo = Jogo(semente=4)
        jogo.saldo_sessao, jogo.maos_restantes, jogo.maos_total = 40.0, 5, 10
        acoes = jogo.listar_acoes_validas()
        X = [rede.atributos(jogo, a) + estendida.extras(jogo) for a in acoes]
        g = estendida.gradiente(X, 0, 1.0)
        self.assertEqual(len(g), len(estendida.theta))


class TestMotorAceitaRedes(unittest.TestCase):
    def test_nome_rede_joga_uma_sessao(self):
        base = rede.Rede.nova(random.Random(2), 8)
        with TemporaryDirectory() as d:
            caminho = Path(d) / "r.json"
            rede.RedeSessao.de_rede(base).salvar(caminho)
            r = sessao([f"rede:{caminho}", "max_pecas"], 12_345, maos=2)
            self.assertEqual(len(r["maos"]), 2)
            self.assertAlmostEqual(sum(r["pontos"]), 0.0)

    def test_rede_comum_tambem_entra_na_sessao(self):
        base = rede.Rede.nova(random.Random(2), 8)
        with TemporaryDirectory() as d:
            caminho = Path(d) / "b.json"
            base.salvar(caminho)
            r = sessao([f"rede:{caminho}", "max_pecas"], 12_346, maos=2)
            self.assertEqual(len(r["maos"]), 2)

    def test_motor_expoe_maos_restantes_a_politica(self):
        vistos = []

        def espiao(jogo, acoes):
            vistos.append((jogo.maos_restantes, jogo.maos_total))
            return politicas.criar("max_pecas", random.Random(0))(jogo, acoes)

        from src.estrategias import sessao as modulo
        modulo._jogar_mao(["max_pecas", "max_pecas"], 777, 0, [0.0, 0.0], None,
                          politicas=[espiao, espiao], maos_restantes=7, maos_total=10)
        self.assertTrue(vistos)
        self.assertTrue(all(v == (7, 10) for v in vistos))


class TestTreinoDeSessao(unittest.TestCase):
    def _treinar(self, d, nome, modo, semente=3):
        base = rede.Rede.nova(random.Random(9), 4)
        ini = Path(d) / "ini.json"
        rede.RedeSessao.de_rede(base).salvar(ini)
        return treinar_sessao(Path(d) / nome, iteracoes=2, sessoes=4, processos=1, inicial=ini,
                              maos=2, recompensa=modo, semente=semente, base_semente=8_000_000)

    def test_determinismo(self):
        with TemporaryDirectory() as d:
            a = self._treinar(d, "a", "sessao")
            b = self._treinar(d, "b", "sessao")
            self.assertEqual(list(a.theta), list(b.theta))

    def test_recompensa_de_sessao_difere_da_de_pontos(self):
        with TemporaryDirectory() as d:
            t = self._treinar(d, "t", "sessao")
            r = self._treinar(d, "r", "pontos")
            self.assertNotEqual(list(t.theta), list(r.theta),
                                "a transformação de recompensa não alterou o treino")

    def test_manifest_registra_o_que_o_plano_exige(self):
        with TemporaryDirectory() as d:
            self._treinar(d, "m", "sessao")
            linhas = (Path(d) / "m" / "treino.jsonl").read_text().splitlines()
            r = json.loads(linhas[0])
            for chave in ("recompensa", "maos", "sigma", "base_semente", "grupo", "iteracao"):
                self.assertIn(chave, r)
            self.assertEqual(r["recompensa"], "sessao")
            self.assertEqual(r["base_semente"], 8_000_000)
            self.assertTrue(json.loads((Path(d) / "m" / "final.json").read_text())["sessao"])

    def test_entradas_de_sessao_saem_do_zero_com_o_treino(self):
        with TemporaryDirectory() as d:
            t = self._treinar(d, "t", "sessao")
            f = t.entradas
            novas = [t.theta[j * f + c] for j in range(t.ocultos) for c in (f - 2, f - 1)]
            self.assertTrue(any(abs(x) > 0 for x in novas),
                            "o gradiente nunca chegou às entradas de sessão")

    def test_rede_sem_sessao_nao_e_aceita(self):
        with TemporaryDirectory() as d:
            base = rede.Rede.nova(random.Random(9), 4)
            ini = Path(d) / "ini.json"
            base.salvar(ini)
            with self.assertRaises(ValueError):
                treinar_sessao(Path(d) / "x", iteracoes=1, sessoes=4, processos=1, inicial=ini,
                               maos=2, recompensa="sessao", semente=1, base_semente=8_000_000)

    def test_recompensa_desconhecida_falha_cedo(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self._treinar(d, "z", "inventada")


if __name__ == "__main__":
    unittest.main()


class TestSubcomando(unittest.TestCase):
    def test_treino_sessao_estende_uma_rede_comum_e_treina(self):
        import sys
        from unittest import mock

        from src import experimentos

        with TemporaryDirectory() as d:
            base = rede.Rede.nova(random.Random(6), 4)
            ini = Path(d) / "comum.json"
            base.salvar(ini)
            saida = Path(d) / "saida"
            argv = ["experimentos", "treino-sessao", "--inicial", str(ini), "--recompensa", "sessao",
                    "--iteracoes", "1", "--sessoes", "4", "--maos", "2", "--processos", "1",
                    "--base-semente", "8000000", "--saida", str(saida)]
            with mock.patch.object(sys, "argv", argv):
                experimentos.main()
            final = json.loads((saida / "final.json").read_text())
            self.assertTrue(final["sessao"])
            self.assertEqual(final["entradas"], rede.ENTRADAS + 2)
            # o ponto de partida estendido fica guardado ao lado, com a origem registrada
            self.assertEqual(json.loads((saida / "inicial.json").read_text())["origem"], str(ini))
            manifest = json.loads((saida / "treino.jsonl").read_text().splitlines()[0])
            self.assertEqual(manifest["base_semente"], 8_000_000)
