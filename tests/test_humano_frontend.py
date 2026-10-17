"""A validação do frontend tem de concordar com o motor.

O HTML valida combinações em JavaScript só para dar retorno imediato (contorno
vermelho) antes de o jogador apertar Jogar. Se essa cópia divergir do motor, a
interface recusa mesa legal ou promete jogada que o servidor vai negar; as duas
falhas são invisíveis sem este teste. Comparamos as duas implementações em
casos aleatórios e dirigidos.
"""
import json
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.jogo.combinacao import Combinacao
from src.jogo.peca import Peca

HTML = Path(__file__).parent.parent / "src" / "humano" / "mesa.html"
CORES = ["azul", "vermelho", "verde", "amarelo"]


def extrair_validacao() -> str:
    """O trecho de validação e pontuação do HTML, como módulo ES."""
    texto = HTML.read_text()
    inicio = texto.index("// --- validação local")
    fim = texto.index("// Pontos que a abertura conta")
    return texto[inicio:fim] + "\nexport { combinacaoValida, pontosDaCombinacao };\n"


def casos_de_teste() -> list[list[dict]]:
    rng = random.Random(0)
    casos = []
    for _ in range(3000):
        pecas = []
        for k in range(rng.randint(1, 8)):
            if rng.random() < 0.25:
                pecas.append({"id": str(k), "cor": "coringa", "numero": 0})
            else:
                pecas.append({"id": str(k), "cor": rng.choice(CORES), "numero": rng.randint(1, 13)})
        casos.append(pecas)
    # Sequências e grupos legítimos, que o aleatório quase nunca produz.
    for cor in CORES:
        for inicio in range(1, 13):
            for tamanho in range(3, 8):
                if inicio + tamanho - 1 <= 13:
                    casos.append([{"id": str(i), "cor": cor, "numero": inicio + i}
                                  for i in range(tamanho)])
    for numero in range(1, 14):
        for tamanho in (3, 4, 5):
            casos.append([{"id": str(i), "cor": CORES[i % 4], "numero": numero}
                          for i in range(tamanho)])
    # Coringa cobrindo buraco, o caso que mais confunde as duas implementações.
    for cor in CORES:
        for inicio in range(1, 11):
            for salto in (2, 3):
                casos.append([{"id": "0", "cor": cor, "numero": inicio},
                              {"id": "1", "cor": "coringa", "numero": 0},
                              {"id": "2", "cor": cor, "numero": inicio + salto}])
    return casos


@unittest.skipIf(shutil.which("node") is None, "node não está instalado")
class TestValidacaoCruzada(unittest.TestCase):
    def test_js_e_motor_concordam_em_todos_os_casos(self):
        casos = casos_de_teste()
        with tempfile.TemporaryDirectory() as pasta:
            d = Path(pasta)
            (d / "valida.mjs").write_text(extrair_validacao())
            (d / "casos.json").write_text(json.dumps(casos))
            (d / "roda.mjs").write_text(
                'import { readFileSync } from "node:fs";\n'
                'import { combinacaoValida, pontosDaCombinacao } from "./valida.mjs";\n'
                'const casos = JSON.parse(readFileSync(new URL("./casos.json", import.meta.url), "utf8"));\n'
                "console.log(JSON.stringify(casos.map((c) => "
                "[combinacaoValida(c), combinacaoValida(c) ? pontosDaCombinacao(c) : 0])));\n")
            r = subprocess.run(["node", str(d / "roda.mjs")], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        do_js = json.loads(r.stdout)
        combinacoes = [Combinacao(pecas=[Peca(**p) for p in c]) for c in casos]
        do_motor = [[c.eh_valido(), c.calcula_pontos() if c.eh_valido() else 0] for c in combinacoes]
        divergencias = [(c, j, m) for c, j, m in zip(casos, do_js, do_motor) if j != m]
        self.assertEqual(divergencias[:5], [], f"{len(divergencias)} de {len(casos)} divergem")
        # Garante que os casos exercitam os dois resultados, senão o teste passa vazio.
        validos = sum(1 for v, _ in do_motor if v)
        self.assertGreater(validos, 100)
        self.assertGreater(len(do_motor) - validos, 100)


def extrair_ordenacao() -> str:
    """O trecho de ordenação de combinações, com a global `mesa` isolada."""
    texto = HTML.read_text()
    inicio = texto.index("const ORDEM_COR")
    fim = texto.index("async function nova()")
    return ("let mesa = [];\n" + extrair_validacao().replace(
        "export { combinacaoValida, pontosDaCombinacao };", "") + texto[inicio:fim]
        + "\nexport function rodar(entrada) { mesa = entrada; ordenarCombinacoes(); return mesa; }\n")


def combinacoes_embaralhadas() -> list[list[dict]]:
    """Combinações válidas fora de ordem: o caso real depois de arrastar peças."""
    rng = random.Random(7)
    casos = []
    for cor in CORES:
        for inicio in range(1, 12):
            for tamanho in (3, 4, 5):
                if inicio + tamanho - 1 <= 13:
                    c = [{"id": str(k), "cor": cor, "numero": inicio + k} for k in range(tamanho)]
                    rng.shuffle(c)
                    casos.append(c)
    for numero in range(1, 14):
        for tamanho in (3, 4):
            c = [{"id": str(k), "cor": CORES[k], "numero": numero} for k in range(tamanho)]
            rng.shuffle(c)
            casos.append(c)
    for cor in CORES:
        for inicio in range(1, 11):
            c = [{"id": "0", "cor": cor, "numero": inicio},
                 {"id": "1", "cor": "coringa", "numero": 0},
                 {"id": "2", "cor": cor, "numero": inicio + 2}]
            rng.shuffle(c)
            casos.append(c)
            c = [{"id": "0", "cor": cor, "numero": inicio},
                 {"id": "1", "cor": "coringa", "numero": 0},
                 {"id": "2", "cor": "coringa", "numero": 0},
                 {"id": "3", "cor": cor, "numero": inicio + 3}]
            rng.shuffle(c)
            casos.append(c)
    return casos


@unittest.skipIf(shutil.which("node") is None, "node não está instalado")
class TestOrdenacaoDeCombinacoes(unittest.TestCase):
    """Reordenar para leitura não pode mudar o que a mesa é para o motor."""

    def test_ordenar_preserva_as_pecas_e_a_validade(self):
        casos = combinacoes_embaralhadas()
        with tempfile.TemporaryDirectory() as pasta:
            d = Path(pasta)
            (d / "m.mjs").write_text(extrair_ordenacao())
            (d / "casos.json").write_text(json.dumps(casos))
            (d / "roda.mjs").write_text(
                'import { readFileSync } from "node:fs";\n'
                'import { rodar } from "./m.mjs";\n'
                'const casos = JSON.parse(readFileSync(new URL("./casos.json", import.meta.url), "utf8"));\n'
                "console.log(JSON.stringify(casos.map((c) => rodar([c.slice()])[0])));\n")
            r = subprocess.run(["node", str(d / "roda.mjs")], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        saidas = json.loads(r.stdout)
        self.assertEqual(len(saidas), len(casos))
        for antes, depois in zip(casos, saidas):
            with self.subTest(antes=antes):
                self.assertEqual(sorted(x["id"] for x in antes),
                                 sorted(x["id"] for x in depois), "peça perdida ou duplicada")
                antes_valida = Combinacao(pecas=[Peca(**x) for x in antes]).eh_valido()
                depois_valida = Combinacao(pecas=[Peca(**x) for x in depois]).eh_valido()
                self.assertEqual(antes_valida, depois_valida)


class TestFrontend(unittest.TestCase):
    @unittest.skipIf(shutil.which("node") is None, "node não está instalado")
    def test_script_da_pagina_tem_sintaxe_valida(self):
        """Chave sobrando no HTML só apareceria ao abrir a página no navegador."""
        texto = HTML.read_text()
        script = texto[texto.index('"use strict"'):texto.rindex("</script>")]
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / "pagina.mjs"
            arquivo.write_text(script)
            r = subprocess.run(["node", "--check", str(arquivo)],
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_fim_de_partida_nao_usa_dialog_modal(self):
        """showModal bloqueia a página inteira e o jogo parece travado no fim."""
        texto = HTML.read_text()
        self.assertNotIn("showModal", texto)
        self.assertNotIn("<dialog", texto)

    def test_drop_insere_na_posicao_do_ponteiro(self):
        """Com push a peça só anda para o fim; reordenar exige splice na posição."""
        texto = HTML.read_text()
        self.assertIn("posicaoDeInsercao", texto)
        self.assertIn("mao.splice(pos, 0", texto)
        self.assertIn("mesa[i].splice(pos, 0", texto)

    def test_acoes_passam_pela_trava_de_uma_por_vez(self):
        """Dois cliques em Comprar compravam duas peças."""
        texto = HTML.read_text()
        self.assertIn("if (ocupado) return;", texto)

    def test_ordem_da_mao_e_preservada_entre_turnos(self):
        """A mão do motor vem em outra ordem e desfazia o Ordenar a cada compra."""
        texto = HTML.read_text()
        self.assertIn("maoPreservandoOrdem", texto)
        self.assertNotIn("mao = servidor.mao.slice()", texto)
        self.assertNotIn("mao = estado.mao.slice()", texto)

    def test_combinacoes_sao_reordenadas_ao_fechar_o_turno(self):
        texto = HTML.read_text()
        self.assertIn("ordenarCombinacoes()", texto)

    def test_replay_tem_pausas_perceptiveis(self):
        """Sem pausa o lance é instantâneo e o jogador não vê o que mudou."""
        texto = HTML.read_text()
        self.assertIn("const RITMO", texto)
        for chave in ("pensar", "porPeca", "comprar", "entrePecas"):
            self.assertIn(chave, texto)

    def test_historico_mostra_ordem_hora_e_o_mais_recente_primeiro(self):
        texto = HTML.read_text()
        self.assertIn("l.hora", texto)
        self.assertIn("l.n", texto)
        self.assertIn("servidor.lances.slice().reverse()", texto)

    def test_cronometro_global_da_partida(self):
        texto = HTML.read_text()
        self.assertIn("ajustarCronometro", texto)
        self.assertIn('id="relogio"', texto)

    def test_configuracao_de_mesa_permite_varios_bots(self):
        """A mesa aceitava 9 oponentes no motor, mas a interface trancava em um."""
        texto = HTML.read_text()
        self.assertIn('id="lista-bots"', texto)
        self.assertIn('id="adicionar-bot"', texto)
        self.assertIn("escolhidos", texto)
        self.assertIn("oponentes: escolhidos", texto)
        self.assertNotIn('$("oponente").value', texto)

    def test_ritmo_comprime_com_a_mesa_cheia(self):
        """Nove bots a ritmo de um seriam 17 s por rodada."""
        texto = HTML.read_text()
        self.assertIn("function fator()", texto)
        self.assertIn("RITMO.pensar * f", texto)

    def test_html_nao_tem_atribuicao_duplicada_do_botao_jogar(self):
        """Uma linha de rascunho duplicada já escondeu a regra real de habilitar o botão."""
        texto = HTML.read_text()
        self.assertEqual(len(re.findall(r'\$\("jogar"\)\.disabled =', texto)), 1)

    def test_html_consome_as_rotas_que_o_servidor_expoe(self):
        texto = HTML.read_text()
        for rota in ("/estado", "/jogar", "/comprar", "/oponentes", "/nova"):
            self.assertIn(f'"{rota}"', texto)
        self.assertNotIn("/dica", texto)


if __name__ == "__main__":
    unittest.main()


def extrair_api() -> str:
    texto = HTML.read_text()
    inicio = texto.index("const SEM_SERVIDOR")  # a constante da mensagem vem logo antes da função
    fim = texto.index("function aviso(")
    return texto[inicio:fim] + "\nexport { api };\n"


@unittest.skipIf(shutil.which("node") is None, "node não está instalado")
class TestFalhaDeRede(unittest.TestCase):
    """O navegador diz só 'Failed to fetch' quando o servidor não responde, e isso não ajuda quem joga.

    A causa mais comum é o servidor ter sido encerrado (terminal fechado, reinício do WSL) com a aba
    ainda aberta. A mensagem tem de dizer isso e o que fazer, e o erro do servidor (HTTP 400 com a
    razão do motor) tem de continuar passando como estava.
    """

    def rodar(self, fetch_js: str) -> dict:
        with tempfile.TemporaryDirectory() as pasta:
            d = Path(pasta)
            (d / "m.mjs").write_text(extrair_api())
            (d / "roda.mjs").write_text(
                f"globalThis.fetch = {fetch_js};\n"
                'import { api } from "./m.mjs";\n'
                "try { const r = await api('/estado'); console.log(JSON.stringify({ok: r})); }\n"
                "catch (e) { console.log(JSON.stringify({erro: e.message})); }\n")
            r = subprocess.run(["node", str(d / "roda.mjs")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_servidor_fora_do_ar_diz_o_que_fazer(self):
        saida = self.rodar("async () => { throw new TypeError('Failed to fetch'); }")
        msg = saida["erro"]
        self.assertNotEqual(msg, "Failed to fetch")
        self.assertIn("servidor", msg.lower())
        self.assertIn("python -m src.humano", msg)
        self.assertIn("recarregue", msg.lower())

    def test_erro_do_motor_continua_chegando_ao_jogador(self):
        saida = self.rodar("async () => ({ ok: false, json: async () => ({ erro: 'Jogada inválida.' }) })")
        self.assertEqual(saida["erro"], "Jogada inválida.")

    def test_resposta_boa_continua_funcionando(self):
        saida = self.rodar("async () => ({ ok: true, json: async () => ({ vez: 0 }) })")
        self.assertEqual(saida["ok"], {"vez": 0})

    def test_resposta_que_nao_e_json_nao_vira_failed_to_fetch(self):
        saida = self.rodar("async () => ({ ok: true, json: async () => { throw new SyntaxError('Unexpected token <'); } })")
        self.assertIn("resposta", saida["erro"].lower())


def extrair_forca() -> str:
    """O trecho da página que decide rótulo e ordem do menu de oponentes."""
    texto = HTML.read_text()
    inicio = texto.index("// --- força dos oponentes")
    fim = texto.index("function desenharConfig(")
    return texto[inicio:fim] + "\nexport { tamanhoMedido, opcoes, rotulo };\n"


@unittest.skipIf(shutil.which("node") is None, "node não está instalado")
class TestForcaNoMenu(unittest.TestCase):
    """O menu mostra a força de cada oponente NO TAMANHO DA MESA que você monta, e ordena por ela."""

    CAT = [
        {"id": "fraco", "nome": "cautious", "forca": {"2": [0.063, 0.007], "3": [0.026, 0.004], "4": [0.021, 0.003]}},
        {"id": "forte", "nome": "saver-then-dump", "forca": {"2": [0.534, 0.011], "3": [0.360, 0.008], "4": [0.257, 0.006]}},
        {"id": "rl", "nome": "points-net", "rede": True,
         "forca": {"2": [0.469, 0.015], "3": [0.320, 0.011], "4": [0.249, 0.009]}},
    ]

    def rodar(self, escolhidos: list[str], selecionado=None) -> dict:
        with tempfile.TemporaryDirectory() as pasta:
            d = Path(pasta)
            (d / "m.mjs").write_text(
                "let catalogo = [];\nlet escolhidos = [];\n"
                "export function definir(c, e) { catalogo = c; escolhidos = e; }\n" + extrair_forca())
            (d / "roda.mjs").write_text(
                'import { definir, tamanhoMedido, opcoes } from "./m.mjs";\n'
                f"definir({json.dumps(self.CAT)}, {json.dumps(escolhidos)});\n"
                f"console.log(JSON.stringify({{n: tamanhoMedido(), html: opcoes({json.dumps(selecionado)})}}));\n")
            r = subprocess.run(["node", str(d / "roda.mjs")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_rotulo_mostra_a_taxa_e_o_acaso_do_tamanho_da_mesa(self):
        dois = self.rodar(["forte"])  # 1 bot + você = 2 jogadores
        self.assertEqual(dois["n"], 2)
        self.assertIn("saver-then-dump — 53% (acaso 50%)", dois["html"])
        tres = self.rodar(["forte", "forte"])
        self.assertEqual(tres["n"], 3)
        self.assertIn("saver-then-dump — 36% (acaso 33%)", tres["html"])
        quatro = self.rodar(["forte"] * 3)
        self.assertIn("saver-then-dump — 26% (acaso 25%)", quatro["html"])

    def test_rede_e_marcada_como_rl(self):
        html = self.rodar(["forte"])["html"]
        self.assertIn("points-net (RL) — 47%", html)
        self.assertNotIn("saver-then-dump (RL)", html)

    def test_ordena_do_mais_forte_ao_mais_fraco_no_tamanho_da_mesa(self):
        html = self.rodar(["forte"])["html"]
        self.assertLess(html.index("saver-then-dump"), html.index("points-net"))
        self.assertLess(html.index("points-net"), html.index("cautious"))

    def test_mesa_maior_que_4_usa_a_medida_de_4_e_diz_isso(self):
        r = self.rodar(["forte"] * 6)  # 7 jogadores
        self.assertEqual(r["n"], 4)
        self.assertIn("(acaso 25%)", r["html"])

    def test_opcao_selecionada_continua_marcada(self):
        html = self.rodar(["forte"], "rl")["html"]
        self.assertIn('value="rl" selected', html)
        self.assertNotIn('value="forte" selected', html)
