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
