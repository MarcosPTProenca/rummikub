"""Cena Manim: lê um replay JSON (variável REPLAY) e anima a partida. Rodar com o python do .venv-animacao."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

from manim import (BOLD, AnimationGroup, Create, DashedLine, Dot, FadeIn, Line, ManimColor, Polygon, Rectangle,
                   RoundedRectangle, Scene, Text, VGroup, VMobject, config)

from src.animacao.layout import ALTURA, BASE_H, BASE_W, LARGURA, MONTE, layout
from src.animacao.rotulos import rotulo

config.frame_width, config.frame_height = LARGURA, ALTURA
config.background_color = ManimColor("#14213D")

COR = {"azul": "#1E6FD9", "vermelho": "#D7263D", "verde": "#1B9E4B", "amarelo": "#E09F00"}
FONTE = "DejaVu Sans"
DESTAQUE = "#00E5A0"


def peca(identificador, cor, numero):
    base = RoundedRectangle(corner_radius=0.08, width=BASE_W, height=BASE_H, fill_opacity=1,
                            fill_color="#F6EFD9", stroke_color="#8A7F66", stroke_width=2)
    if cor == "coringa":
        base.set_fill("#F4C430")
        texto = Text("J", font=FONTE, weight=BOLD, font_size=30, color="#5A3A00")
    else:
        texto = Text(str(numero), font=FONTE, weight=BOLD, font_size=30, color=COR[cor])
    if texto.width > BASE_W * 0.8:
        texto.scale_to_fit_width(BASE_W * 0.8)
    grupo = VGroup(base, texto)
    grupo.base = base
    return grupo


def rotulo_texto(conteudo, tamanho, cor="#FFFFFF", peso=None):
    t = Text(conteudo, font=FONTE, font_size=tamanho, color=cor, **({"weight": peso} if peso else {}))
    if t.width > LARGURA - 0.6:
        t.scale_to_fit_width(LARGURA - 0.6)
    return t


class Partida(Scene):
    def construct(self):
        r = json.loads(Path(os.environ["REPLAY"]).read_text())
        ritmo = r.get("ritmo", 1.0)
        baixo = r["assento_a"]
        nome_a, nome_b = rotulo(r["estrategia_a"]), rotulo(r["estrategia_b"])
        nomes = {baixo: nome_a, 1 - baixo: nome_b}

        titulo = rotulo_texto(r.get("titulo", nome_a), 40 if "titulo" in r else 46, peso=BOLD).move_to([0, 6.3, 0])
        sub = rotulo_texto(r.get("subtitulo", f"vs {nome_b}   ·   seed {r['semente']}"), 24, "#9DB4D8").move_to([0, 5.7, 0])
        self.add(titulo, sub)

        rotulo_cima = rotulo_texto(f"Hand: {nome_b}", 22, "#9DB4D8").move_to([0, 5.0, 0])
        rotulo_baixo = rotulo_texto(f"Hand: {nome_a}", 22, "#9DB4D8").move_to([0, -3.6, 0])
        rotulo_mesa = rotulo_texto("TABLE", 20, "#5C7099").move_to([0, 3.1, 0])
        self.add(rotulo_cima, rotulo_baixo, rotulo_mesa)

        monte = RoundedRectangle(corner_radius=0.05, width=BASE_W * 0.6, height=BASE_H * 0.6, fill_opacity=1,
                                 fill_color="#2B3A67", stroke_color="#5C7099", stroke_width=2).move_to([*MONTE, 0])
        self.add(monte)

        tiles, escala, destacadas = {}, {}, set()
        legenda = None
        placar = None

        def rodape(estado):
            nonlocal placar
            if placar:
                self.remove(placar)
            placar = rotulo_texto(f"Tiles left: {nome_a} {len(estado['maos'][baixo])}  ·  {nome_b} "
                                  f"{len(estado['maos'][1 - baixo])}  ·  Draw pile {estado['monte']}", 22,
                                  "#9DB4D8").move_to([0, -5.85, 0])
            self.add(placar)

        def aplicar(estado, tempo, novas=()):
            posicoes = layout(estado, baixo)["pecas"]
            todas = {p[0]: p for m in estado["maos"] for p in m} | {p[0]: p for c in estado["mesa"] for p in c}
            animacoes = []
            for i, (x, y, s) in posicoes.items():
                if i not in tiles:
                    t = peca(i, todas[i][1], todas[i][2]).move_to([*MONTE, 0])
                    tiles[i], escala[i] = t, 1.0
                    self.add(t)
                alvo = tiles[i]
                fator = s / escala[i]
                escala[i] = s
                animacoes.append(alvo.animate.scale(fator).move_to([x, y, 0]))
            if animacoes:
                self.play(AnimationGroup(*animacoes, lag_ratio=0), run_time=tempo)
            for i in list(tiles):
                if i not in posicoes:
                    self.remove(tiles.pop(i))

        def destacar(ids):
            for i in destacadas:
                if i in tiles:
                    tiles[i].base.set_stroke("#8A7F66", width=2)
            destacadas.clear()
            for i in ids:
                tiles[i].base.set_stroke(DESTAQUE, width=6)
                tiles[i].set_z_index(5)
                destacadas.add(i)

        anterior = r["inicio"]
        rodape(anterior)
        aplicar(anterior, 1.2 / ritmo)
        self.wait(0.6 / ritmo)
        for passo in r["passos"]:
            estado = passo["estado"]
            if legenda:
                self.remove(legenda)
            quem = nomes[passo["jogador"]]
            legenda = rotulo_texto(f"{quem}: {passo['texto']}", 26, "#FFFFFF" if passo["jogador"] == baixo else "#FFB703")
            if legenda.width > 8.6:
                legenda.scale_to_fit_width(8.6)
            legenda.move_to([-0.4, -6.4, 0])
            self.add(legenda)
            rodape(estado)
            mesa_antes = {p[0] for c in anterior["mesa"] for p in c}
            mesa_depois = {p[0] for c in estado["mesa"] for p in c}
            mao_antes = {p[0] for p in anterior["maos"][passo["jogador"]]}
            novas = [i for i in mesa_depois if i in mao_antes]
            aplicar(estado, (0.35 if passo["tipo"] == "comprar" else 0.75) / ritmo)
            destacar(novas)
            self.wait((0.15 if passo["tipo"] == "comprar" else 0.45) / ritmo)
            anterior = estado

        vencedor = r["vencedor"]
        if "fim" in r and r.get("cortado"):
            fim, detalhe = r["fim"]
        elif vencedor is None:
            fim = "Draw"
        else:
            fim = f"{nomes[vencedor]} wins"
        pontos = estado["pontos"]
        if not (r.get("cortado") and "fim" in r):
            detalhe = ("Out of tiles" if r["motivo"] == "mao_vazia" else
                       f"Draw pile empty - fewer points left wins ({pontos[baixo]} vs {pontos[1 - baixo]})")
        painel = Rectangle(width=LARGURA, height=1.85, fill_color="#000000", fill_opacity=0.9, stroke_width=0)
        painel.move_to([0, -5.85, 0])
        fim_t = rotulo_texto(fim, 44, "#00E5A0", BOLD).move_to([0, -5.5, 0])
        det_t = rotulo_texto(detalhe, 22, "#FFFFFF").move_to([0, -6.2, 0])
        self.remove(legenda, placar)
        self.play(FadeIn(painel), FadeIn(fim_t), FadeIn(det_t), run_time=0.6 / ritmo)
        self.wait(2.0 / ritmo)


class Cartela(Scene):
    """Cartela de texto: linhas {t, s, c, b} ou {cols, s, c}; lê o JSON de CARTELA."""

    def construct(self):
        dados = json.loads(Path(os.environ["CARTELA"]).read_text())
        linhas = dados["linhas"]
        passo = 0.95
        y = (len(linhas) - 1) * passo / 2
        grupo = []
        for linha in linhas:
            if "cols" in linha:
                for texto, x in zip(linha["cols"], (-3.9, -1.3, 1.5, 3.9)):
                    t = rotulo_texto(texto, linha["s"], linha["c"])
                    t.scale_to_fit_width(min(t.width, 2.9))
                    grupo.append(t.move_to([x, y, 0]))
            elif linha["t"]:
                grupo.append(rotulo_texto(linha["t"], linha["s"], linha.get("c", "#FFFFFF"),
                                          BOLD if linha.get("b") else None).move_to([0, y, 0]))
            y -= passo
        self.play(*[FadeIn(t) for t in grupo], run_time=0.8)
        self.wait(dados["duracao"])


class Curva(Scene):
    """Curva de aprendizado: lê o JSON de CURVA (dados_curva); eixo x em escala log(1 + iteração)."""

    X0, X1, Y0, Y1 = -4.3, 4.6, -2.4, 4.2

    def construct(self):
        d = json.loads(Path(os.environ["CURVA"]).read_text())
        fim = math.log10(1 + d["x"][-1])
        px = lambda it: self.X0 + (self.X1 - self.X0) * math.log10(1 + it) / fim
        py = lambda v: self.Y0 + (self.Y1 - self.Y0) * v
        self.add(rotulo_texto("Learning curve", 44, peso=BOLD).move_to([0, 6.2, 0]),
                 rotulo_texto("Measured against Max tiles, 80-200 games per point", 24, "#9DB4D8").move_to([0, 5.5, 0]))
        eixos = VGroup(Line([self.X0, self.Y0, 0], [self.X1, self.Y0, 0], color="#5C7099"),
                       Line([self.X0, self.Y0, 0], [self.X0, self.Y1, 0], color="#5C7099"))
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            eixos.add(rotulo_texto(f"{v:.0%}", 20, "#9DB4D8").move_to([self.X0 - 0.55, py(v), 0]))
            if v:
                eixos.add(Line([self.X0, py(v), 0], [self.X1, py(v), 0], color="#2B3A67", stroke_width=1))
        for it in (0, 10, 100, 1000):
            eixos.add(rotulo_texto(str(it), 20, "#9DB4D8").move_to([px(it), self.Y0 - 0.4, 0]))
        eixos.add(rotulo_texto("Training iteration (log scale)", 22, "#9DB4D8").move_to([(self.X0 + self.X1) / 2, self.Y0 - 1.0, 0]))
        eixos.add(DashedLine([self.X0, py(0.5), 0], [self.X1, py(0.5), 0], color="#FFFFFF", stroke_width=2, dash_length=0.12))
        eixos.add(rotulo_texto("even with Max tiles", 18, "#FFFFFF").move_to([self.X0 + 1.2, py(0.5) + 0.25, 0]))
        self.play(FadeIn(eixos), run_time=0.6)
        for i, serie in enumerate(d["series"]):
            pontos = [[px(it), py(y), 0] for it, y in zip(d["x"], serie["y"])]
            linha = VMobject(color=serie["cor"], stroke_width=6 if i == 0 else 4).set_points_as_corners(pontos)
            animacoes = [Create(linha)]
            if serie["ic"]:
                cima = [[px(it), py(min(1, y + h)), 0] for it, y, h in zip(d["x"], serie["y"], serie["ic"])]
                baixo = [[px(it), py(max(0, y - h)), 0] for it, y, h in zip(d["x"], serie["y"], serie["ic"])]
                animacoes.append(FadeIn(Polygon(*cima, *reversed(baixo), color=serie["cor"], fill_opacity=0.18, stroke_width=0)))
            self.play(*animacoes, run_time=2.4 if i == 0 else 1.4)
            self.play(FadeIn(Dot([*pontos[-1][:2], 0], color=serie["cor"], radius=0.1)), run_time=0.2)
            legenda_y = -5.0 - 0.6 * i
            self.add(Line([-4.3, legenda_y, 0], [-3.8, legenda_y, 0], color=serie["cor"], stroke_width=6))
            texto = rotulo_texto(serie["rotulo"], 22, "#FFFFFF")
            self.add(texto.move_to([-3.5 + texto.width / 2, legenda_y, 0]))
        final = rotulo_texto(f"{d['series'][0]['y'][-1]:.0%} at iteration {d['x'][-1]}", 26, "#00E5A0", BOLD)
        self.add(final.move_to([self.X1 - final.width / 2, py(0.3), 0]))
        self.wait(2.5)
