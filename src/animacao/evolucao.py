"""Vídeo de evolução da rede: mesma partida (semente e oponente) em marcos do treino, mais a taxa medida em muitas partidas."""
import json
import math
import os
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

from src import experimentos
from src.animacao.pipeline import CENA, RAIZ, VENV, comando_manim
from src.animacao.replay import gravar
from src.estrategias.arena import jogar
from src.estrategias.politicas import medidas
from src.estrategias.rede import Rede

MARCOS_BUSCA = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 500, 800, 1100)
MARCO_FINAL = 1500
LIMITE_MESA = 0.2
LIMITE_COMPRA = 0.05
SEMENTE_VIDEO = 110_000
SEMENTE_MEDIDA = 115_000
OPONENTE = "max_pecas"
RITMO = 2.5


def arquivo_da_rede(marco: int, pasta: Path, temporaria: Path) -> Path:
    """Marco 0 não tem checkpoint: reconstrói a rede sorteada com a mesma semente do treino (0, 16 unidades)."""
    if marco == 0:
        arquivo = Path(temporaria) / "iter_000000.json"
        Rede.nova(random.Random(0), 16).salvar(arquivo, iteracao=0)
        return arquivo
    return Path(pasta) / f"iter_{marco:06d}.json"


def comportamento(nome: str, oponente: str, semente: int, assento: int) -> dict:
    """Contagens de uma partida do foco: compras com jogada disponível, peças por jogada, reorganizações, abertura."""
    c = {"vez_com_jogada": 0, "comprou_com_jogada": 0, "jogadas": 0, "pecas": 0, "mesa": 0, "turno_abertura": None}
    foco = experimentos.construir(nome, semente, assento)
    turnos = 0

    def observado(jogo, acoes):
        nonlocal turnos
        acao = foco(jogo, acoes)
        turnos += 1
        tem_jogada = any(a["tipo"] != "comprar" for a in acoes)
        c["vez_com_jogada"] += tem_jogada
        if acao["tipo"] == "comprar":
            c["comprou_com_jogada"] += tem_jogada
        else:
            c["jogadas"] += 1
            c["pecas"] += len(acao["ids_pecas"]) if acao["tipo"] == "baixar" else medidas(jogo, acao)[0]
            c["mesa"] += acao["tipo"] == "mesa"
            if c["turno_abertura"] is None:
                c["turno_abertura"] = turnos
        return acao

    politicas = [None, None]
    politicas[assento] = observado
    politicas[1 - assento] = experimentos.construir(oponente, semente, 1 - assento)
    resultado = jogar(politicas, semente)
    c["vitoria"] = experimentos.pontos(resultado["vencedor"], assento)
    return c


def _wilson(p: float, n: int) -> float:
    z2 = 1.96 ** 2
    return 1.96 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)


def agregar_comportamento(partidas: list[dict]) -> dict:
    soma = lambda k: sum(p[k] for p in partidas)
    aberturas = [p["turno_abertura"] for p in partidas if p["turno_abertura"] is not None]
    n = len(partidas)
    vitorias = soma("vitoria") / n
    return {"partidas": n, "vitorias": vitorias, "ic": _wilson(vitorias, n),
            "compra_com_jogada": soma("comprou_com_jogada") / soma("vez_com_jogada") if soma("vez_com_jogada") else 0.0,
            "pecas_por_jogada": soma("pecas") / soma("jogadas") if soma("jogadas") else 0.0,
            "mesa": soma("mesa") / soma("jogadas") if soma("jogadas") else 0.0,
            "abriu": len(aberturas) / len(partidas),
            "turno_abertura": sum(aberturas) / len(aberturas) if aberturas else None}


def _comportamento(tarefa):
    return comportamento(*tarefa)


def medir_comportamento(arquivo: Path, sementes: int, processos: int = 2) -> dict:
    tarefas = [(f"rede:{arquivo}", OPONENTE, SEMENTE_MEDIDA + s, a) for s in range(sementes) for a in (0, 1)]
    with Pool(processos) as pool:
        return agregar_comportamento(pool.map(_comportamento, tarefas))


def _da_rede(replay: dict) -> list[dict]:
    return [p for p in replay["passos"] if p["jogador"] == 0]


def ilustra(replays: dict[str, dict]) -> bool:
    """O clipe de cada etapa mostra o que a medição diz dela: comprar tendo jogada, jogar na mesa, não comprar tendo jogada."""
    for tag, replay in replays.items():
        passos = _da_rede(replay)
        compras_evitaveis = sum(p["tipo"] == "comprar" and p["tinha_jogada"] for p in passos)
        jogadas = sum(p["tipo"] != "comprar" for p in passos)
        if tag == "inicio" and compras_evitaveis < 1:
            return False
        if tag in ("joga", "plato") and (jogadas < 2 or compras_evitaveis > 0):
            return False
        if tag == "mesa" and not any(p["tipo"] == "mesa" for p in passos):
            return False
    return True


def escolher_semente(etapas: list[tuple[int, str]], arquivos: dict, passos: int, inicio: int = SEMENTE_VIDEO,
                     limite: int = 500) -> int:
    for semente in range(inicio, inicio + limite):
        if all(ilustra({tag: cortar(gravar(f"rede:{arquivos[marco]}", OPONENTE, semente, 0, "pontos"), passos, "", "", [])})
               for marco, tag in etapas):
            return semente
    raise RuntimeError(f"nenhuma semente em {limite} ilustra as etapas")


def cortar(replay: dict, passos: int, titulo: str, subtitulo: str, fim: list[str], ritmo: float = 1.0) -> dict:
    novo = {**replay, "passos": replay["passos"][:passos], "titulo": titulo, "subtitulo": subtitulo, "fim": fim,
            "ritmo": ritmo}
    if len(replay["passos"]) > passos:
        novo["cortado"] = passos
    return novo


def pontos_de_virada(serie: dict[int, dict], final: int) -> list[tuple[int, str]]:
    """Primeiro marco em que cada aprendizado aparece: usar a mesa, jogar sempre que pode; mais o início e o fim."""
    marcos = sorted(serie)
    etapas = [(marcos[0], "inicio")]
    mesa = next((m for m in marcos if serie[m]["mesa"] >= LIMITE_MESA), None)
    joga = next((m for m in marcos if serie[m]["compra_com_jogada"] <= LIMITE_COMPRA), None)
    etapas += sorted((m, t) for m, t in ((mesa, "mesa"), (joga, "joga")) if m is not None)
    return etapas + [(final, "plato")]


def dados_curva(serie: dict[int, dict], medidas: dict[int, dict]) -> dict:
    """Pontos da curva de aprendizado: a medição com mais partidas vale em cada iteração."""
    pontos = {**serie, **medidas}
    x = sorted(pontos)
    por_marco = [pontos[m] for m in x]
    return {"x": x, "series": [
        {"nome": "win", "rotulo": "Win rate vs Max tiles", "cor": "#00E5A0",
         "y": [r["vitorias"] for r in por_marco], "ic": [_wilson(r["vitorias"], r["partidas"]) for r in por_marco]},
        {"nome": "draw", "rotulo": "Draws although a play exists", "cor": "#FFB703",
         "y": [r["compra_com_jogada"] for r in por_marco], "ic": None},
        {"nome": "table", "rotulo": "Plays that use the table", "cor": "#9DB4D8",
         "y": [r["mesa"] for r in por_marco], "ic": None}]}


def legenda_do_marco(marco: int, tag: str, resumo: dict, passos: int) -> tuple[str, str, list[str]]:
    compra, mesa = resumo["compra_com_jogada"], resumo["mesa"]
    titulo, sub = {
        "inicio": (f"Iteration {marco}: untrained",
                   f"Draws a tile on {compra:.0%} of turns where a play was possible"),
        "mesa": (f"Iteration {marco}: starts using the table",
                 f"{mesa:.0%} of its plays add to or rearrange sets on the table"),
        "joga": (f"Iteration {marco}: plays whenever it can",
                 "Never draws when a play exists" if compra == 0 else f"Draws only {compra:.0%} of the time when a play exists"),
        "plato": (f"Iteration {marco}: nothing new",
                  f"Draws {compra:.0%} of the time with a play; table plays {mesa:.0%}"),
    }[tag]
    fim = [f"Wins {resumo['vitorias']:.0%} vs Max tiles",
           f"{resumo['partidas']} games, 95% CI +/-{resumo['ic'] * 100:.0f} pts  -  clip: first {passos} steps"]
    return titulo, sub, fim


def comando_concatenar(lista: Path, saida: Path) -> list[str]:
    return [str(VENV / "ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista),
            "-c", "copy", str(saida)]


def cartela_inicio(etapas: int, passos: int, partidas: int) -> dict:
    return {"linhas": [
        {"t": "How a network learns Rummikub", "s": 48, "c": "#FFFFFF", "b": True},
        {"t": "Same deal, same opponent (Max tiles)", "s": 28, "c": "#9DB4D8"},
        {"t": f"{etapas} moments, picked where its behavior changes", "s": 28, "c": "#9DB4D8"},
        {"t": "", "s": 20},
        {"t": "Small neural network, trained by REINFORCE", "s": 26, "c": "#FFFFFF"},
        {"t": "with self-play against heuristics", "s": 26, "c": "#FFFFFF"},
        {"t": "", "s": 20},
        {"t": f"Clips: first {passos} steps of one illustrative game.", "s": 24, "c": "#FFB703"},
        {"t": f"Rates: {partidas} separate games.", "s": 24, "c": "#FFB703"}], "duracao": 2}


def cartela_fim(etapas: list[tuple[int, str, dict]]) -> dict:
    primeiro, ultimo = etapas[0][2], etapas[-1][2]
    linhas = [{"t": "What the network learned", "s": 46, "c": "#FFFFFF", "b": True},
              {"t": "", "s": 18},
              {"cols": ["Iteration", "Win rate", "Draws with a play", "Table plays"], "s": 20, "c": "#9DB4D8"}]
    for marco, _, r in etapas:
        linhas.append({"cols": [str(marco), f"{r['vitorias']:.0%}", f"{r['compra_com_jogada']:.0%}", f"{r['mesa']:.0%}"],
                       "s": 26, "c": "#FFFFFF"})
    if abs(ultimo["vitorias"] - 0.5) <= ultimo["ic"]:
        nivel = "It ended about even with Max tiles."
    elif ultimo["vitorias"] > 0.5:
        nivel = "It beats Max tiles."
    else:
        nivel = "It still loses to Max tiles."
    linhas += [{"t": "Draws with a play = turns it drew although a play existed.", "s": 18, "c": "#5C7099"},
               {"t": f"Against Max tiles: {primeiro['vitorias']:.0%} before training, {ultimo['vitorias']:.0%} after.",
                "s": 26, "c": "#00E5A0"},
               {"t": nivel, "s": 24, "c": "#FFB703"},
               {"t": "It did not reach the stronger Joker saver strategy.", "s": 24, "c": "#FFB703"}]
    return {"linhas": linhas, "duracao": 4}


def _renderizar_cena(cena: str, nome: str, variavel: str, arquivo: Path, pasta: Path, final: bool) -> Path:
    comando = comando_manim(nome, pasta, final)
    comando[-1] = cena
    subprocess.run(comando, check=True, cwd=RAIZ, env={**os.environ, variavel: str(arquivo)}, stdout=subprocess.DEVNULL)
    mp4 = next((pasta / "media" / "videos" / CENA.stem).glob(f"*/{nome}.mp4"))
    destino = pasta / f"{nome}.mp4"
    mp4.replace(destino)
    return destino


def _medir_cacheado(cache: Path, marcos, arquivos: dict, sementes: int, processos: int) -> dict:
    medidas = {int(k): v for k, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
    for marco in marcos:
        if marco not in medidas:
            medidas[marco] = medir_comportamento(arquivos[marco], sementes, processos)
            cache.write_text(json.dumps(medidas))
            print(f"iteração {marco}: {medidas[marco]}", flush=True)
    return medidas


def gerar(saida: Path, pasta_redes: Path, sementes: int, passos: int, processos: int, final: bool,
          sementes_busca: int = 40) -> Path:
    saida.mkdir(parents=True, exist_ok=True)
    arquivos = {m: arquivo_da_rede(m, pasta_redes, saida) for m in (*MARCOS_BUSCA, MARCO_FINAL)}
    serie = _medir_cacheado(saida / "serie.json", MARCOS_BUSCA, arquivos, sementes_busca, processos)
    virada = pontos_de_virada(serie, MARCO_FINAL)
    medidas = _medir_cacheado(saida / "medidas.json", [m for m, _ in virada], arquivos, sementes, processos)
    etapas = [(m, t, medidas[m]) for m, t in virada]
    semente = escolher_semente(virada, arquivos, passos)
    print(f"semente do vídeo: {semente}", flush=True)
    clipes = [_renderizar_cena("Cartela", "00_inicio", "CARTELA",
                               _gravar_json(saida / "inicio.json", cartela_inicio(len(etapas), passos, 2 * sementes)), saida, final)]
    for marco, tag, resumo in etapas:
        replay = gravar(f"rede:{arquivos[marco]}", OPONENTE, semente, 0, "pontos")
        replay = cortar(replay, passos, *legenda_do_marco(marco, tag, resumo, passos), ritmo=RITMO)
        arquivo = _gravar_json(saida / f"replay_{marco:06d}.json", replay)
        clipes.append(_renderizar_cena("Partida", f"{len(clipes):02d}_iter_{marco}", "REPLAY", arquivo, saida, final))
        print(f"clipe {marco}: {clipes[-1]}", flush=True)
    curva = _gravar_json(saida / "curva.json", dados_curva(serie, medidas))
    clipes.append(_renderizar_cena("Curva", f"{len(clipes):02d}_curva", "CURVA", curva, saida, final))
    clipes.append(_renderizar_cena("Cartela", f"{len(clipes):02d}_fim", "CARTELA",
                                   _gravar_json(saida / "fim.json", cartela_fim(etapas)), saida, final))
    lista = saida / "lista.txt"
    lista.write_text("".join(f"file '{c.resolve()}'\n" for c in clipes))
    video = saida / "evolucao.mp4"
    subprocess.run(comando_concatenar(lista, video), check=True)
    return video


def _gravar_json(arquivo: Path, dados: dict) -> Path:
    arquivo.write_text(json.dumps(dados))
    return arquivo
