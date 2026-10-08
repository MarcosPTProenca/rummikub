"""Replay -> Manim -> MP4 -> GIF. Funções puras montam os comandos; `renderizar` os executa."""
import json
import os
import subprocess
from pathlib import Path

from src import experimentos
from src.animacao.replay import carregar_q, gravar
from src.estrategias.jev import Orcamento

RAIZ = Path(__file__).resolve().parents[2]
VENV = RAIZ / ".venv-animacao" / "bin"
CENA = RAIZ / "src" / "animacao" / "cena.py"
HEURISTICAS = ["aleatorio", "so_baixar", "minimo", "cauteloso", "max_pontos", "max_pecas", "poupar_coringa"]
ESTRATEGIAS_ANIMADAS = HEURISTICAS + ["rl_meta", "jev_meta", "jev_jogadas", "pimc:4:5",
                                      "rede:resultados/rede/final.json"]
BARATAS = [e for e in ESTRATEGIAS_ANIMADAS if not e.startswith(("jev", "pimc"))]


def nome_arquivo(estrategia: str) -> str:
    return estrategia.split(":")[0]


def comando_manim(nome: str, pasta: Path, final: bool) -> list[str]:
    return [str(VENV / "manim"), "render", "-r", "1080,1350", "--fps", "30" if final else "15",
            "--disable_caching", "--progress_bar", "none", "--media_dir", str(pasta / "media"), "-o", nome, str(CENA), "Partida"]


def comando_gif(mp4: Path, gif: Path, segundos: int = 25) -> list[str]:
    filtro = "fps=12,scale=540:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
    return [str(VENV / "ffmpeg"), "-y", "-loglevel", "error", "-t", str(segundos), "-i", str(mp4),
            "-vf", filtro, str(gif)]


def escolher_semente(estrategias, inicio: int, limite: int, decisoes, tentativas: int = 200):
    """Primeira semente em que todas as partidas terminam com <= limite decisões; relata as que estouraram."""
    relatorio = {}
    for semente in range(inicio, inicio + tentativas):
        for estrategia in estrategias:
            n = decisoes(estrategia, semente)
            if n > limite:
                relatorio[semente] = {estrategia: n}
                break
        else:
            return semente, relatorio
    raise RuntimeError(f"nenhuma semente em [{inicio}, {inicio + tentativas}) cabe em {limite} decisões")


def decisoes_da_partida(estrategia: str, semente: int, oponente: str = "max_pecas", agente=None) -> int:
    return experimentos.partida(estrategia, oponente, semente, 0, agente)["decisoes"]


def renderizar(estrategia: str, semente: int, saida: Path, oponente: str = "max_pecas", final: bool = True,
               agente=None, gif_segundos: int = 25) -> dict:
    nome = nome_arquivo(estrategia)
    saida.mkdir(parents=True, exist_ok=True)
    replay = gravar(estrategia, oponente, semente, 0, "pontos", agente)
    arquivo = saida / f"{nome}.json"
    arquivo.write_text(json.dumps(replay))
    subprocess.run(comando_manim(nome, saida, final), check=True, cwd=RAIZ,
                   env={**os.environ, "REPLAY": str(arquivo)}, stdout=subprocess.DEVNULL)
    mp4 = next((saida / "media" / "videos" / "cena").glob(f"*/{nome}.mp4"))
    destino = saida / f"{nome}.mp4"
    mp4.replace(destino)
    gif = saida / f"{nome}.gif"
    subprocess.run(comando_gif(destino, gif, gif_segundos), check=True)
    return {"estrategia": estrategia, "mp4": destino, "gif": gif, "decisoes": replay["decisoes"],
            "vencedor": replay["vencedor"], "motivo": replay["motivo"]}


def preparar_api(env: Path, orcamento: float) -> None:
    experimentos._iniciar_jev(experimentos.carregar_chave(env), orcamento, Orcamento(orcamento))


__all__ = ["ESTRATEGIAS_ANIMADAS", "BARATAS", "carregar_q", "comando_gif", "comando_manim", "decisoes_da_partida",
           "escolher_semente", "nome_arquivo", "preparar_api", "renderizar"]
