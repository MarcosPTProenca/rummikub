"""Grava uma partida passo a passo (estados completos) para alimentar as animações."""
import json
from pathlib import Path

from src import experimentos
from src.agentes.q_learning import AgenteQLearning
from src.estrategias.arena import jogar


def texto_da_jogada(tipo: str, n: int, soma: int, coringas: int, guardou_coringa: bool, abertura: bool = False) -> str:
    if tipo == "comprar":
        return "Draws a tile, keeping the joker" if guardou_coringa else "Draws a tile"
    corpo = f"plays {n} tile{'s' if n != 1 else ''} ({soma} pts{', with a joker' if coringas else ''})"
    if tipo == "mesa":
        return f"Rearranges the table and {corpo}"
    return ("Opens: " if abertura else "") + corpo[0].upper() + corpo[1:]


_NOMES_DAS_CORES = {"azul": "Blue", "vermelho": "Red", "verde": "Green", "amarelo": "Yellow"}
_ORDEM_DAS_CORES = list(_NOMES_DAS_CORES) + ["coringa"]


def descrever_pecas(pecas: list, limite: int = 4) -> str:
    ordenadas = sorted(pecas, key=lambda p: (_ORDEM_DAS_CORES.index(p[1]), p[2]))
    nomes = ["Joker" if p[1] == "coringa" else f"{_NOMES_DAS_CORES[p[1]]} {p[2]}" for p in ordenadas]
    resto = len(nomes) - limite
    return ", ".join(nomes[:limite]) + (f" +{resto} more" if resto > 0 else "")


def _peca(p) -> list:
    return [p.id, p.cor, p.numero]


def _estado(jogo) -> dict:
    return {
        "maos": [[_peca(p) for p in j.mao] for j in jogo.jogadores],
        "mesa": [[_peca(p) for p in c.pecas] for c in jogo.mesa],
        "monte": len(jogo.monte),
        "abriu": [bool(j.abriu) for j in jogo.jogadores],
        "pontos": [sum(30 if p.cor == "coringa" else p.numero for p in j.mao) for j in jogo.jogadores],
    }


def carregar_q(arquivo: Path, replica: int = 0) -> AgenteQLearning:
    """Reconstrói a tabela Q gravada em treino*.jsonl (chaves de estado voltam a bool/int)."""
    def valor(s: str):
        return {"True": True, "False": False}.get(s) if s in ("True", "False") else int(s)
    for linha in Path(arquivo).read_text().splitlines():
        dados = json.loads(linha)
        if dados["replica"] == replica:
            agente = AgenteQLearning(gamma=1.0)
            agente.epsilon = 0.0
            agente.tabela_q = {(tuple(valor(s) for s in estado), opcao): v for estado, opcao, v in dados["tabela_q"]}
            return agente
    raise KeyError(f"réplica {replica} não encontrada em {arquivo}")


def gravar(a: str, b: str, semente: int, assento_a: int = 0, desempate: str = "pontos",
           agente: AgenteQLearning | None = None) -> dict:
    jogadores = [None, None]
    jogadores[assento_a] = experimentos.construir(a, semente, assento_a, agente)
    jogadores[1 - assento_a] = experimentos.construir(b, semente, 1 - assento_a, agente)
    inicio, passos, abriu_antes = {}, [], [False, False]
    maos_antes = []
    havia_jogada = []

    def vendo(politica):
        def escolher(jogo, acoes):
            havia_jogada.append(any(a["tipo"] != "comprar" for a in acoes))
            return politica(jogo, acoes)
        return escolher

    jogadores = [vendo(j) for j in jogadores]

    def observador(jogo, lado, acao, medidas):
        n, soma, coringas = medidas
        depois = {p[0] for p in _estado(jogo)["maos"][lado]}
        jogadas = [p for p in maos_antes[-1][lado] if p[0] not in depois]
        guardou = acao["tipo"] == "comprar" and any(p.cor == "coringa" for p in jogo.jogadores[lado].mao)
        abertura = jogo.jogadores[lado].abriu and not abriu_antes[lado]
        abriu_antes[lado] = bool(jogo.jogadores[lado].abriu)
        texto = texto_da_jogada(acao["tipo"], n, soma, coringas, guardou, abertura)
        tinha_jogada = havia_jogada[-1]
        if acao["tipo"] == "comprar" and tinha_jogada:
            texto += " (could have played)"
        estado = _estado(jogo)
        passos.append({"jogador": lado, "tipo": acao["tipo"], "n": n, "pontos": soma, "coringas": coringas,
                       "tinha_jogada": tinha_jogada,
                       "pecas": jogadas if acao["tipo"] != "comprar" else [],
                       "texto": f"{texto} - {descrever_pecas(jogadas)}" if acao["tipo"] != "comprar" and jogadas else texto,
                       "estado": estado})
        maos_antes.append(estado["maos"])

    from src.jogo.jogo import Jogo
    inicio = _estado(Jogo(semente=semente, desempate=desempate))
    maos_antes.append(inicio["maos"])
    resultado = jogar(jogadores, semente, desempate=desempate, observador=observador)
    final = passos[-1]["estado"] if passos else inicio
    vencedor = resultado["vencedor"]
    if resultado["truncada"]:
        motivo = "truncada"
    elif vencedor is not None and not final["maos"][vencedor]:
        motivo = "mao_vazia"
    else:
        motivo = "monte_esgotado"
    return {"estrategia_a": a, "estrategia_b": b, "semente": semente, "assento_a": assento_a, "desempate": desempate,
            "vencedor": vencedor, "decisoes": resultado["decisoes"], "motivo": motivo, "inicio": inicio, "passos": passos}
