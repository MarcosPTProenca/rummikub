"""Oponente externo: embrulha o solver ILP de terceiros (mjpieters/rummikub-solver, MIT)
numa política para o nosso motor. O solver roda num venv separado (.venv-solver) porque
depende de cvxpy; aqui só falamos com ele por um worker em subprocesso (JSON por linha).

A cada turno pedimos a jogada que maximiza peças baixadas; se baixa >=1 peça, baixamos
essas combinações, senão compramos. É uma política gulosa por turno (sem retenção de
coringa nem gestão de mão), mas independente das nossas heurísticas e de base acadêmica.
"""
import json
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
PY_SOLVER = RAIZ / ".venv-solver" / "bin" / "python"
WORKER = Path(__file__).with_name("_solver_worker.py")

# Ordem de cores do nosso motor -> blocos de 13 ids no solver (Black, Blue, Orange, Red).
CORES = ["azul", "vermelho", "verde", "amarelo"]


def _id_solver(peca: dict) -> int:
    """Peça do nosso motor (dict cor/numero) -> id 1..53 do solver."""
    if peca["cor"] == "coringa":
        return 53
    return CORES.index(peca["cor"]) * 13 + peca["numero"]


class SolverExterno:
    """Política que delega a decisão de baixa ao solver ILP externo."""

    def __init__(self, *_, **__):
        self._proc = None

    def _worker(self):
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [str(PY_SOLVER), str(WORKER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
        return self._proc

    def __call__(self, jogo, acoes):
        eu = jogo.jogador_atual
        mao = [{"cor": p.cor, "numero": p.numero, "id": p.id} for p in eu.mao]
        mesa = [{"cor": p.cor, "numero": p.numero} for c in jogo.mesa for p in c.pecas]
        pedido = {"rack": [_id_solver(p) for p in mao],
                  "table": [_id_solver(p) for p in mesa],
                  "initial": not eu.abriu}
        w = self._worker()
        w.stdin.write(json.dumps(pedido) + "\n"); w.stdin.flush()
        resposta = json.loads(w.stdout.readline())
        ids_a_baixar = resposta["sets"]  # lista de listas de ids-de-solver (uma por combinação)

        if not ids_a_baixar:
            return next(a for a in acoes if a["tipo"] == "comprar")

        # Casa cada id-de-solver com uma peça concreta da mão (consumindo).
        restante = list(mao)
        ids_concretos = []
        for grupo in ids_a_baixar:
            for alvo in grupo:
                achou = next((p for p in restante if _id_solver(p) == alvo), None)
                if achou is not None:
                    restante.remove(achou)
                    ids_concretos.append(achou["id"])

        # Procura entre as ações válidas a baixa que cobre exatamente essas peças.
        alvo = tuple(sorted(ids_concretos))
        for a in acoes:
            if a["tipo"] == "baixar" and tuple(sorted(a["ids_pecas"])) == alvo:
                return a
        # Se o motor não lista essa baixa combinada, cai para a maior baixa disponível.
        baixas = [a for a in acoes if a["tipo"] == "baixar"]
        if baixas:
            return max(baixas, key=lambda a: len(a["ids_pecas"]))
        return next(a for a in acoes if a["tipo"] == "comprar")

    def __del__(self):
        proc = getattr(self, "_proc", None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        if proc:
            for fluxo in (proc.stdin, proc.stdout):
                if fluxo and not fluxo.closed:
                    fluxo.close()
