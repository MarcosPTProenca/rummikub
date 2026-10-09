"""Worker do solver ILP externo. Roda no .venv-solver (tem cvxpy/rummikub-solver).
Lê um pedido JSON por linha {rack, table, initial} e devolve {sets} com os conjuntos
a baixar (listas de ids 1..53), escolhidos para maximizar peças baixadas no turno.
"""
import json
import sys

import rummikub_solver as R

RS = R.RuleSet()  # padrão = regras oficiais: 13 números, 2 cópias, 4 cores, 2 coringas, min 30


def resolver(pedido):
    gs = RS.new_game()
    if pedido["rack"]:
        gs.add_rack(*pedido["rack"])
    if pedido["table"]:
        gs.add_table(*pedido["table"])
    gs.initial = bool(pedido["initial"])
    try:
        sol = RS.solve(gs, R.SolverMode.TILE_COUNT)
    except Exception:
        return {"sets": []}
    if sol is None or not sol.tiles:
        return {"sets": []}
    return {"sets": [[int(t) for t in conjunto] for conjunto in sol.sets]}


def main():
    for linha in sys.stdin:
        linha = linha.strip()
        if not linha:
            continue
        sys.stdout.write(json.dumps(resolver(json.loads(linha))) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
