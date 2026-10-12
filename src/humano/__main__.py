"""Abre a mesa humana no navegador: `python -m src.humano`."""
import argparse
import webbrowser
from pathlib import Path

from src.humano.mesa import OPONENTES
from src.humano.servidor import criar_servidor


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Jogue Rummikub contra as heurísticas do estudo.")
    p.add_argument("--porta", type=int, default=8000)
    p.add_argument("--oponente", nargs="+", default=["poupar_coringa"], choices=tuple(OPONENTES),
                   help="Um oponente por assento (até 9).")
    p.add_argument("--semente", type=int, help="Repete a mesma distribuição de peças.")
    p.add_argument("--registro", type=Path, default=Path("resultados/humano.jsonl"),
                   help="JSONL com uma linha por partida encerrada.")
    p.add_argument("--sem-registro", action="store_true", help="Não grava as partidas.")
    args = p.parse_args(argv)

    registro = None if args.sem_registro else args.registro
    servidor = criar_servidor(porta=args.porta, oponentes=args.oponente,
                              semente=args.semente, registro=registro)
    url = f"http://127.0.0.1:{servidor.server_address[1]}/"
    print(f"Mesa em {url} (Ctrl+C para encerrar)")
    if registro:
        print(f"Partidas gravadas em {registro}")
    webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
