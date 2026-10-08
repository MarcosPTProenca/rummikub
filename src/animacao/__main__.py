import argparse
import json
from pathlib import Path

from src import experimentos
from src.animacao.replay import carregar_q, gravar
from src.animacao import pipeline
from src.estrategias.jev import Orcamento


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.animacao")
    sub = parser.add_subparsers(dest="comando", required=True)
    r = sub.add_parser("replay", help="Joga uma partida e grava o replay em JSON.")
    r.add_argument("--estrategia", required=True, help="Nome (heurística, rl_meta, jev_meta, jev_jogadas, pimc:D:K, rede:ARQ).")
    r.add_argument("--oponente", default="max_pecas")
    r.add_argument("--semente", type=int, default=110_000)
    r.add_argument("--assento", type=int, choices=(0, 1), default=0)
    r.add_argument("--desempate", choices=("pontos", "pecas", "empate"), default="pontos")
    r.add_argument("--q", type=Path, default=Path("resultados/treino_pontos.jsonl"), help="Tabela Q para rl_meta.")
    r.add_argument("--env", type=Path, default=experimentos.ENV_POC)
    r.add_argument("--orcamento", type=float, default=0.05, help="Teto de gasto da API em US$ (Jev).")
    r.add_argument("--saida", type=Path, required=True)
    s = sub.add_parser("semente", help="Primeira semente em que as estratégias baratas terminam com poucas decisões.")
    s.add_argument("--inicio", type=int, default=110_000)
    s.add_argument("--limite", type=int, default=120)
    s.add_argument("--oponente", default="max_pecas")
    s.add_argument("--q", type=Path, default=Path("resultados/treino_pontos.jsonl"))
    e = sub.add_parser("evolucao", help="Vídeo da rede em marcos do treino (mesma partida, taxa medida).")
    e.add_argument("--redes", type=Path, default=Path("resultados/rede"))
    e.add_argument("--sementes", type=int, default=100, help="Sementes de medição (2 assentos cada).")
    e.add_argument("--passos", type=int, default=12, help="Turnos mostrados por clipe.")
    e.add_argument("--processos", type=int, default=2)
    e.add_argument("--teste", action="store_true")
    e.add_argument("--saida", type=Path, default=Path("animacao/saida/evolucao"))
    v = sub.add_parser("renderizar", help="Replay -> Manim -> MP4 -> GIF.")
    v.add_argument("--estrategia", action="append", help="Repetível; omita com --todas.")
    v.add_argument("--todas", action="store_true")
    v.add_argument("--semente", type=int, default=110_000)
    v.add_argument("--oponente", default="max_pecas")
    v.add_argument("--teste", action="store_true", help="480p-ish, 15 fps: só para conferir o visual.")
    v.add_argument("--gif-segundos", type=int, default=25)
    v.add_argument("--saida", type=Path, default=Path("animacao/saida"))
    v.add_argument("--q", type=Path, default=Path("resultados/treino_pontos.jsonl"))
    v.add_argument("--env", type=Path, default=experimentos.ENV_POC)
    v.add_argument("--orcamento", type=float, default=0.05, help="Teto de gasto da API em US$ (Jev).")
    args = parser.parse_args()
    if args.comando == "evolucao":
        from src.animacao import evolucao
        print(evolucao.gerar(args.saida, args.redes, args.sementes, args.passos, args.processos, not args.teste))
        return
    if args.comando == "semente":
        agente = carregar_q(args.q)
        semente, relatorio = pipeline.escolher_semente(
            pipeline.BARATAS, args.inicio, args.limite,
            lambda e, sem: pipeline.decisoes_da_partida(e, sem, args.oponente, agente))
        for sem, falha in relatorio.items():
            print(f"{sem}: estoura {falha}")
        print(f"semente escolhida: {semente}")
        return
    if args.comando == "renderizar":
        estrategias = pipeline.ESTRATEGIAS_ANIMADAS if args.todas else args.estrategia
        if not estrategias:
            parser.error("informe --estrategia ou --todas")
        if any(e.startswith("jev") for e in estrategias):
            pipeline.preparar_api(args.env, args.orcamento)
        agente = carregar_q(args.q) if "rl_meta" in estrategias else None
        for e in estrategias:
            r = pipeline.renderizar(e, args.semente, args.saida, args.oponente, not args.teste, agente,
                                    args.gif_segundos)
            print(f"{e}: {r['mp4']} ({r['decisoes']} decisões, vencedor={r['vencedor']}, {r['motivo']})", flush=True)
        return
    agente = carregar_q(args.q) if "rl_meta" in (args.estrategia, args.oponente) else None
    if args.estrategia.startswith("jev") or args.oponente.startswith("jev"):
        experimentos._iniciar_jev(experimentos.carregar_chave(args.env), args.orcamento, Orcamento(args.orcamento))
    replay = gravar(args.estrategia, args.oponente, args.semente, args.assento, args.desempate, agente)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(replay))
    print(f"{args.saida}: {replay['decisoes']} decisões, vencedor={replay['vencedor']} ({replay['motivo']})")


if __name__ == "__main__":
    main()
