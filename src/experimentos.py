"""Campanha de experimentos: torneio entre estratégias, Q-learning meta e agente Jev."""
import argparse
import json
import os
import random
import shlex
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

from src.agentes.q_learning import AgenteQLearning
from src.estrategias.arena import jogar
from src.estrategias.jev_jogadas import JevJogadas
from src.estrategias.clonagem import clonar
from src.estrategias.rede import PoliticaRede, Rede, treinar_rede
from src.estrategias.jev import JevMeta, Orcamento, transporte_http
from src.estrategias.meta import MetaQ, OPCOES, treinar_meta
from src.estrategias.pimc import PIMC
from src.estrategias.politicas import ESTRATEGIAS, EXTRAS, criar, sem_roubo

PERFIS_LIMITES = {"padrao": {}, "x4": {"planos": 512, "visitas": 4000, "nos": 1024},
                  "x16": {"planos": 2048, "visitas": 16000, "nos": 4096},
                  "xl": {"planos": 4096, "visitas": 64000, "nos": 8192, "pecas": 24, "conjuntos": 3}}
ENV_POC = Path(".env")  # só lido se TYPESAFE_API_KEY não estiver no ambiente
_orcamento: Orcamento | None = None
_chave: str | None = None


def pontos(vencedor: int | None, assento: int) -> float:
    return 0.5 if vencedor is None else float(vencedor == assento)


def pontos_mesa(vencedor: int | None, assento: int, n: int) -> float:
    return 1 / n if vencedor is None else float(vencedor == assento)


def partida_mesa(foco: str, oponentes: list[str], semente: int, assento_foco: int, desempate: str = "pontos",
                 perfil: str = "padrao", agente: AgenteQLearning | None = None) -> dict:
    """Uma estratégia-foco contra N-1 oponentes (nos outros assentos, em ordem)."""
    nomes = list(oponentes)
    nomes.insert(assento_foco, foco)
    jogadores = [construir(nome, semente, assento, agente) for assento, nome in enumerate(nomes)]
    r = jogar(jogadores, semente, desempate=desempate, limites=PERFIS_LIMITES[perfil])
    n = len(nomes)
    jev = {}
    if foco in ("jev_meta", "jev_jogadas"):
        jogador = jogadores[assento_foco]
        jev = {"jev": {**jogador.contagem, **({"escolhas": jogador.escolhas} if foco == "jev_meta" else {})}}
    return {**jev, "foco": foco, "mesa": nomes, "n": n, "semente": semente, "assento_foco": assento_foco,
            "desempate": desempate, "perfil": perfil, "vencedor": r["vencedor"],
            "pontos_foco": pontos_mesa(r["vencedor"], assento_foco, n), "decisoes": r["decisoes"],
            "truncada": r["truncada"], "tele_foco": r["telemetria"][assento_foco],
            "coringas_mao_foco": r["coringas_mao"][assento_foco], "maos": r["maos"],
            "saida_com_coringa_foco": r["saida_com_coringa"] and r["vencedor"] == assento_foco,
            "saturacao": r["saturacao"]}


def tarefas_liga(sementes: int, inicio: int = 140_000, desempate: str = "pontos") -> list[tuple]:
    return [(a, b, inicio + s, assento, None, False, desempate)
            for a, b in combinations([*ESTRATEGIAS, *EXTRAS], 2) for s in range(sementes) for assento in (0, 1)]


def tarefas_robustez(agentes: list[str], sementes: int, inicio: int = 170_000, desempate: str = "pontos",
                     perfil: str = "padrao", oponentes: list[str] | None = None) -> list[tuple]:
    """Cada agente contra os 12 oponentes (ou os dados), mesmas sementes e os dois assentos."""
    rivais = oponentes or [*ESTRATEGIAS, *EXTRAS]
    return [(a, b, inicio + s, assento, None, False, desempate, perfil)
            for a in agentes for b in rivais if a != b for s in range(sementes) for assento in (0, 1)]


def tarefas_mesa(n: int, focos: list[str], campos: list[str], sementes: int, inicio: int = 120_000,
                 desempate: str = "pontos", perfil: str = "padrao", referencia: str | None = None) -> list[tuple]:
    """Com `referencia`, os oponentes do campo misto são os sorteados para essa estratégia (comparação pareada).
    Tarefas (foco, oponentes 'a|b|..', semente, assento_foco, desempate, perfil); o foco roda por todos os assentos."""
    tarefas = []
    for foco in focos:
        for campo in campos:
            for s in range(sementes):
                if campo == "iguais":
                    oponentes = ["max_pecas"] * (n - 1)
                elif campo == "externo":
                    oponentes = ["solver_ilp"] * (n - 1)
                else:
                    base = referencia or foco
                    sorteio = random.Random(f"{base}|{n}|{inicio + s}")
                    outras = [e for e in ESTRATEGIAS if e != base]
                    oponentes = sorteio.sample(outras, n - 1) if n - 1 <= len(outras) else sorteio.choices(outras, k=n - 1)
                tarefas += [(foco, "|".join(oponentes), inicio + s, assento, desempate, perfil) for assento in range(n)]
    return tarefas


def construir(nome: str, semente: int, assento: int, agente: AgenteQLearning | None = None):
    rng = random.Random(semente * 10 + assento)
    if nome == "rl_meta":
        return MetaQ(agente, rng)
    if nome.startswith("rede:"):
        return PoliticaRede.da_rede(Rede.carregar(nome[5:]), rng, guloso=True)
    if nome == "solver_ilp":
        from .estrategias.solver_externo import SolverExterno
        return SolverExterno()
    if nome.startswith("pimc"):
        tipo, d, k = (nome.split(":") + ["4", "5"])[:3]
        return PIMC(rng, int(d), int(k), ve_tudo=tipo == "pimcv")
    if nome == "jev_meta":
        return JevMeta(transporte_http(_chave), rng, _orcamento)
    if nome == "jev_jogadas":
        return JevJogadas(transporte_http(_chave), rng, _orcamento)
    return criar(nome, rng)


def partida(a: str, b: str, semente: int, assento_a: int, agente: AgenteQLearning | None = None,
            congelar: bool = False, desempate: str = "pontos", perfil: str = "padrao") -> dict:
    jogadores = [None, None]
    jogadores[assento_a] = construir(a, semente, assento_a, agente)
    jogadores[1 - assento_a] = construir(b, semente, 1 - assento_a, agente)
    if congelar:
        jogadores = [sem_roubo(j) for j in jogadores]
    resultado = jogar(jogadores, semente, desempate=desempate, limites=PERFIS_LIMITES[perfil])
    registro = {"a": a, "b": b, "semente": semente, "assento_a": assento_a, "congelar": congelar, "desempate": desempate,
                "perfil": perfil, "saturacao": resultado["saturacao"],
                "pontos_a": pontos(resultado["vencedor"], assento_a),
                "decisoes": resultado["decisoes"], "truncada": resultado["truncada"],
                "tele_a": resultado["telemetria"][assento_a], "tele_b": resultado["telemetria"][1 - assento_a],
                "coringas_mao_a": resultado["coringas_mao"][assento_a],
                "coringas_mao_b": resultado["coringas_mao"][1 - assento_a],
                "saida_com_coringa_a": resultado["saida_com_coringa"] and resultado["vencedor"] == assento_a}
    for nome, jogador, assento in ((a, jogadores[assento_a], assento_a), (b, jogadores[1 - assento_a], 1 - assento_a)):
        if nome == "jev_meta":
            registro["jev"] = {**jogador.contagem, "escolhas": jogador.escolhas}
        elif nome == "jev_jogadas":
            registro["jev"] = {**jogador.contagem, "concordancia": jogador.concordancia,
                               "confiancas": jogador.confiancas}
    return registro


def _tarefa_partida(tarefa):
    return partida(*tarefa)


def _tarefa_mesa(tarefa):
    return partida_mesa(tarefa[0], tarefa[1].split("|"), *tarefa[2:])


def _tarefa_replica(tarefa) -> dict:
    replica, episodios, avaliacao, semente, desempate, jogadores = tarefa
    agente = AgenteQLearning(gamma=1.0, semente=semente)
    retornos = treinar_meta(agente, episodios, list(ESTRATEGIAS), semente_inicial=semente * 100_000,
                            desempate=desempate, jogadores=jogadores)
    agente.epsilon = 0.0
    if jogadores == 2:
        avaliadas = [partida("rl_meta", oponente, 9_000_000 + s, assento, agente, desempate=desempate)
                     for oponente in ESTRATEGIAS for s in range(avaliacao) for assento in (0, 1)]
    else:
        avaliadas = [partida_mesa(f, ops.split("|"), s, assento, d, perfil, agente)
                     for f, ops, s, assento, d, perfil in tarefas_mesa(
                         jogadores, ["rl_meta"], ["iguais", "misto"], avaliacao, inicio=9_000_000, desempate=desempate)]
    for registro in avaliadas:
        registro["replica"] = replica
    tabela = [[list(map(str, estado)), opcao, valor] for (estado, opcao), valor in agente.tabela_q.items()]
    return {"replica": replica, "desempate": desempate, "jogadores": jogadores, "retornos": retornos, "avaliacao": avaliadas, "tabela_q": tabela}


def _iniciar_jev(chave: str, limite: float, orcamento: Orcamento) -> None:
    global _chave, _orcamento
    _chave, _orcamento = chave, orcamento


def carregar_chave(arquivo: Path) -> str:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    for linha in arquivo.read_text().splitlines():
        if linha.lstrip().startswith(("TYPESAFE_API_KEY=", "export TYPESAFE_API_KEY=")):
            return shlex.split(linha.split("=", 1)[1], comments=True)[0]
    raise SystemExit("TYPESAFE_API_KEY não encontrada.")


def executar(tarefas, saida: Path, processos: int, iniciador=None, args=(), funcao=None) -> None:
    """Gravar cada resultado assim que chega; tarefas já gravadas são puladas."""
    feitos = set()
    if saida.exists():
        feitos = {tuple(json.loads(l)["_tarefa"]) for l in saida.read_text().splitlines()}
    pendentes = [t for t in tarefas if tuple(t) not in feitos]
    print(f"{len(pendentes)} partidas pendentes ({len(feitos)} já gravadas)", flush=True)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with Pool(processos, initializer=iniciador, initargs=args) as pool, saida.open("a") as arquivo:
        for i, (tarefa, registro) in enumerate(zip(pendentes, pool.imap(funcao or _tarefa_partida, pendentes)), 1):
            registro["_tarefa"] = list(tarefa)
            arquivo.write(json.dumps(registro) + "\n")
            arquivo.flush()
            if i % 100 == 0:
                print(f"{i}/{len(pendentes)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    torneio = sub.add_parser("torneio", help="Todos contra todos entre as estratégias heurísticas.")
    torneio.add_argument("--sementes", type=int, default=80)
    liga = sub.add_parser("liga", help="Todos contra todos: 7 heurísticas + 5 estratégias exploratórias.")
    liga.add_argument("--sementes", type=int, default=80)
    treino = sub.add_parser("treinar", help="Treinar réplicas do Q-learning meta e avaliá-las.")
    treino.add_argument("--replicas", type=int, default=5)
    treino.add_argument("--episodios", type=int, default=1200)
    treino.add_argument("--avaliacao", type=int, default=20, help="Sementes por oponente.")
    treino.add_argument("--jogadores", type=int, choices=(2, 3, 4), default=2)
    mec = sub.add_parser("mecanismo", help="Ablações: por que poupar coringas vence.")
    mec.add_argument("--sementes", type=int, default=1000)
    mec.add_argument("--apenas", nargs="+", help="Só os confrontos cujo primeiro jogador está na lista.")
    mec.add_argument("--sem-congelar", action="store_true", help="Pular as condições com coringas congelados (lentas).")
    lim = sub.add_parser("limites", help="Torneio entre heurísticas variando o corte das reorganizações.")
    lim.add_argument("--perfis", nargs="+", default=["padrao"], choices=list(PERFIS_LIMITES))
    lim.add_argument("--sementes", type=int, default=40)
    mesa = sub.add_parser("mesa", help="Uma estratégia-foco contra N-1 oponentes (3 ou 4 jogadores).")
    mesa.add_argument("--jogadores", type=int, nargs="+", default=[3, 4], choices=range(3, 11))
    mesa.add_argument("--semente-inicial", type=int, default=120_000)
    mesa.add_argument("--focos", nargs="+", default=list(ESTRATEGIAS))
    mesa.add_argument("--campos", nargs="+", default=["iguais", "misto"], choices=("iguais", "misto", "externo"))
    mesa.add_argument("--sementes", type=int, default=40)
    mesa.add_argument("--orcamento", type=float, default=2.0, help="Limite em US$ se o foco for um agente Jev.")
    mesa.add_argument("--env", type=Path, default=ENV_POC)
    mesa.add_argument("--referencia", help="Estratégia cujos oponentes do campo misto o foco deve enfrentar.")
    mesa.add_argument("--rotulo", default="mesa", help="Nome do arquivo em resultados/.")
    busca = sub.add_parser("busca", help="Agente de busca (PIMC) contra heurísticas, pareado.")
    busca.add_argument("--jogador", default="pimc:4:5", help="pimc:<determinizações>:<candidatas>")
    busca.add_argument("--oponentes", nargs="+", default=["poupar_coringa", "max_pecas"])
    busca.add_argument("--sementes", type=int, default=60)
    busca.add_argument("--semente-inicial", type=int, default=100_000)
    rede = sub.add_parser("rede", help="Treina uma rede que pontua jogadas (REINFORCE, self-play e liga).")
    rede.add_argument("--iteracoes", type=int, default=300)
    rede.add_argument("--jogos", type=int, default=24, help="Partidas por iteração.")
    rede.add_argument("--ocultos", type=int, default=16)
    rede.add_argument("--lr", type=float, default=3e-3)
    rede.add_argument("--checkpoint-a-cada", type=int, default=50)
    rede.add_argument("--semente", type=int, default=0)
    rede.add_argument("--inicial", type=Path, help="Continua de um checkpoint.")
    rede.add_argument("--desempate", choices=("pontos", "pecas", "empate"), default="pontos")
    rede.add_argument("--processos", type=int, default=3)
    rede.add_argument("--jogadores", type=int, choices=range(2, 11), default=2)
    rede.add_argument("--mesas", type=int, nargs="+", choices=range(2, 11), default=(),
                      help="Currículo: cada jogo de treino sorteia o tamanho da mesa desta lista.")
    rede.add_argument("--rearranjar", action="store_true", help="Treina com reorganização da mesa ligada.")
    rede.add_argument("--liga", nargs="*", default=[], choices=tuple(EXTRAS),
                      help="Estratégias extras como oponentes de treino, além das 7 originais.")
    rede.add_argument("--forma", type=float, default=0.0, help="Peso do reward shaping pela queda da mão (0 = sem).")
    rede.add_argument("--grupo", type=int, default=1, help="Partidas com a mesma mão por grupo (vantagem relativa ao grupo).")
    rede.add_argument("--residual", type=float, default=0.0,
                      help="RL residual: soma este peso à jogada do poupar_coringa; rede zerada (0 = sem).")
    rede.add_argument("--algoritmo", choices=("reinforce", "dmc", "ppo"), default="reinforce",
                      help="reinforce (padrão), dmc (Deep Monte Carlo, ε-guloso) ou ppo (crítico, clip, KL à rede inicial).")
    rede.add_argument("--epsilon", type=float, default=0.05, help="dmc: probabilidade de explorar.")
    rede.add_argument("--clip", type=float, default=0.2, help="ppo: faixa da razão.")
    rede.add_argument("--kl", type=float, default=0.1, help="ppo: peso da KL à rede inicial.")
    rede.add_argument("--epocas", type=int, default=4, help="dmc/ppo: passos de gradiente por iteração.")
    rede.add_argument("--saida", type=Path, default=Path("resultados/rede"))
    rob = sub.add_parser("robustez", help="Agentes contra os 12 oponentes (vistos e não vistos no treino), com perfil de limites.")
    rob.add_argument("--agentes", nargs="+", required=True)
    rob.add_argument("--oponentes", nargs="+")
    rob.add_argument("--sementes", type=int, default=40)
    rob.add_argument("--perfil", choices=tuple(PERFIS_LIMITES), default="padrao")
    rob.add_argument("--processos", type=int, default=5)
    rob.add_argument("--saida", type=Path, required=True)
    clone = sub.add_parser("clonar", help="Clona uma heurística na rede (aprendizado supervisionado).")
    clone.add_argument("--demonstrador", default="poupar_coringa")
    clone.add_argument("--oponentes", nargs="+", default=list(ESTRATEGIAS))
    clone.add_argument("--partidas", type=int, default=300)
    clone.add_argument("--epocas", type=int, default=5)
    clone.add_argument("--ocultos", type=int, default=16)
    clone.add_argument("--lr", type=float, default=3e-3)
    clone.add_argument("--semente", type=int, default=0)
    clone.add_argument("--jogadores", type=int, choices=(2, 3, 4), default=2)
    clone.add_argument("--desempate", choices=("pontos", "pecas", "empate"), default="pontos")
    clone.add_argument("--processos", type=int, default=3)
    clone.add_argument("--saida", type=Path, default=Path("resultados/rede_clone"))
    jev = sub.add_parser("jev", help="Agente Jev contra oponentes escolhidos.")
    jev.add_argument("--oponentes", nargs="+", default=["max_pecas", "aleatorio"])
    jev.add_argument("--sementes", type=int, default=100)
    jev.add_argument("--orcamento", type=float, default=1.8, help="Limite em US$.")
    jev.add_argument("--env", type=Path, default=ENV_POC)
    jev.add_argument("--agente", choices=("meta", "jogadas"), default="meta",
                     help="meta: escolhe entre 7 estratégias; jogadas: escolhe entre todas as jogadas legais.")
    for p in (torneio, liga, treino, mec, busca, jev, mesa):
        p.add_argument("--desempate", choices=("pontos", "pecas", "empate"), default="pontos",
                       help="Regra com o monte esgotado; 'empate' reproduz os resultados de 2026-10 (arquivos sem sufixo).")
    for p in (torneio, liga, treino, mec, busca, jev, lim, mesa):
        p.add_argument("--processos", type=int, default=6)
        p.add_argument("--saida", type=Path)
    args = parser.parse_args()
    pasta = Path("resultados")
    desempate = getattr(args, "desempate", "empate")
    if args.comando == "jev" and args.agente == "meta" and "--desempate" not in sys.argv:
        desempate = "empate"
    sufixo = "" if desempate == "empate" else f"_{desempate}"
    if args.comando == "torneio":
        tarefas = [(a, b, 1000 + s, assento, None, False, desempate) for a, b in combinations(ESTRATEGIAS, 2)
                   for s in range(args.sementes) for assento in (0, 1)]
        executar(tarefas, args.saida or pasta / f"torneio{sufixo}.jsonl", args.processos)
    elif args.comando == "liga":
        executar(tarefas_liga(args.sementes, desempate=desempate), args.saida or pasta / f"liga{sufixo}.jsonl", args.processos)
    elif args.comando == "mecanismo":
        confrontos = [("poupar_coringa", "max_pecas", False), ("max_pecas", "max_pecas", False),
                      ("poupar_coringa", "max_pecas", True), ("max_pecas", "max_pecas", True)]
        confrontos += [(v, "max_pecas", False) for v in
                       ("penal_0.5", "penal_1", "penal_2", "penal_4", "bonus_1", "bonus_2",
                        "poupar_antes", "poupar_depois", "poupar_sem_final")]
        confrontos = [c for c in confrontos if (not args.apenas or c[0] in args.apenas)
                      and not (args.sem_congelar and c[2])]
        tarefas = [(a, b, 70_000 + s, assento, None, congelar, desempate) for a, b, congelar in confrontos
                   for s in range(args.sementes) for assento in (0, 1)]
        executar(tarefas, args.saida or pasta / f"mecanismo{sufixo}.jsonl", args.processos)
    elif args.comando == "limites":
        tarefas = [(a, b, 150_000 + s, assento, None, False, desempate, perfil)
                   for perfil in args.perfis for a, b in combinations(ESTRATEGIAS, 2)
                   for s in range(args.sementes) for assento in (0, 1)]
        executar(tarefas, args.saida or pasta / "limites.jsonl", args.processos)
    elif args.comando == "mesa":
        tarefas = [t for n in args.jogadores
                   for t in tarefas_mesa(n, args.focos, args.campos, args.sementes, args.semente_inicial,
                                          desempate=desempate, referencia=args.referencia)]
        orcamento = Orcamento(args.orcamento)
        usa_jev = any(f.startswith("jev") for f in args.focos)
        executar(tarefas, args.saida or pasta / f"{args.rotulo}{sufixo}.jsonl", args.processos, funcao=_tarefa_mesa,
                 iniciador=_iniciar_jev if usa_jev else None,
                 args=(carregar_chave(args.env), args.orcamento, orcamento) if usa_jev else ())
        if usa_jev:
            print(f"gasto Jev: US$ {orcamento.gasto:.4f} (limite {args.orcamento})")
    elif args.comando == "busca":
        tarefas = [(args.jogador, o, args.semente_inicial + s, assento, None, False, desempate)
                   for o in args.oponentes for s in range(args.sementes) for assento in (0, 1)]
        executar(tarefas, args.saida or pasta / f"busca{sufixo}.jsonl", args.processos)
    elif args.comando == "rede":
        treinar_rede(args.saida, args.iteracoes, args.jogos, args.processos, args.ocultos, args.semente,
                     args.desempate, args.checkpoint_a_cada, args.lr, inicial=args.inicial, jogadores=args.jogadores,
                     grupo=args.grupo, rearranjar=args.rearranjar, forma=args.forma,
                     liga=tuple(args.liga), residual=args.residual, algoritmo=args.algoritmo, epsilon=args.epsilon,
                     clip=args.clip, kl=args.kl, epocas=args.epocas, mesas=tuple(args.mesas))
    elif args.comando == "robustez":
        executar(tarefas_robustez(args.agentes, args.sementes, perfil=args.perfil, oponentes=args.oponentes),
                 args.saida, args.processos)
    elif args.comando == "clonar":
        clonar(args.saida, args.demonstrador, args.oponentes, args.partidas, args.epocas, args.processos,
               args.ocultos, args.semente, args.lr, args.jogadores, args.desempate)
    elif args.comando == "treinar":
        saida = args.saida or pasta / f"treino{sufixo}.jsonl"
        saida.parent.mkdir(parents=True, exist_ok=True)
        tarefas = [(r, args.episodios, args.avaliacao, r + 1, desempate, args.jogadores) for r in range(args.replicas)]
        with Pool(args.processos) as pool, saida.open("w") as arquivo:
            for resultado in pool.imap_unordered(_tarefa_replica, tarefas):
                arquivo.write(json.dumps(resultado) + "\n")
                arquivo.flush()
                print(f"réplica {resultado['replica']} concluída", flush=True)
    else:
        orcamento = Orcamento(args.orcamento)
        nome, base = ("jev_jogadas", 4000) if args.agente == "jogadas" else ("jev_meta", 2000)
        tarefas = [(nome, o, base + s, assento, None, False, desempate) for o in args.oponentes
                   for s in range(args.sementes) for assento in (0, 1)]
        padrao = f"jev_{args.agente}{sufixo}.jsonl" if args.agente == "jogadas" or sufixo else "jev.jsonl"
        executar(tarefas, args.saida or pasta / padrao, args.processos,
                 _iniciar_jev, (carregar_chave(args.env), args.orcamento, orcamento))
        print(f"gasto Jev: US$ {orcamento.gasto:.4f} (limite {args.orcamento})")


if __name__ == "__main__":
    main()
