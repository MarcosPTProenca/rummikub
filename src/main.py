import argparse
import json

from src.agentes.aleatorio import AgenteAleatorio
from src.agentes.q_learning import AgenteQLearning
from src.jogo.jogo import Jogo
from src.rl.ambiente import AmbienteRL
from src.rl.avaliacao import avaliar
from src.rl.treinamento import treinar


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar e avaliar Q-learning no Rummikub simplificado.")
    parser.add_argument('--episodios', type=int, default=100, help='Episódios de treinamento (padrão: 100).')
    parser.add_argument('--avaliacao', type=int, default=100, help='Episódios de avaliação (padrão: 100).')
    parser.add_argument('--semente', type=int, default=42, help='Semente inicial (padrão: 42).')
    parser.add_argument('--limite-passos', type=int, default=200, help='Decisões por episódio (padrão: 200).')
    args = parser.parse_args()
    if args.episodios < 1 or args.avaliacao < 1 or args.limite_passos < 1:
        parser.error('Episódios e limite de passos devem ser positivos.')

    ambiente = AmbienteRL(lambda semente: Jogo(semente=semente),
                          AgenteAleatorio(), limite_passos=args.limite_passos)
    agente = AgenteQLearning(semente=args.semente)
    resumos = treinar(ambiente, agente, args.episodios, semente_inicial=args.semente)
    metricas = avaliar(ambiente, agente, args.avaliacao,
                       semente_inicial=args.semente + args.episodios)
    print(json.dumps({
        'episodios_treinados': len(resumos),
        'entradas_q': len(agente.tabela_q),
        'avaliacao': metricas,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
