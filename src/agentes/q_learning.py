import random

from src.rl.contratos import ChaveAcao, ChaveObservacao


class AgenteQLearning:
    """Aprender valores para pares observação–ação."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        semente: int | None = None,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("Alpha deve estar em (0, 1].")
        if not 0 <= gamma <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("Gamma e epsilon devem estar em [0, 1].")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.aleatorio = random.Random(semente)
        self.tabela_q: dict[tuple[ChaveObservacao, ChaveAcao], float] = {}

    def valor(self, observacao: ChaveObservacao, acao: ChaveAcao) -> float:
        """Consultar Q sem modificar a tabela."""
        return self.tabela_q.get((observacao, acao), 0.0)

    def escolher_acao(
        self,
        observacao: ChaveObservacao,
        acoes: list[ChaveAcao],
        explorar: bool = True,
    ) -> ChaveAcao:
        if not acoes:
            raise ValueError("Não há ações disponíveis.")
        if explorar and self.aleatorio.random() < self.epsilon:
            return self.aleatorio.choice(acoes)
        maior = max(self.valor(observacao, acao) for acao in acoes)
        return self.aleatorio.choice([
            acao for acao in acoes if self.valor(observacao, acao) == maior
        ])

    def atualizar(
        self,
        observacao: ChaveObservacao,
        acao: ChaveAcao,
        recompensa: float,
        proxima_observacao: ChaveObservacao,
        proximas_acoes: list[ChaveAcao],
        terminado: bool,
    ) -> None:
        alvo = recompensa
        if not terminado:
            if not proximas_acoes:
                raise ValueError("Uma transição ativa exige próximas ações.")
            alvo += self.gamma * max(
                self.valor(proxima_observacao, proxima) for proxima in proximas_acoes
            )
        atual = self.valor(observacao, acao)
        self.tabela_q[(observacao, acao)] = atual + self.alpha * (alvo - atual)
