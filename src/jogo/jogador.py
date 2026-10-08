from pydantic import BaseModel, ConfigDict, Field
from .peca import Peca

class Jogador(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    mao: list[Peca] = Field(default_factory=list)
    abriu: bool = False

    def receber_peca(self, peca: Peca) -> None:
        if any(item.id == peca.id for item in self.mao):
            raise ValueError("A peça já está na mão do jogador.")

        self.mao.append(peca)

    def remover_pecas(self, pecas: list[Peca]) -> None:
        if not self.possui_pecas(pecas):
            raise ValueError("As peças solicitadas não estão disponíveis na mão.")

        ids_remover = {peca.id for peca in pecas}
        self.mao = [
            peca for peca in self.mao
            if peca.id not in ids_remover
        ]

    def possui_pecas(self, pecas: list[Peca]) -> bool:
        ids_solicitados = {peca.id for peca in pecas}
        ids_mao = {peca.id for peca in self.mao}

        sem_repeticao = len(ids_solicitados) == len(pecas)
        todas_na_mao = ids_solicitados.issubset(ids_mao)

        return sem_repeticao and todas_na_mao

    def esta_sem_peca(self) -> bool:
        return len(self.mao) == 0


if __name__ == "__main__":
    peca_1 = Peca(id="1", cor="azul", numero=7)
    peca_2 = Peca(id="2", cor="vermelho", numero=7)
    peca_3 = Peca(id="3", cor="verde", numero=7)

    jogador = Jogador(id="0")
    outro_jogador = Jogador(id="1")

    assert jogador.esta_sem_peca()

    jogador.receber_peca(peca_1)
    jogador.receber_peca(peca_2)

    assert len(jogador.mao) == 2
    assert not jogador.esta_sem_peca()
    assert outro_jogador.esta_sem_peca()

    assert jogador.possui_pecas([peca_1, peca_2])
    assert not jogador.possui_pecas([peca_3])
    assert not jogador.possui_pecas([peca_1, peca_1])

    try:
        jogador.receber_peca(peca_1)
    except ValueError:
        pass
    else:
        raise AssertionError("Não deveria aceitar uma peça duplicada.")

    try:
        jogador.remover_pecas([peca_1, peca_3])
    except ValueError:
        pass
    else:
        raise AssertionError("Não deveria remover peças indisponíveis.")

    # A tentativa inválida não deve alterar a mão.
    assert len(jogador.mao) == 2
    assert jogador.possui_pecas([peca_1, peca_2])

    jogador.remover_pecas([peca_1])

    assert len(jogador.mao) == 1
    assert not jogador.possui_pecas([peca_1])
    assert jogador.possui_pecas([peca_2])

    jogador.remover_pecas([peca_2])

    assert jogador.esta_sem_peca()

    print("Todos os testes do jogador passaram.")