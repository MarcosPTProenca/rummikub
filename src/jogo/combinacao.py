from pydantic import BaseModel, Field
from .peca import Peca

class Combinacao(BaseModel):

    pecas: list[Peca] = Field(default_factory=list)
    
    def eh_grupo(self) -> bool:
        normais = [peca for peca in self.pecas if peca.cor != "coringa"]
        return (
            3 <= len(self.pecas) <= 4
            and len({peca.id for peca in self.pecas}) == len(self.pecas)
            and len({peca.cor for peca in normais}) == len(normais)
            and len({peca.numero for peca in normais}) == 1
        )

    def eh_sequencia(self) -> bool:
        normais = [peca for peca in self.pecas if peca.cor != "coringa"]
        numeros = [peca.numero for peca in normais]
        return (
            3 <= len(self.pecas) <= 13
            and len({peca.id for peca in self.pecas}) == len(self.pecas)
            and len({peca.cor for peca in normais}) == 1
            and len(set(numeros)) == len(numeros)
            and max(numeros) - min(numeros) < len(self.pecas)
        )

    def eh_valido(self) -> bool:
        return self.eh_grupo() or self.eh_sequencia()

    def calcula_pontos(self) -> int:
        if not self.eh_valido():
            raise ValueError("Não é possível pontuar uma combinação inválida.")

        numeros = [peca.numero for peca in self.pecas if peca.cor != "coringa"]
        pontos = []
        if self.eh_grupo():
            pontos.append(numeros[0] * len(self.pecas))
        if self.eh_sequencia():
            # Coringas ambíguos assumem a combinação válida de maior pontuação.
            inicio = min(min(numeros), 14 - len(self.pecas))
            pontos.append(sum(range(inicio, inicio + len(self.pecas))))
        return max(pontos)

if __name__ == "__main__":
    grupo = Combinacao(pecas=[
        Peca(id="1", cor="azul", numero=7),
        Peca(id="2", cor="vermelho", numero=7),
        Peca(id="3", cor="verde", numero=7),
    ])

    sequencia = Combinacao(pecas=[
        Peca(id="4", cor="azul", numero=5),
        Peca(id="5", cor="azul", numero=3),
        Peca(id="6", cor="azul", numero=4),
    ])

    invalida = Combinacao(pecas=[
        Peca(id="7", cor="azul", numero=3),
        Peca(id="8", cor="azul", numero=5),
        Peca(id="9", cor="azul", numero=6),
    ])

    duplicada = Combinacao(pecas=[
        Peca(id="10", cor="amarelo", numero=3),
        Peca(id="11", cor="amarelo", numero=3),
        Peca(id="12", cor="amarelo", numero=4),
    ])

    assert grupo.eh_grupo()
    assert not grupo.eh_sequencia()
    assert grupo.eh_valido()
    assert grupo.calcula_pontos() == 21

    assert sequencia.eh_sequencia()
    assert not sequencia.eh_grupo()
    assert sequencia.eh_valido()
    assert sequencia.calcula_pontos() == 12

    assert not invalida.eh_valido()
    assert not duplicada.eh_valido()
    assert not Combinacao(pecas=[]).eh_valido()
    assert not Combinacao(pecas=grupo.pecas[:2]).eh_valido()

    try:
        invalida.calcula_pontos()
    except ValueError as erro:
        print(f"Erro esperado: {erro}")
    else:
        raise AssertionError("Uma combinação inválida deveria gerar ValueError.")

    print("Todos os testes passaram.")


