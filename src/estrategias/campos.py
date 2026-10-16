"""Campos de oponentes do torneio de mesa (plano 21).

Um campo é a lista dos oponentes de uma mesa, função só de (campo, n, semente). Como não
depende de quem está sendo avaliado, todo candidato enfrenta exatamente os mesmos oponentes
na mesma semente, e a diferença entre dois candidatos pode ser pareada pela mão. O campo
"misto" do artigo excluía o foco do sorteio, então mudava de uma linha da tabela para outra.
"""
import random

# As seis estratégias fortes: as que lideram pontos ou vitórias com duas pessoas.
POOL_FORTE = ["max_pecas", "max_pontos", "poupar_coringa", "poupar_descarta",
              "poupar_descarta_perto", "gastar_coringa"]
CAMPOS = ("iguais", "forte")


def oponentes_do_campo(campo: str, n: int, semente: int) -> list[str]:
    """Os n-1 oponentes: 'iguais' = cópias de max-tiles; 'forte' = sorteio com reposição entre as seis."""
    if campo == "iguais":
        return ["max_pecas"] * (n - 1)
    if campo == "forte":
        sorteio = random.Random(f"{campo}|{n}|{semente}")
        return sorteio.choices(POOL_FORTE, k=n - 1)
    raise ValueError(f"campo desconhecido: {campo!r} (use {' ou '.join(CAMPOS)})")
