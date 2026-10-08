ROTULOS = {
    "aleatorio": "Random", "so_baixar": "Play-only", "minimo": "Minimum", "cauteloso": "Cautious",
    "max_pontos": "Max points", "max_pecas": "Max tiles", "poupar_coringa": "Joker saver",
    "rl_meta": "Q-learning meta", "jev_meta": "Jev strategy picker", "jev_jogadas": "Jev move picker",
}


def rotulo(nome: str) -> str:
    if nome.startswith("pimc"):
        return "PIMC search"
    if nome.startswith("rede"):
        return "Trained network"
    return ROTULOS.get(nome, nome)
