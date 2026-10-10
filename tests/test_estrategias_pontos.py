import random
import unittest

from src.jogo.combinacao import Combinacao
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from src.estrategias.politicas import PONTOS, criar
from tests.test_turnos import pecas


def cenario(mao, mesa=(), monte=None, maos_oponentes=None):
    """Partida N=2 com mão, mesa e monte dados; os dois jogadores já abriram."""
    jogo = Jogo(semente=42)
    jogo.jogador_atual.mao = mao
    for jogador in jogo.jogadores:
        jogador.abriu = True
    if maos_oponentes is not None:
        jogo.jogadores[1].mao = maos_oponentes
    jogo.mesa = [Combinacao(pecas=list(c)) for c in mesa]
    if monte is not None:
        jogo.monte = list(monte)
    return jogo


def escolher(nome, jogo):
    politica = criar(nome, random.Random(0))
    return politica(jogo, jogo.listar_acoes_validas(True))


def coringa(id_="C"):
    return Peca(id=id_, cor="coringa", numero=0)


class TestEstrategiasPontos(unittest.TestCase):
    def setUp(self):
        # Mão: sequência azul 5-6 + coringa (fecha com o coringa) e um 9 solto.
        self.mao = [Peca(id="a5", cor="azul", numero=5), Peca(id="a6", cor="azul", numero=6),
                    coringa(), Peca(id="v9", cor="vermelho", numero=9)]

    def test_as_tres_estrategias_existem(self):
        self.assertEqual(set(PONTOS), {"poupar_descarta", "poupar_descarta_perto", "gastar_coringa"})

    def test_com_monte_cheio_poupar_descarta_age_como_o_saver(self):
        jogo = cenario(self.mao, monte=[Peca(id="m1", cor="verde", numero=1)] * 5)
        saver = escolher("poupar_coringa", jogo)
        descarta = escolher("poupar_descarta", cenario(self.mao, monte=[Peca(id="m1", cor="verde", numero=1)] * 5))
        self.assertEqual(saver["tipo"], descarta["tipo"])
        self.assertEqual(sorted(saver["ids_pecas"]), sorted(descarta["ids_pecas"]))

    def test_com_monte_vazio_poupar_descarta_gasta_o_coringa(self):
        jogo = cenario(self.mao, monte=[])
        saver = escolher("poupar_coringa", jogo)
        descarta = escolher("poupar_descarta", cenario(self.mao, monte=[]))
        self.assertEqual(sum(i == "C" for i in saver["ids_pecas"]), 0)
        self.assertEqual(sum(i == "C" for i in descarta["ids_pecas"]), 1)

    def test_oponente_perto_de_sair_faz_descartar_o_coringa(self):
        cheio = [Peca(id=f"m{i}", cor="verde", numero=1) for i in range(5)]
        longe = cenario(self.mao, monte=cheio, maos_oponentes=pecas([("azul", 1)] * 8, "o"))
        perto = cenario(self.mao, monte=cheio, maos_oponentes=pecas([("azul", 1)] * 3, "o"))
        self.assertEqual(sum(i == "C" for i in escolher("poupar_descarta_perto", longe)["ids_pecas"]), 0)
        self.assertEqual(sum(i == "C" for i in escolher("poupar_descarta_perto", perto)["ids_pecas"]), 1)

    def test_gastar_coringa_prefere_a_jogada_com_coringa_mesmo_sendo_menor(self):
        # Sequência azul 5-6-7 (3 peças, sem coringa) ou azul 5-6-coringa (3 peças, com).
        mao = [Peca(id="a5", cor="azul", numero=5), Peca(id="a6", cor="azul", numero=6),
               Peca(id="a7", cor="azul", numero=7), coringa()]
        jogo = cenario(mao, monte=[Peca(id="m1", cor="verde", numero=1)] * 5)
        acao = escolher("gastar_coringa", jogo)
        self.assertGreaterEqual(sum(i == "C" for i in acao["ids_pecas"]), 1)

    def test_todas_jogam_a_jogada_que_esvazia_a_mao(self):
        mao = [Peca(id="a5", cor="azul", numero=5), Peca(id="a6", cor="azul", numero=6), coringa()]
        for nome in PONTOS:
            jogo = cenario(mao, monte=[Peca(id="m1", cor="verde", numero=1)] * 5)
            acao = escolher(nome, jogo)
            self.assertEqual(len(acao["ids_pecas"]), 3, nome)

    def test_sem_jogada_todas_compram(self):
        for nome in PONTOS:
            jogo = cenario(pecas([("azul", 2), ("verde", 5)], "s"), monte=[Peca(id="m", cor="verde", numero=1)])
            self.assertEqual(escolher(nome, jogo)["tipo"], "comprar", nome)


if __name__ == "__main__":
    unittest.main()
