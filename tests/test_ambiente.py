import unittest
from copy import deepcopy

from src.agentes.aleatorio import AgenteAleatorio
from src.jogo.jogo import Jogo
from src.jogo.peca import Peca
from src.rl.ambiente import AmbienteRL


COMPRAR = {'tipo': 'comprar', 'ids_pecas': []}


def fabrica_controlada(maos, monte=()):
    def criar(semente):
        jogo = Jogo(semente=semente)
        for jogador, mao in zip(jogo.jogadores, maos):
            jogador.mao = list(mao)
            jogador.abriu = True  # Estes cenários exercitam turnos após a abertura.
        jogo.monte = list(monte)
        return jogo
    return criar


def grupo(prefixo=''):
    return [Peca(id=prefixo + str(i), cor=cor, numero=7)
            for i, cor in enumerate(('azul', 'verde', 'vermelho'))]


def criar_ambiente(**kwargs):
    return AmbienteRL(kwargs.pop('fabrica_jogo', lambda s: Jogo(semente=s)),
                      AgenteAleatorio(), **kwargs)


class TestAmbiente(unittest.TestCase):
    def test_parametros(self):
        for nome, valores in {'indice_agente': [-1, 2],
                              'numero_maximo': [0, 14],
                              'limite_passos': [0, -1]}.items():
            for valor in valores:
                with self.subTest(nome=nome, valor=valor):
                    with self.assertRaises(ValueError):
                        criar_ambiente(**{nome: valor})

    def test_reiniciar_reproduz_inclusive_adversario(self):
        ambiente = criar_ambiente(indice_agente=1)
        primeira = ambiente.reiniciar(42)
        jogo = deepcopy(ambiente.jogo.__dict__)
        resultado = ambiente.passo(COMPRAR)
        segunda = ambiente.reiniciar(42)
        self.assertEqual(primeira, segunda)
        for campo in ('jogadores', 'monte', 'mesa', 'indice_jogador_atual'):
            self.assertEqual(jogo[campo], ambiente.jogo.__dict__[campo])
        self.assertEqual(ambiente.passos, 0)
        self.assertFalse(ambiente.truncado)
        self.assertEqual(ambiente.passo(COMPRAR), resultado)

    def test_passo_exige_reinicio_e_turno_do_agente(self):
        ambiente = criar_ambiente()
        for operacao in (lambda: ambiente.passo(COMPRAR),
                         lambda: ambiente.executar_acao(COMPRAR),
                         ambiente.calcular_recompensa):
            with self.assertRaises(RuntimeError):
                operacao()
        ambiente.reiniciar(42)
        ambiente.jogo.indice_jogador_atual = 1
        with self.assertRaises(RuntimeError):
            ambiente.passo(COMPRAR)
        self.assertEqual(ambiente.passos, 0)

    def test_acao_invalida_nao_muda_estado(self):
        ambiente = criar_ambiente()
        ambiente.reiniciar(42)
        anterior = deepcopy(ambiente.jogo.__dict__)
        for acao in ({}, {'tipo': 'comprar', 'ids_pecas': ['0']},
                     {'tipo': 'baixar', 'ids_pecas': ['ausente']},
                     {'tipo': 'baixar', 'ids_pecas': [ambiente.jogo.jogador_atual.mao[0].id]}):
            with self.assertRaises(ValueError):
                ambiente.passo(acao)
            for campo in ('jogadores', 'monte', 'mesa', 'indice_jogador_atual'):
                self.assertEqual(anterior[campo], ambiente.jogo.__dict__[campo])
            self.assertEqual(ambiente.passos, 0)
        with self.assertRaises(ValueError):
            ambiente.executar_acao({'tipo': 'comprar', 'ids_pecas': '0'})

    def test_vitoria_nao_executa_adversario(self):
        pecas = grupo()
        ambiente = criar_ambiente(fabrica_jogo=fabrica_controlada([pecas, grupo('op')]))
        ambiente.reiniciar(0)
        resultado = ambiente.passo({'tipo': 'baixar', 'ids_pecas': [p.id for p in pecas]})
        self.assertEqual(resultado['recompensa'], 1)
        self.assertTrue(resultado['terminado'])
        self.assertFalse(resultado['truncado'])
        self.assertEqual(resultado['acoes_validas'], [])
        self.assertEqual(len(ambiente.jogo.jogadores[1].mao), 3)
        self.assertEqual(ambiente.passos, 1)
        with self.assertRaises(RuntimeError):
            ambiente.passo(COMPRAR)

    def test_derrota_compara_id_do_vencedor(self):
        ambiente = criar_ambiente()
        ambiente.reiniciar(0)
        ambiente.jogo.terminou = True
        ambiente.jogo.vencedor = deepcopy(ambiente.jogo.jogadores[1])
        self.assertEqual(ambiente.calcular_recompensa(), -1)
        ambiente.jogo.vencedor = deepcopy(ambiente.jogo.jogadores[0])
        self.assertEqual(ambiente.calcular_recompensa(), 1)

    def test_empate_por_passagens(self):
        ambiente = criar_ambiente(fabrica_jogo=fabrica_controlada([grupo()[:1], grupo('op')[:1]]))
        ambiente.reiniciar(0)
        resultado = ambiente.passo(COMPRAR)
        self.assertTrue(resultado['terminado'])
        self.assertEqual(resultado['recompensa'], 0)
        self.assertEqual(resultado['acoes_validas'], [])
        self.assertEqual(ambiente.jogo.passagens_consecutivas, 2)

    def test_truncamento_preserva_acoes_e_bloqueia_novo_passo(self):
        ambiente = criar_ambiente(limite_passos=1)
        ambiente.reiniciar(42)
        resultado = ambiente.passo(COMPRAR)
        self.assertTrue(resultado['truncado'])
        self.assertFalse(resultado['terminado'])
        self.assertTrue(resultado['acoes_validas'])
        self.assertEqual(resultado['recompensa'], 0)
        self.assertEqual(ambiente.passos, 1)
        self.assertEqual(ambiente.jogo.indice_jogador_atual, 0)
        with self.assertRaises(RuntimeError):
            ambiente.passo(COMPRAR)
        ambiente.reiniciar(42)
        self.assertFalse(ambiente.truncado)

    def test_fabrica_exige_primeira_decisao_e_dois_jogadores(self):
        def terminada(semente):
            jogo = Jogo(semente=semente)
            jogo.terminou = True
            return jogo
        for fabrica in (lambda s: Jogo(quantidade_jogadores=3, semente=s), terminada):
            with self.subTest(fabrica=fabrica):
                with self.assertRaises(ValueError):
                    criar_ambiente(fabrica_jogo=fabrica).reiniciar(0)
        fabrica = fabrica_controlada([grupo(), grupo('op')])
        for semente in range(20):
            try:
                criar_ambiente(fabrica_jogo=fabrica, indice_agente=1).reiniciar(semente)
            except ValueError:
                break
        else:
            self.fail('A fábrica que termina antes da decisão do agente deveria ser rejeitada.')


if __name__ == '__main__':
    unittest.main()
