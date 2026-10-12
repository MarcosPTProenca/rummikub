"""Testes do servidor local da mesa humana.

O servidor é deliberadamente mínimo: serve um HTML estático e quatro rotas de
estado. Os testes cobrem o contrato que o frontend consome e a recusa de
entrada malformada, que vem da rede e não é confiável.
"""
import json
import unittest
from http.client import HTTPConnection
from threading import Thread

from src.humano.servidor import criar_servidor


class TestServidor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servidor = criar_servidor(porta=0, oponentes=["max_pecas"], semente=1, registro=None)
        cls.porta = cls.servidor.server_address[1]
        cls.thread = Thread(target=cls.servidor.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.servidor.shutdown()
        cls.servidor.server_close()

    def pedir(self, metodo, caminho, corpo=None):
        c = HTTPConnection("127.0.0.1", self.porta, timeout=10)
        dados = json.dumps(corpo) if corpo is not None else None
        c.request(metodo, caminho, body=dados, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        texto = r.read().decode()
        c.close()
        return r.status, texto

    def setUp(self):
        # O servidor guarda uma partida; sem reiniciar, um teste herda o estado do anterior.
        self.pedir("POST", "/nova", {"oponentes": ["max_pecas"], "semente": 1})

    def test_raiz_serve_o_html_da_mesa(self):
        status, corpo = self.pedir("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("<!doctype html>", corpo.lower())

    def test_estado_devolve_json_com_a_mao_e_a_mesa(self):
        status, corpo = self.pedir("GET", "/estado")
        self.assertEqual(status, 200)
        e = json.loads(corpo)
        self.assertEqual(len(e["mao"]), 14)
        self.assertIn("mesa", e)
        self.assertIn("oponentes", e)

    def test_oponentes_lista_as_heuristicas_disponiveis(self):
        status, corpo = self.pedir("GET", "/oponentes")
        self.assertEqual(status, 200)
        lista = json.loads(corpo)
        self.assertGreaterEqual(len(lista), 7)  # 7 heurísticas, mais a rede se estiver no disco
        self.assertEqual(set(lista[0]), {"id", "nome", "vitorias", "sobre"})
        # Mais forte primeiro, para a interface listar por dificuldade.
        self.assertEqual(lista[0]["id"], "poupar_coringa")

    def test_jogada_invalida_responde_400_com_a_razao(self):
        status, corpo = self.pedir("POST", "/jogar", {"combinacoes": [["0", "1", "2"]]})
        self.assertEqual(status, 400)
        self.assertIn("erro", json.loads(corpo))

    def test_corpo_que_nao_e_json_responde_400(self):
        c = HTTPConnection("127.0.0.1", self.porta, timeout=10)
        c.request("POST", "/jogar", body="nao e json", headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)
        c.close()

    def test_combinacoes_com_tipo_errado_responde_400(self):
        for corpo in ({"combinacoes": "abc"}, {"combinacoes": [1, 2]}, {}, {"combinacoes": [["a", 1]]}):
            status, _ = self.pedir("POST", "/jogar", corpo)
            self.assertEqual(status, 400, corpo)

    def test_rota_desconhecida_responde_404(self):
        status, _ = self.pedir("GET", "/nao-existe")
        self.assertEqual(status, 404)

    def test_dica_nao_e_exposta(self):
        """Jogar contra o agente perde o sentido se a interface entrega a jogada dele."""
        status, _ = self.pedir("GET", "/dica")
        self.assertEqual(status, 404)

    def test_comprar_avanca_a_partida(self):
        antes = json.loads(self.pedir("GET", "/estado")[1])
        status, _ = self.pedir("POST", "/comprar", {})
        self.assertEqual(status, 200)
        depois = json.loads(self.pedir("GET", "/estado")[1])
        self.assertNotEqual(antes["decisoes"], depois["decisoes"])

    def test_nova_partida_reinicia_com_outros_oponentes(self):
        status, corpo = self.pedir("POST", "/nova", {"oponentes": ["cauteloso"], "semente": 5})
        self.assertEqual(status, 200)
        e = json.loads(self.pedir("GET", "/estado")[1])
        self.assertEqual([o["nome"] for o in e["oponentes"]], ["cautious"])
        self.assertEqual(len(e["mao"]), 14)

    def test_nova_partida_com_varios_oponentes(self):
        status, corpo = self.pedir("POST", "/nova",
                                   {"oponentes": ["max_pecas", "cauteloso", "minimo"], "semente": 9})
        self.assertEqual(status, 200)
        e = json.loads(corpo)
        self.assertEqual([o["nome"] for o in e["oponentes"]], ["max-tiles", "cautious", "min-play"])
        self.assertEqual(len(e["mao"]), 14)

    def test_nova_partida_com_oponentes_repetidos(self):
        """Três cópias da mesma estratégia é uma mesa legítima."""
        status, corpo = self.pedir("POST", "/nova",
                                   {"oponentes": ["max_pecas", "max_pecas"], "semente": 9})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(corpo)["oponentes"]), 2)

    def test_nova_partida_com_oponentes_demais_responde_400(self):
        status, _ = self.pedir("POST", "/nova", {"oponentes": ["max_pecas"] * 10})
        self.assertEqual(status, 400)

    def test_nova_partida_sem_oponente_responde_400(self):
        status, _ = self.pedir("POST", "/nova", {"oponentes": []})
        self.assertEqual(status, 400)

    def test_nova_partida_com_oponente_invalido_responde_400(self):
        status, _ = self.pedir("POST", "/nova", {"oponentes": ["drop table"]})
        self.assertEqual(status, 400)

    def test_requisicoes_concorrentes_nao_corrompem_a_partida(self):
        """Várias threads sobre um mesmo jogo: o total de peças é invariante."""
        from concurrent.futures import ThreadPoolExecutor
        self.pedir("POST", "/nova", {"oponentes": ["max_pecas"], "semente": 3})
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: self.pedir("POST", "/comprar", {}), range(8)))
        e = json.loads(self.pedir("GET", "/estado")[1])
        total = (len(e["mao"]) + sum(len(c) for c in e["mesa"])
                 + sum(o["pecas"] for o in e["oponentes"]) + e["monte"])
        self.assertEqual(total, 106)

    def test_respostas_proibem_cache(self):
        """Sem no-store o navegador pode servir /estado velho, ou o HTML antigo apos uma edicao."""
        for metodo, rota in (("GET", "/"), ("GET", "/estado"), ("GET", "/oponentes")):
            c = HTTPConnection("127.0.0.1", self.porta, timeout=10)
            c.request(metodo, rota)
            r = c.getresponse()
            r.read()
            self.assertEqual(r.getheader("Cache-Control"), "no-store", rota)
            c.close()

    def test_servidor_so_escuta_em_localhost(self):
        self.assertEqual(self.servidor.server_address[0], "127.0.0.1")

    def test_atende_em_paralelo_com_conexao_ociosa_aberta(self):
        """Com HTTP/1.1 o navegador mantém conexões vivas; servidor de uma thread travaria."""
        ociosa = HTTPConnection("127.0.0.1", self.porta, timeout=10)
        ociosa.request("GET", "/estado")
        ociosa.getresponse().read()  # resposta lida, conexão deixada aberta de propósito
        try:
            status, _ = self.pedir("GET", "/estado")
            self.assertEqual(status, 200)
        finally:
            ociosa.close()

    def test_varias_conexoes_simultaneas_sao_atendidas(self):
        conexoes = []
        try:
            for _ in range(5):
                c = HTTPConnection("127.0.0.1", self.porta, timeout=10)
                c.request("GET", "/estado")
                c.getresponse().read()
                conexoes.append(c)
            status, _ = self.pedir("GET", "/oponentes")
            self.assertEqual(status, 200)
        finally:
            for c in conexoes:
                c.close()


if __name__ == "__main__":
    unittest.main()
