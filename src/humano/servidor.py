"""Servidor local da mesa humana: HTML estático e quatro rotas de estado.

Escuta apenas em 127.0.0.1 e não tem autenticação, porque serve um jogo de uma
pessoa na própria máquina; não exponha a porta na rede. Usa só a biblioteca
padrão: um jogo local não justifica dependência de framework web.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.humano.mesa import OPONENTES, Partida

PAGINA = Path(__file__).parent / "mesa.html"


def _lista_oponentes() -> list[dict]:
    """Heurísticas do mais forte ao mais fraco, pela taxa de vitória do torneio; depois as redes, que
    trazem uma dica no lugar da taxa porque a taxa do torneio não é a métrica delas."""
    return sorted(({"id": k, **v} for k, v in OPONENTES.items()),
                  key=lambda o: (bool(o.get("rede")), -(o["vitorias"] or 0), o["nome"]))


def _combinacoes_validas(corpo) -> list[list[str]] | None:
    """A entrada vem da rede: aceita só lista não vazia de listas de strings."""
    c = corpo.get("combinacoes") if isinstance(corpo, dict) else None
    if not isinstance(c, list) or not c:
        return None
    if not all(isinstance(g, list) and g and all(isinstance(i, str) for i in g) for g in c):
        return None
    return c


class Manipulador(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silencia o log por requisição
        pass

    # --- utilidades ----------------------------------------------------

    def _responder(self, status: int, corpo: bytes, tipo: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        # O estado muda a cada lance e o HTML muda a cada edição: servir do cache
        # mostra uma mesa que não existe mais, ou a interface antiga.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def _json(self, dados, status: int = 200) -> None:
        self._responder(status, json.dumps(dados).encode(), "application/json; charset=utf-8")

    def _erro(self, mensagem: str) -> None:
        self._json({"erro": mensagem}, 400)

    def _corpo(self):
        tamanho = int(self.headers.get("Content-Length") or 0)
        if not tamanho:
            return {}
        try:
            return json.loads(self.rfile.read(tamanho))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # --- rotas ---------------------------------------------------------

    def do_GET(self):
        if self.path == "/":
            self._responder(200, PAGINA.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/estado":
            self._json(self.server.partida.estado())
        elif self.path == "/oponentes":
            self._json(_lista_oponentes())
        else:
            self._json({"erro": "Rota desconhecida."}, 404)

    def do_POST(self):
        corpo = self._corpo()
        if corpo is None:
            return self._erro("Corpo não é JSON válido.")
        # Uma thread por conexão e uma única partida compartilhada: sem o lock,
        # dois POSTs simultâneos mutariam o mesmo jogo no meio de um turno.
        with self.server.trava:
            self._post(corpo)

    def _post(self, corpo):
        if self.path == "/jogar":
            combinacoes = _combinacoes_validas(corpo)
            if combinacoes is None:
                return self._erro("Informe combinações como listas não vazias de IDs.")
            r = self.server.partida.jogar(combinacoes)
            return self._json(self.server.partida.estado()) if r["ok"] else self._erro(r["erro"])
        if self.path == "/comprar":
            r = self.server.partida.comprar()
            return self._json(self.server.partida.estado()) if r["ok"] else self._erro(r["erro"])
        if self.path == "/nova":
            # Lista vazia é erro do cliente, não pedido de padrão: `or` aqui
            # esconderia o engano atrás de uma partida que ninguém pediu.
            oponentes = corpo.get("oponentes")
            if oponentes is None:
                oponentes = ["poupar_coringa"]
            if not isinstance(oponentes, list) or not all(isinstance(o, str) for o in oponentes):
                return self._erro("Informe os oponentes como lista de nomes.")
            semente = corpo.get("semente")
            if semente is not None and not isinstance(semente, int):
                return self._erro("A semente deve ser um inteiro.")
            try:
                self.server.partida = Partida(oponentes, semente=semente,
                                              registro=self.server.registro)
            except ValueError as e:
                return self._erro(str(e))
            return self._json(self.server.partida.estado())
        self._json({"erro": "Rota desconhecida."}, 404)


def criar_servidor(porta: int = 8000, oponentes: list[str] | None = None,
                   semente: int | None = None, registro=None) -> ThreadingHTTPServer:
    # Com HTTP/1.1 o navegador deixa conexões abertas em keep-alive; um servidor
    # de uma única thread fica preso na conexão ociosa e trava a página.
    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Manipulador)
    servidor.trava = threading.Lock()
    servidor.registro = registro
    servidor.partida = Partida(oponentes or ["poupar_coringa"], semente=semente, registro=registro)
    return servidor
