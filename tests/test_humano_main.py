"""Teste do ponto de entrada `python -m src.humano`."""
import unittest
from unittest.mock import patch

from src.humano.__main__ import main


class TestEntrada(unittest.TestCase):
    def test_monta_o_servidor_com_os_argumentos_e_abre_o_navegador(self):
        with patch("src.humano.__main__.criar_servidor") as criar, \
             patch("src.humano.__main__.webbrowser.open") as abrir:
            criar.return_value.server_address = ("127.0.0.1", 8123)
            criar.return_value.serve_forever.side_effect = KeyboardInterrupt
            main(["--porta", "8123", "--oponente", "cauteloso", "--semente", "7"])
        self.assertEqual(criar.call_args.kwargs["porta"], 8123)
        self.assertEqual(criar.call_args.kwargs["oponentes"], ["cauteloso"])
        self.assertEqual(criar.call_args.kwargs["semente"], 7)
        abrir.assert_called_once_with("http://127.0.0.1:8123/")

    def test_grava_as_partidas_por_padrao(self):
        with patch("src.humano.__main__.criar_servidor") as criar, \
             patch("src.humano.__main__.webbrowser.open"):
            criar.return_value.server_address = ("127.0.0.1", 8000)
            criar.return_value.serve_forever.side_effect = KeyboardInterrupt
            main([])
        self.assertEqual(criar.call_args.kwargs["registro"].name, "humano.jsonl")

    def test_sem_registro_desliga_a_gravacao(self):
        with patch("src.humano.__main__.criar_servidor") as criar, \
             patch("src.humano.__main__.webbrowser.open"):
            criar.return_value.server_address = ("127.0.0.1", 8000)
            criar.return_value.serve_forever.side_effect = KeyboardInterrupt
            main(["--sem-registro"])
        self.assertIsNone(criar.call_args.kwargs["registro"])

    def test_oponente_invalido_sai_com_erro(self):
        with self.assertRaises(SystemExit):
            main(["--oponente", "nao_existe"])


if __name__ == "__main__":
    unittest.main()
