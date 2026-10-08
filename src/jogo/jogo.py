import random
from itertools import combinations, product
from .peca import Peca
from .jogador import Jogador
from .combinacao import Combinacao


class Jogo:
    def __init__(
        self,
        quantidade_jogadores: int = 2,
        pecas_por_jogador: int = 14,
        semente: int | None = None,
        desempate: str = "pontos",
    ):
        if desempate not in ("pontos", "pecas", "empate"):
            raise ValueError("Desempate deve ser 'pontos', 'pecas' ou 'empate'.")
        self.desempate = desempate

        if not 2 <= quantidade_jogadores <= 4:
            raise ValueError("A partida deve ter entre 2 e 4 jogadores.")

        if not 1 <= pecas_por_jogador <= 106 // quantidade_jogadores:
            raise ValueError("Quantidade inicial de peças inválida.")

        self.aleatorio = random.Random(semente)

        self.jogadores = [
            Jogador(id=str(i))
            for i in range(quantidade_jogadores)
        ]

        self.monte: list[Peca] = []
        self.mesa: list[Combinacao] = []

        self.indice_jogador_atual = 0
        self.passagens_consecutivas = 0
        self.terminou = False
        self.vencedor: Jogador | None = None

        self.criar_pecas()
        self.aleatorio.shuffle(self.monte)
        self.distribuir_pecas(pecas_por_jogador)

    @property
    def jogador_atual(self) -> Jogador:
        return self.jogadores[self.indice_jogador_atual]

    def criar_pecas(self) -> None:
        cores = ["azul", "vermelho", "verde", "amarelo"]
        pecas = []

        for cor in cores:
            for numero in range(1, 14):
                for _ in range(2):
                    pecas.append(
                        Peca(
                            id=str(len(pecas)),
                            cor=cor,
                            numero=numero,
                        )
                    )

        for _ in range(2):
            pecas.append(Peca(id=str(len(pecas)), cor="coringa", numero=0))

        self.monte = pecas

    def distribuir_pecas(self, quantidade: int) -> None:
        for _ in range(quantidade):
            for jogador in self.jogadores:
                jogador.receber_peca(self.monte.pop())

    def verificar_partida_ativa(self) -> None:
        if self.terminou:
            raise ValueError("A partida já terminou.")

    def avancar_turno(self) -> None:
        self.indice_jogador_atual = (
            self.indice_jogador_atual + 1
        ) % len(self.jogadores)

    def _vencedor_por_desempate(self) -> Jogador | None:
        """Monte esgotado: vence a menor mão (pontos, coringa = 30, ou nº de peças)."""
        if self.desempate == "empate":
            return None
        if self.desempate == "pecas":
            valor = len
        else:
            def valor(mao):
                return sum(30 if p.cor == "coringa" else p.numero for p in mao)
        valores = [valor(j.mao) for j in self.jogadores]
        menor = min(valores)
        return self.jogadores[valores.index(menor)] if valores.count(menor) == 1 else None

    def comprar_peca(self) -> Peca | None:
        self.verificar_partida_ativa()

        if not self.monte:
            self.passagens_consecutivas += 1

            if self.passagens_consecutivas == len(self.jogadores):
                self.terminou = True
                self.vencedor = self._vencedor_por_desempate()
            else:
                self.avancar_turno()

            return None

        peca = self.monte.pop()
        self.jogador_atual.receber_peca(peca)

        self.passagens_consecutivas = 0
        self.avancar_turno()

        return peca

    def jogar_turno(self, combinacoes: list[list[str]]) -> None:
        """Aplicar a mesa final inteira, atomicamente, e encerrar um único turno."""
        self.verificar_partida_ativa()
        if not isinstance(combinacoes, list) or not combinacoes or not all(
            isinstance(c, list) and c and all(isinstance(i, str) for i in c)
            for c in combinacoes
        ):
            raise ValueError("Informe combinações como listas não vazias de IDs.")
        selecionados = [i for c in combinacoes for i in c]
        if len(selecionados) != len(set(selecionados)):
            raise ValueError("Uma peça não pode ser selecionada duas vezes.")
        mao = {p.id: p for p in self.jogador_atual.mao}
        mesa = {p.id: p for c in self.mesa for p in c.pecas}
        usados = set(selecionados)
        if not usados.issubset(mao.keys() | mesa.keys()) or not mesa.keys() <= usados:
            raise ValueError("A mesa deve conservar suas peças e usar apenas peças da mão.")
        novas = usados & mao.keys()
        if not novas:
            raise ValueError("O turno deve baixar ao menos uma peça da mão.")
        por_id = mesa | mao
        proposta = [Combinacao(pecas=[por_id[i] for i in c]) for c in combinacoes]
        if not all(c.eh_valido() for c in proposta):
            raise ValueError("Todas as combinações da mesa final devem ser válidas.")
        if not self.jogador_atual.abriu:
            conjuntos_anteriores = {frozenset(p.id for p in c.pecas) for c in self.mesa}
            conjuntos_finais = {frozenset(c) for c in combinacoes}
            if not conjuntos_anteriores <= conjuntos_finais:
                raise ValueError("Na abertura não é permitido manipular a mesa.")
            pontos = sum(c.calcula_pontos() for c in proposta if c.pecas[0].id in mao)
            if pontos < 30:
                raise ValueError("A abertura deve baixar pelo menos 30 pontos da própria mão.")
        jogador = self.jogador_atual
        jogador.remover_pecas([mao[i] for i in novas])
        jogador.abriu = True
        self.mesa = proposta
        self.passagens_consecutivas = 0
        if jogador.esta_sem_peca():
            self.vencedor = jogador
            self.terminou = True
        else:
            self.avancar_turno()

    def baixar_combinacoes(self, combinacoes: list[list[str]]) -> None:
        """Baixar várias combinações novas no mesmo turno, inclusive na abertura."""
        if not isinstance(combinacoes, list) or not all(isinstance(c, list) for c in combinacoes):
            raise ValueError("Informe uma lista de combinações.")
        mao = {p.id for p in self.jogador_atual.mao}
        if any(not isinstance(i, str) or i not in mao for c in combinacoes for i in c):
            raise ValueError("As peças selecionadas não estão na mão.")
        self.jogar_turno([[p.id for p in c.pecas] for c in self.mesa] + combinacoes)

    def baixar_combinacao(self, ids_pecas: list[str]) -> None:
        self.baixar_combinacoes([ids_pecas])

    def acrescentar_pecas(self, indice_combinacao: int, ids_pecas: list[str]) -> None:
        if not self.jogador_atual.abriu:
            raise ValueError("Faça a abertura antes de acrescentar peças à mesa.")
        if type(indice_combinacao) is not int or not 0 <= indice_combinacao < len(self.mesa):
            raise ValueError("Índice de combinação inválido.")
        if not isinstance(ids_pecas, list):
            raise ValueError("Informe uma lista de IDs.")
        proposta = [[p.id for p in c.pecas] for c in self.mesa]
        proposta[indice_combinacao] += ids_pecas
        self.jogar_turno(proposta)

    @staticmethod
    def _listar_baixas(pecas: list[Peca]) -> list[dict]:
        acoes = {}
        coringas = []
        por_numero: dict[int, dict[str, list[Peca]]] = {}
        por_cor: dict[str, dict[int, list[Peca]]] = {}

        for peca in pecas:
            if peca.cor == "coringa":
                coringas.append(peca)
                continue
            por_numero.setdefault(peca.numero, {}).setdefault(peca.cor, []).append(peca)
            por_cor.setdefault(peca.cor, {}).setdefault(peca.numero, []).append(peca)

        # Grupos: coringas ocupam as cores ausentes, sem repetir cores normais.
        for pecas_por_cor in por_numero.values():
            cores = sorted(pecas_por_cor)
            for tamanho in (3, 4):
                for quantidade in range(min(len(coringas), tamanho - 1) + 1):
                    for cores_escolhidas in combinations(cores, tamanho - quantidade):
                        opcoes = [pecas_por_cor[cor] for cor in cores_escolhidas]
                        for normais in product(*opcoes):
                            for escolhidos in combinations(coringas, quantidade):
                                ids = [peca.id for peca in normais + escolhidos]
                                acoes.setdefault(tuple(sorted(ids)), {"tipo": "baixar", "ids_pecas": ids})

        # Sequências: preencher lacunas ou substituir números presentes na mão.
        for pecas_por_numero in por_cor.values():
            for inicio in (range(1, 12) if coringas else sorted(pecas_por_numero)):
                for fim in range(inicio + 2, 14):
                    numeros = range(inicio, fim + 1)
                    presentes = [n for n in numeros if n in pecas_por_numero]
                    faltantes = len(numeros) - len(presentes)
                    if faltantes > len(coringas):
                        break
                    for quantidade in range(faltantes, min(len(coringas), len(numeros) - 1) + 1):
                        for substituidos in combinations(presentes, quantidade - faltantes):
                            opcoes = [pecas_por_numero[n] for n in presentes if n not in substituidos]
                            for normais in product(*opcoes):
                                for escolhidos in combinations(coringas, quantidade):
                                    ids = [peca.id for peca in normais + escolhidos]
                                    acoes.setdefault(tuple(sorted(ids)), {"tipo": "baixar", "ids_pecas": ids})

        # Diferentes posições dos coringas podem produzir a mesma seleção física.
        return list(acoes.values())

    def listar_acoes_validas(self, rearranjar: bool = True, limites: dict | None = None,
                             saturacao: dict | None = None) -> list[dict]:
        if self.terminou:
            return []
        # limites: planos, visitas, pecas (rearranjo), nos (cobertura), conjuntos; saturacao: quais cortes foram atingidos.
        lim = {"planos": 128, "visitas": 1000, "pecas": 18, "nos": 256, "conjuntos": 2, **(limites or {})}
        sat = saturacao if saturacao is not None else {}
        sat.update(planos=False, visitas=False, pecas=False, nos=False)
        mao = self.jogador_atual.mao
        por_id = {p.id: p for p in mao}
        mesa = [[p.id for p in c.pecas] for c in self.mesa]
        baixas = self._listar_baixas(mao)
        pontos = {tuple(a["ids_pecas"]): Combinacao(pecas=[por_id[i] for i in a["ids_pecas"]]).calcula_pontos()
                  for a in baixas}
        abriu = self.jogador_atual.abriu
        acoes = [a for a in baixas if abriu or pontos[tuple(a["ids_pecas"])] >= 30]
        vistos = {tuple(sorted(tuple(sorted(c)) for c in mesa + [a["ids_pecas"]])) for a in acoes}
        planos = []
        # ponytail: RL amostra 128 planos e rearranjos de até 2 conjuntos/18 peças;
        # jogar_turno aceita a mesa inteira. Busca global quando o agente precisar.
        def adicionar(layout):
            chave = tuple(sorted(tuple(sorted(c)) for c in layout))
            novos = sorted(set(i for c in layout for i in c) & por_id.keys())
            if novos and chave not in vistos:
                if len(planos) >= lim["planos"]:
                    sat["planos"] = True
                    return
                vistos.add(chave)
                planos.append({"tipo": "mesa", "ids_pecas": novos,
                               "combinacoes": [list(c) for c in layout]})

        ordenadas = sorted(baixas, key=lambda a: (-len(a["ids_pecas"]), -pontos[tuple(a["ids_pecas"])], a["ids_pecas"]))
        visitas = 0
        def combinar(inicio, escolhidas, usados, soma):
            nonlocal visitas
            visitas += 1
            if visitas > lim["visitas"]:
                sat["visitas"] = True
                return
            if len(planos) >= lim["planos"]:
                sat["planos"] = True
                return
            if len(escolhidas) >= 2 and (abriu or soma >= 30):
                adicionar(mesa + escolhidas)
            for j in range(inicio, len(ordenadas)):
                selecionados = ordenadas[j]["ids_pecas"]
                if not usados.intersection(selecionados):
                    combinar(j + 1, escolhidas + [selecionados], usados | set(selecionados),
                             soma + pontos[tuple(selecionados)])
                if visitas > lim["visitas"] or len(planos) >= lim["planos"]:
                    break
        combinar(0, [], set(), 0)

        if abriu and mesa:
            # Um plano guloso também permite vários encaixes e novas baixas juntos.
            layout = [list(c.pecas) for c in self.mesa]
            restantes = list(mao)
            mudou = True
            while mudou:
                mudou = False
                for p in list(restantes):
                    for c in layout:
                        if Combinacao(pecas=c + [p]).eh_valido():
                            c.append(p)
                            restantes.remove(p)
                            mudou = True
                            break
            livres = {p.id for p in restantes}
            for a in self._listar_baixas(restantes):
                if set(a["ids_pecas"]) <= livres:
                    layout.append([por_id[i] for i in a["ids_pecas"]])
                    livres.difference_update(a["ids_pecas"])
            adicionar([[p.id for p in c] for c in layout])
            for indice, c in enumerate(self.mesa):
                for p in mao:
                    if Combinacao(pecas=c.pecas + [p]).eh_valido():
                        proposta = [list(ids) for ids in mesa]
                        proposta[indice].append(p.id)
                        adicionar(proposta)

            for quantidade in (range(1, lim["conjuntos"] + 1) if rearranjar else ()):
                for indices in combinations(range(len(self.mesa)), quantidade):
                    if len(planos) >= lim["planos"]:
                        sat["planos"] = True
                        break
                    locais = [p for j in indices for p in self.mesa[j].pecas]
                    if len(locais) > lim["pecas"]:
                        sat["pecas"] = True
                        continue
                    ids_locais = {p.id for p in locais}
                    candidatas = self._listar_baixas(locais + mao)
                    antigas = [a["ids_pecas"] for a in candidatas if set(a["ids_pecas"]) <= ids_locais]
                    por_peca = {i: [c for c in antigas if i in c] for i in ids_locais}
                    falhas = set()
                    nos = 0
                    def cobrir(resto):
                        nonlocal nos
                        if not resto:
                            return []
                        nos += 1
                        if nos > lim["nos"]:
                            sat["nos"] = True
                            return None
                        if resto in falhas:
                            return None
                        i = min(resto, key=lambda i: (sum(set(c) <= resto for c in por_peca[i]), i))
                        for c in por_peca[i]:
                            if set(c) <= resto:
                                cauda = cobrir(resto - set(c))
                                if cauda is not None:
                                    return [c] + cauda
                        falhas.add(resto)
                        return None
                    candidatas.sort(key=lambda a: (-len(set(a["ids_pecas"]) & por_id.keys()), -len(a["ids_pecas"])))
                    for a in candidatas:
                        usados = set(a["ids_pecas"])
                        if not usados & por_id.keys() or not usados & ids_locais:
                            continue
                        cobertura = cobrir(frozenset(ids_locais - usados))
                        if cobertura is not None:
                            outras = [c for j, c in enumerate(mesa) if j not in indices]
                            adicionar(outras + [a["ids_pecas"]] + cobertura)
                        if len(planos) >= lim["planos"]:
                            sat["planos"] = True
                            break
        return acoes + planos + [{"tipo": "comprar", "ids_pecas": []}]

if __name__ == "__main__":
    jogo = Jogo(semente=42)

    assert len(jogo.jogadores) == 2
    assert all(len(jogador.mao) == 14 for jogador in jogo.jogadores)
    assert len(jogo.monte) == 78

    todas_as_pecas = jogo.monte + [
        peca
        for jogador in jogo.jogadores
        for peca in jogador.mao
    ]

    assert len({peca.id for peca in todas_as_pecas}) == 106

    jogador_que_comprou = jogo.jogador_atual
    peca_comprada = jogo.comprar_peca()

    assert peca_comprada is not None
    assert jogador_que_comprou.possui_pecas([peca_comprada])
    assert len(jogador_que_comprou.mao) == 15
    assert len(jogo.monte) == 77
    assert jogo.jogador_atual.id == "1"

    print("Testes de inicialização e compra passaram.")

    for jogador in jogo.jogadores:
        print(f"\nJogador {jogador.id}:")

        for peca in jogador.mao:
            print(f"  ID {peca.id}: {peca.cor} {peca.numero}")

    acoes = jogo.listar_acoes_validas()

    for acao in acoes:
        print(acao)

    # Toda ação de baixar encontrada deve formar uma combinação válida.
    pecas_por_id = {
        peca.id: peca
        for peca in jogo.jogador_atual.mao
    }

    for acao in acoes:
        if acao["tipo"] == "baixar":
            combinacao = Combinacao(
                pecas=[
                    pecas_por_id[id_peca]
                    for id_peca in acao["ids_pecas"]
                ]
            )

            assert combinacao.eh_valido()

    assert acoes[-1]["tipo"] == "comprar"

    print("Todas as combinações encontradas são válidas.")