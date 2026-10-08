# Rummikub: which strategy wins, and why?

A from-scratch Rummikub simulator (official rules: 106 tiles, two jokers, 14-tile hands, 30-point opening meld, several melds per turn, extending melds, table rearrangement) and a simulation study of heuristics, reinforcement learning, search and an LLM decision-maker.

The paper is in [`artigo/main.pdf`](artigo/main.pdf) (LaTeX source in [`artigo/main.tex`](artigo/main.tex)).

<p align="center">
  <img src="docs/evolucao.gif" alt="A policy network learning to play Rummikub: iteration 0, 15, 25 and 1500" width="320"><br>
  <sub>A small policy network, from random play to its trained form (full video: <a href="docs/evolucao.mp4">docs/evolucao.mp4</a>).
  The network only relearned "play as many tiles as possible"; it did not discover anything new.</sub>
</p>

## Findings

All results are conditional on this simulator and this strategy menu; "best" means best in this benchmark, not optimal Rummikub play.

- **Winner: the joker-saver** (plays as many tiles as possible, but spends a joker only if the move empties the hand). Mean score 0.726 ± 0.027 against the field in a paired round-robin of seven strategies; max-tiles scores 0.676.
- **Why:** it wins by holding a wildcard until it can go out (71% of its wins end with a joker, against 4% for the control). Four ablations isolate this: a soft joker penalty does not reproduce the effect, forbidding joker removal from the table does not remove it, and forbidding the joker even as the final tile destroys the advantage.
- **Robustness:** the ranking survives the official pool-exhaustion tie-break, a larger rearrangement-search cap and a 12-strategy league. At three and four players the saver's edge over max-tiles shrinks and then vanishes.
- **Learning does not beat it:** a tabular strategy selector, the Jev System One model as selector or move chooser, determinised Monte Carlo search, and a small policy network trained by REINFORCE all fail to beat the best heuristics. Cloning the saver and refining with group-relative advantages reaches parity (at best 0.550 ± 0.040 against max-tiles and 0.495 ± 0.040 against the saver, exploratory). Reward shaping, a wider league, a residual network over the saver and a search that sees hidden hands add nothing.

Negative results are reported as found; see the paper's Limitations section for what was not tested.

## Layout

| Path | Contents |
|---|---|
| `src/jogo/` | rules engine (tiles, melds, turns, table rearrangement) |
| `src/estrategias/` | strategies, policy network (pure Python), cloning, PIMC search, Jev agents, tournament arena |
| `src/experimentos.py` | command-line entry point for every experiment (`torneio`, `liga`, `mecanismo`, `limites`, `mesa`, `busca`, `rede`, `clonar`, `jev`, `robustez`, ...) |
| `src/analise*.py` | tables and figures for the paper |
| `src/animacao/` | Manim animations of games and of the learning run |
| `tests/` | unit and end-to-end tests |
| `resultados/` | per-game JSONL records and training logs behind every table (no API keys) |
| `artigo/` | paper: LaTeX source, tables, figures, PDF |

Identifiers and CLI flags are in Portuguese.

## Running

Developed on Python 3.14. The core has a single dependency, `pydantic`.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests          # full test suite

# paired tournament of the seven strategies
.venv/bin/python -m src.experimentos torneio --help

# train the policy network (REINFORCE; also --algoritmo dmc | ppo)
.venv/bin/python -m src.experimentos rede --iteracoes 300 --jogos 24 --processos 4 --saida resultados/minha_rede
```

Tables and figures need `matplotlib` and `numpy` (`pip install -r requirements-analise.txt`), then `python -m src.analise`.
The animations need Manim and ffmpeg and are not part of the core.

The Jev experiments call a paid external API and need `TYPESAFE_API_KEY` in the environment (or in a `.env` file passed with `--env`). Everything else runs offline. Cached results for the Jev runs are in `resultados/`.

## Reproducibility notes

- Evaluations are paired (same deal, seats alternated) with separate seed ranges per experiment; confidence intervals are over games.
- Experiments are labelled exploratory or confirmatory in the paper; only the sustained claims are in the abstract.

## Author

Marcos de Pinho Tavares Proença.
