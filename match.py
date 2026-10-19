#!/usr/bin/env python
"""Play engine-vs-engine matches and estimate the Elo difference.

Players:
    net:<checkpoint>:<nodes>     the neural MCTS engine, e.g. net:runs/r1/best.pt:800
    baseline:<depth>             classical alpha-beta engine from baseline.py
    stockfish:<elo>              Stockfish with UCI_LimitStrength (needs --stockfish PATH)

Each opening is played twice with colours swapped, so the result measures
strength rather than opening luck.

    python match.py net:runs/r1/best.pt:800 baseline:3 --games 40
    python match.py net:runs/r1/best.pt:800 stockfish:1500 --stockfish tools/stockfish.exe
"""
import argparse
import math
import os

import chess
import chess.engine

import baseline

OPENINGS = [  # short, common, roughly balanced
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 c7c5 b1c3 b8c6", "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5", "e2e4 d7d5 e4d5 d8d5", "e2e4 e7e5 f1c4 g8f6", "e2e4 g7g6 d2d4 f8g7",
    "d2d4 d7d5 c2c4 e7e6", "d2d4 d7d5 c2c4 c7c6", "d2d4 g8f6 c2c4 g7g6", "d2d4 g8f6 c2c4 e7e6",
    "d2d4 d7d5 g1f3 g8f6", "d2d4 f7f5 g1f3 g8f6", "c2c4 e7e5 b1c3 g8f6", "c2c4 c7c5 g1f3 b8c6",
    "g1f3 d7d5 g2g3 g8f6", "g1f3 g8f6 c2c4 b7b6", "e2e4 e7e5 g1f3 g8f6", "d2d4 g8f6 g1f3 e7e6",
]


class Player:
    def __init__(self, spec: str, stockfish_path=None, movetime=0.1):
        self.spec = spec
        kind, _, arg = spec.partition(":")
        self.kind, self._sf = kind, None
        if kind == "net":
            from search import Engine
            path, _, nodes = arg.rpartition(":")
            self.engine = Engine(path, nodes=int(nodes))
        elif kind == "baseline":
            self.depth = int(arg or 3)
        elif kind == "stockfish":
            if not stockfish_path:
                raise SystemExit("stockfish:<elo> needs --stockfish PATH")
            self._sf = chess.engine.SimpleEngine.popen_uci(os.path.abspath(stockfish_path))
            self._sf.configure({"UCI_LimitStrength": True, "UCI_Elo": int(arg)})
            self.movetime = movetime
        else:
            raise SystemExit(f"unknown player {spec!r}")

    def move(self, board: chess.Board) -> chess.Move:
        if self.kind == "net":
            return self.engine.choose(board)
        if self.kind == "baseline":
            return baseline.choose(board, self.depth)
        return self._sf.play(board, chess.engine.Limit(time=self.movetime)).move

    def close(self):
        if self._sf:
            self._sf.quit()


def play_game(white, black, opening: str, max_plies=300) -> float:
    """Return White's score (1, 0.5, 0)."""
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        board.push((white if board.turn == chess.WHITE else black).move(board))
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def elo_estimate(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    """Elo difference and 95% interval from a W/D/L record (trinomial normal approximation)."""
    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score ** 2) / n
    se = math.sqrt(var / n)

    def to_elo(s):
        s = min(max(s, 0.5 / n), 1 - 0.5 / n)
        return -400 * math.log10(1 / s - 1)
    return to_elo(score), to_elo(score - 1.96 * se), to_elo(score + 1.96 * se)


def run_match(a: Player, b: Player, games: int, log=print) -> tuple[int, int, int]:
    """Games alternate colours over OPENINGS. Returns a's (wins, draws, losses)."""
    w = d = l = 0
    for i in range(games):
        opening = OPENINGS[(i // 2) % len(OPENINGS)]
        a_white = i % 2 == 0
        score = play_game(a, b, opening) if a_white else 1 - play_game(b, a, opening)
        w, d, l = w + (score == 1), d + (score == 0.5), l + (score == 0)
        log(f"game {i + 1}/{games} ({'white' if a_white else 'black'}, {opening}): "
            f"{ {1: 'win', 0.5: 'draw', 0: 'loss'}[score] }   running +{w} ={d} -{l}")
    return w, d, l


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--stockfish", help="path to a Stockfish binary")
    p.add_argument("--sf-movetime", type=float, default=0.1)
    args = p.parse_args()
    a = Player(args.a, args.stockfish, args.sf_movetime)
    b = Player(args.b, args.stockfish, args.sf_movetime)
    try:
        w, d, l = run_match(a, b, args.games)
    finally:
        a.close()
        b.close()
    elo, lo, hi = elo_estimate(w, d, l)
    print(f"\n{args.a} vs {args.b}: +{w} ={d} -{l}  score {(w + d / 2) / (w + d + l):.1%}  "
          f"Elo diff {elo:+.0f} (95% CI {lo:+.0f} .. {hi:+.0f})")


if __name__ == "__main__":
    main()
