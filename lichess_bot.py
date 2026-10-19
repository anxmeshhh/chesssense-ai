#!/usr/bin/env python
"""Run the engine as a Lichess BOT account (bots are allowed on Lichess and labelled as such).

One-time setup (do this on a NEW Lichess account; upgrading is permanent and
only works on accounts that have never played a game):
  1. Create the account, then a token at https://lichess.org/account/oauth/token
     with the "Play games with the bot API" (bot:play) scope.
  2. Upgrade it:  curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer <token>"
  3. Run:         set LICHESS_TOKEN=<token>   (PowerShell: $env:LICHESS_TOKEN="<token>")
                  python lichess_bot.py --checkpoint runs/r1/best.pt

The bot accepts standard-variant challenges with a real clock (bullet and up),
plays up to --max-games at once, and declines everything else.
"""
import argparse
import json
import os
import threading

import chess
import requests

API = "https://lichess.org/api"
SPEEDS = {"bullet", "blitz", "rapid", "classical"}


def decline_reason(challenge: dict, active_games: int, max_games: int) -> str | None:
    """None to accept, otherwise a Lichess decline reason key."""
    if challenge.get("variant", {}).get("key") != "standard":
        return "standard"
    if challenge.get("timeControl", {}).get("type") != "clock" or challenge.get("speed") not in SPEEDS:
        return "timeControl"
    if active_games >= max_games:
        return "later"
    return None


def think_time(state: dict, color: chess.Color, latency=0.3, cap=15.0) -> float:
    """Seconds to spend on this move from the remaining clock and increment."""
    side = "w" if color == chess.WHITE else "b"
    remaining = state[f"{side}time"] / 1000
    increment = state[f"{side}inc"] / 1000
    budget = min(remaining / 40 + 0.75 * increment, remaining / 10, cap)
    return max(0.05, budget - latency)


def board_from(initial_fen: str, moves: str) -> chess.Board:
    board = chess.Board() if initial_fen in ("startpos", None) else chess.Board(initial_fen)
    for uci in moves.split():
        board.push_uci(uci)
    return board


class Bot:
    def __init__(self, engine, session, max_games=1, max_nodes=20_000):
        self.engine, self.session = engine, session
        self.max_games, self.max_nodes = max_games, max_nodes
        self.lock = threading.Lock()                # one search at a time
        self.active = set()
        self.user_id = None

    def _events(self, url):
        response = self.session.get(url, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:                                   # blank lines are keep-alives
                yield json.loads(line)

    def run(self):
        self.user_id = self.session.get(f"{API}/account").json()["id"]
        print(f"logged in as {self.user_id}, waiting for challenges", flush=True)
        for event in self._events(f"{API}/stream/event"):
            if event["type"] == "challenge":
                ch = event["challenge"]
                if ch.get("challenger", {}).get("id") == self.user_id:
                    continue                           # our own outgoing challenge
                reason = decline_reason(ch, len(self.active), self.max_games)
                if reason:
                    self.session.post(f"{API}/challenge/{ch['id']}/decline", data={"reason": reason})
                else:
                    self.session.post(f"{API}/challenge/{ch['id']}/accept")
            elif event["type"] == "gameStart":
                game_id = event["game"]["gameId"]
                if game_id not in self.active:
                    self.active.add(game_id)
                    threading.Thread(target=self.play_game, args=(game_id,), daemon=True).start()

    def play_game(self, game_id: str):
        try:
            color, initial_fen = None, "startpos"
            for event in self._events(f"{API}/bot/game/stream/{game_id}"):
                if event["type"] == "gameFull":
                    color = chess.WHITE if event["white"].get("id") == self.user_id else chess.BLACK
                    initial_fen = event.get("initialFen", "startpos")
                    state = event["state"]
                elif event["type"] == "gameState":
                    state = event
                else:
                    continue
                if state["status"] != "started":
                    break
                board = board_from(initial_fen, state["moves"])
                if board.turn != color or board.is_game_over():
                    continue
                with self.lock:
                    move = self.engine.analyse(board, nodes=self.max_nodes,
                                               movetime=think_time(state, color))["move"]
                r = self.session.post(f"{API}/bot/game/{game_id}/move/{move.uci()}")
                if not r.ok:
                    print(f"[{game_id}] move {move.uci()} rejected: {r.text}", flush=True)
        finally:
            self.active.discard(game_id)
            print(f"[{game_id}] finished", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs/r1/best.pt")
    p.add_argument("--max-games", type=int, default=1)
    p.add_argument("--max-nodes", type=int, default=20_000)
    args = p.parse_args()
    token = os.environ.get("LICHESS_TOKEN")
    if not token:
        raise SystemExit("set LICHESS_TOKEN first (see the instructions at the top of this file)")
    from search import Engine
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    Bot(Engine(args.checkpoint), session, args.max_games, args.max_nodes).run()


if __name__ == "__main__":
    main()
