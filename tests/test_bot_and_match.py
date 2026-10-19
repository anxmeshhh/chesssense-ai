import json

import chess
import pytest

import lichess_bot
import match


# --- lichess_bot -----------------------------------------------------------
def challenge(variant="standard", tc="clock", speed="blitz"):
    return {"id": "c1", "variant": {"key": variant}, "timeControl": {"type": tc}, "speed": speed,
            "challenger": {"id": "friend"}}


def test_decline_reasons():
    assert lichess_bot.decline_reason(challenge(), 0, 1) is None
    assert lichess_bot.decline_reason(challenge(variant="chess960"), 0, 1) == "standard"
    assert lichess_bot.decline_reason(challenge(tc="correspondence"), 0, 1) == "timeControl"
    assert lichess_bot.decline_reason(challenge(speed="ultraBullet"), 0, 1) == "timeControl"
    assert lichess_bot.decline_reason(challenge(), 1, 1) == "later"


def test_think_time():
    state = {"wtime": 180_000, "btime": 60_000, "winc": 2000, "binc": 0}
    assert lichess_bot.think_time(state, chess.WHITE) == pytest.approx(180 / 40 + 1.5 - 0.3)
    assert lichess_bot.think_time(state, chess.BLACK) == pytest.approx(60 / 40 - 0.3)
    assert lichess_bot.think_time({"wtime": 500, "btime": 0, "winc": 0, "binc": 0}, chess.WHITE) == 0.05


class FakeResponse:
    def __init__(self, lines=(), payload=None, ok=True):
        self.lines, self.payload, self.ok, self.text = lines, payload, ok, ""

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, streams):
        self.streams, self.posts = streams, []

    def get(self, url, stream=False):
        if url.endswith("/account"):
            return FakeResponse(payload={"id": "mybot"})
        return FakeResponse(lines=[json.dumps(e).encode() for e in self.streams[url.rsplit("/", 1)[-1]]])

    def post(self, url, data=None):
        self.posts.append((url, data))
        return FakeResponse()


class FakeEngine:
    def analyse(self, board, nodes=None, movetime=None):
        return {"move": sorted(board.legal_moves, key=lambda m: m.uci())[0]}


class InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def test_bot_accepts_and_plays_only_on_its_turn(monkeypatch):
    monkeypatch.setattr(lichess_bot.threading, "Thread", InlineThread)
    clock = {"wtime": 60_000, "btime": 60_000, "winc": 0, "binc": 0}
    streams = {
        "event": [{"type": "challenge", "challenge": challenge()},
                  {"type": "challenge", "challenge": dict(challenge(variant="atomic"), id="c2")},
                  {"type": "gameStart", "game": {"gameId": "g1"}}],
        "g1": [{"type": "gameFull", "white": {"id": "friend"}, "black": {"id": "mybot"},
                "initialFen": "startpos", "state": dict(clock, moves="", status="started")},
               {"type": "gameState", **clock, "moves": "e2e4", "status": "started"},
               {"type": "chatLine", "text": "hi"},
               {"type": "gameState", **clock, "moves": "e2e4 a7a5 d2d4", "status": "started"},
               {"type": "gameState", **clock, "moves": "e2e4 a7a5 d2d4 a5a4", "status": "resign"}],
    }
    session = FakeSession(streams)
    bot = lichess_bot.Bot(FakeEngine(), session)
    bot.run()
    urls = [u for u, _ in session.posts]
    assert urls[0].endswith("/challenge/c1/accept")
    assert urls[1].endswith("/challenge/c2/decline") and session.posts[1][1] == {"reason": "standard"}
    moves = [u.rsplit("/", 1)[-1] for u in urls if "/move/" in u]
    assert len(moves) == 2                              # once after e2e4, once after d2d4
    assert all(chess.Move.from_uci(m) for m in moves)


# --- match -----------------------------------------------------------------
def test_elo_estimate():
    elo, lo, hi = match.elo_estimate(10, 0, 10)
    assert elo == pytest.approx(0) and lo < 0 < hi
    elo, lo, hi = match.elo_estimate(30, 5, 5)
    assert 250 < elo < 400 and lo < elo < hi
    assert match.elo_estimate(10, 0, 0)[0] > 400          # clamped, not infinite


def test_openings_are_legal():
    for line in match.OPENINGS:
        board = chess.Board()
        for uci in line.split():
            board.push_uci(uci)


def test_baseline_self_match_runs():
    a, b = match.Player("baseline:1"), match.Player("baseline:1")
    w, d, l = match.run_match(a, b, 2, log=lambda *_: None)
    assert w + d + l == 2
