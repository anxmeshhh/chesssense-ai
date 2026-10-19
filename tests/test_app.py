import chess
import pytest
import torch

import app as app_module
import model
import search
from test_data import PGN

FINISHED_GAME = PGN.split("\n\n[Event")[0]           # scholar's mate, 1-0


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    torch.manual_seed(0)
    net = model.ChessNet(blocks=1, channels=16)
    ckpt = tmp_path_factory.mktemp("ckpt") / "net.pt"
    torch.save({"model": net.state_dict(), "config": net.config}, ckpt)
    engine = search.Engine(str(ckpt), nodes=20, device="cpu")
    return app_module.create_app(engine).test_client()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and b"chessboard" in r.data


def test_engine_move_is_legal(client):
    r = client.post("/api/move", json={"moves": ["e2e4", "e7e5"], "nodes": 20})
    assert r.status_code == 200
    board = chess.Board()
    board.push_uci("e2e4"), board.push_uci("e7e5")
    assert chess.Move.from_uci(r.json["move"]) in board.legal_moves
    assert isinstance(r.json["cp"], int) and r.json["pv"]


def test_illegal_and_finished_positions_rejected(client):
    assert client.post("/api/move", json={"moves": ["e2e5"]}).status_code == 400
    mate = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    assert client.post("/api/move", json={"moves": mate}).status_code == 400


def test_review_finished_game(client):
    r = client.post("/api/review", json={"pgn": FINISHED_GAME, "nodes": 20})
    assert r.status_code == 200, r.json
    rows = r.json["rows"]
    assert [row["san"] for row in rows] == ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
    assert r.json["summary"]["white"]["moves"] == 4 and r.json["summary"]["black"]["moves"] == 3
    assert all(row["label"] in ("", "inaccuracy", "mistake", "blunder") for row in rows)


def test_review_refuses_unfinished_game(client):
    pgn = '[Result "*"]\n\n1. e4 e5 2. Nf3 *\n'
    r = client.post("/api/review", json={"pgn": pgn})
    assert r.status_code == 400 and "finished" in r.json["error"]
    assert client.post("/api/review", json={"pgn": "not a pgn"}).status_code == 400
