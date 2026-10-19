import io
from pathlib import Path

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


def test_live_analysis_returns_a_legal_suggestion(client):
    moves = ["e2e4", "e7e5"]
    r = client.post("/api/analyse", json={"moves": moves, "nodes": 20})
    assert r.status_code == 200
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    move = chess.Move.from_uci(r.json["move"])
    assert move in board.legal_moves
    assert r.json["san"] == board.san(move)
    assert r.json["pv"] and isinstance(r.json["cp"], int)


def test_live_analysis_rejects_invalid_or_finished_positions(client):
    assert client.post("/api/analyse", json={"moves": ["e2e5"]}).status_code == 400
    mate = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    assert client.post("/api/analyse", json={"moves": mate}).status_code == 400


def test_analysis_board_page_is_served(client):
    r = client.get("/")
    assert b'id="analyze-now"' in r.data and b'id="aarrow"' in r.data and b'id="a-fen"' in r.data


def test_analysis_from_fen_returns_squares_for_the_arrow(client):
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    r = client.post("/api/analyse", json={"fen": fen, "nodes": 20})
    assert r.status_code == 200, r.json
    board = chess.Board(fen)
    move = chess.Move.from_uci(r.json["move"])
    assert move in board.legal_moves and r.json["fen"] == fen
    assert r.json["from_square"] == chess.square_name(move.from_square)
    assert r.json["to_square"] == chess.square_name(move.to_square)
    assert r.json["san"] == board.san(move)


def test_analysis_from_fen_plays_moves_on_top(client):
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    r = client.post("/api/analyse", json={"fen": fen, "moves": ["e2e4", "e8d7"], "nodes": 20})
    assert r.status_code == 200, r.json
    board = chess.Board(fen)
    board.push_uci("e2e4"), board.push_uci("e8d7")
    assert r.json["fen"] == board.fen()
    assert chess.Move.from_uci(r.json["move"]) in board.legal_moves


@pytest.mark.parametrize("fen, reason", [
    ("not a fen", "invalid FEN"),
    ("8/8/8/8/8/8/8/4K3 w - - 0 1", "Black has no king"),
    ("4k3/8/8/8/8/8/8/P3K3 w - - 0 1", "first or last rank"),
    ("4k3/8/8/8/8/8/8/4K2R w KQkq - 0 1", "castling rights"),
    ("4k3/8/8/8/8/8/4q3/4K3 b - - 0 1", "not to move is in check"),
])
def test_analysis_rejects_illegal_set_up_positions(client, fen, reason):
    r = client.post("/api/analyse", json={"fen": fen, "nodes": 20})
    assert r.status_code == 400 and reason in r.json["error"]


def test_analysis_rejects_finished_fen(client):
    stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    r = client.post("/api/analyse", json={"fen": stalemate, "nodes": 20})
    assert r.status_code == 400 and "over" in r.json["error"]


def test_screen_analysis_recognizes_and_analyzes_reference_board(client):
    image = Path("vision_assets/reference_board.png").read_bytes()
    r = client.post("/api/screen-analyse", data={
        "board": (io.BytesIO(image), "board.png"),
        "orientation": "white", "turn": "w", "castling": "KQkq", "nodes": "20",
    }, content_type="multipart/form-data")
    assert r.status_code == 200, r.json
    assert r.json["fen"] == chess.STARTING_FEN
    board = chess.Board(r.json["fen"])
    assert chess.Move.from_uci(r.json["move"]) in board.legal_moves
    assert r.json["pv"] and r.json["confidence"] > 0.8


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
