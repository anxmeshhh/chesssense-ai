import chess
import numpy as np
import pytest
import zstandard

import data

PGN = """\
[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "2100"]
[BlackElo "2050"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0

[Event "Rated Blitz game"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "2200"]
[TimeControl "300+0"]

1. e4 e5 0-1

[Event "Rated Bullet game"]
[Result "1-0"]
[WhiteElo "2300"]
[BlackElo "2400"]
[TimeControl "60+0"]

1. d4 d5 1-0

[Event "Rated Rapid game"]
[Result "1/2-1/2"]
[WhiteElo "2200"]
[BlackElo "2200"]
[TimeControl "600+5"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 1/2-1/2

[Event "Rated Blitz game"]
[Result "0-1"]
[WhiteElo "2500"]
[BlackElo "2500"]
[TimeControl "180+2"]
[Termination "Abandoned"]

1. e4 0-1

"""

GAME1 = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
GAME4 = ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6"]

FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1n1k3/1P6/8/8/8/8/8/4K2R w K - 0 1",          # white promotions + castling
    "4k2r/8/8/8/8/8/1p6/R1N1K3 b k - 0 1",          # black promotions + castling
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",            # en passant
]


@pytest.mark.parametrize("fen", FENS)
def test_move_encoding_roundtrip(fen):
    board = chess.Board(fen)
    indices = set()
    for move in board.legal_moves:
        idx = data.encode_move(move, board.turn)
        assert 0 <= idx < data.POLICY_SIZE
        assert data.decode_move(idx, board) == move
        indices.add(idx)
    assert len(indices) == board.legal_moves.count()


def test_underpromotions_use_extra_range():
    board = chess.Board(FENS[3])
    under = [m for m in board.legal_moves if m.promotion and m.promotion != chess.QUEEN]
    assert len(under) == 9
    assert all(data.encode_move(m, board.turn) >= 4096 for m in under)


def test_move_encoding_is_colour_symmetric():
    white = chess.Board()
    black = chess.Board(FENS[1])
    assert (data.encode_move(chess.Move.from_uci("e2e4"), white.turn)
            == data.encode_move(chess.Move.from_uci("e7e5"), black.turn))


def test_board_encoding_start_and_mirrored():
    planes = data.unpack_boards(np.array([data.encode_board(chess.Board())[0]], dtype=np.uint64))[0]
    assert planes.shape == (12, 8, 8) and planes.sum() == 32
    assert planes[0, 1].sum() == 8 and planes[5, 0, 4] == 1 and planes[11, 7, 4] == 1

    masks, meta = data.encode_board(chess.Board(FENS[1]))   # after 1.e4, Black to move
    planes = data.unpack_boards(np.array([masks], dtype=np.uint64))[0]
    assert meta == 0b1111
    assert planes[0, 1].sum() == 8                  # own (black) pawns on "rank 2"
    assert planes[5, 0, 4] == 1                     # own king on e1 after mirroring
    assert planes[6, 4, 4] == 1                     # white e4 pawn appears on e5
    assert planes[6, 1].sum() == 0 and planes[6, 6].sum() == 7


def test_castling_meta():
    _, meta = data.encode_board(chess.Board(FENS[2]))  # white: K only, black: none
    assert meta == 0b0001
    _, meta = data.encode_board(chess.Board(FENS[3]))  # black to move: own K only
    assert meta == 0b0001


def test_iter_games_and_filters():
    games = list(data.iter_games(PGN.splitlines(keepends=True)))
    assert len(games) == 5
    kept = [data.keep_game(dict(data.HEADER_RE.findall(g)), 2000, 180) for g in games]
    assert kept == [True, False, False, True, False]
    assert data.keep_game(dict(data.HEADER_RE.findall(games[2])), 2000, 0)


def replay(arrays, sans):
    """Decode stored moves from the start position and compare to the expected SANs."""
    board = chess.Board()
    for i, san in enumerate(sans):
        masks, meta = data.encode_board(board)
        assert list(arrays["boards"][i]) == masks and arrays["meta"][i] == meta
        move = data.decode_move(int(arrays["moves"][i]), board)
        assert board.san(move) == san
        board.push(move)


@pytest.mark.parametrize("workers", [0, 2])
def test_build_end_to_end(tmp_path, workers):
    src = tmp_path / "games.pgn"
    src.write_text(PGN)
    info = data.build(str(src), str(tmp_path / "out"), min_plies=1, val_every=0,
                      workers=workers, log_every=0)
    assert info["games_seen"] == 5 and info["games_kept"] == 2
    assert info["train_positions"] == len(GAME1) + len(GAME4)

    arrays = data.load_shards(str(tmp_path / "out" / "train"))
    replay(arrays, GAME1)
    replay({k: v[len(GAME1):] for k, v in arrays.items()}, GAME4)
    assert list(arrays["values"][:7]) == [1, -1, 1, -1, 1, -1, 1]
    assert not arrays["values"][7:].any()


def test_build_zst_val_split_and_min_plies(tmp_path):
    src = tmp_path / "games.pgn.zst"
    src.write_bytes(zstandard.ZstdCompressor().compress(PGN.encode()))
    info = data.build(str(src), str(tmp_path / "out"), min_plies=1, val_every=2,
                      workers=0, log_every=0)
    assert info["train_positions"] == len(GAME1) and info["val_positions"] == len(GAME4)

    info = data.build(str(src), str(tmp_path / "out2"), min_plies=10, workers=0, log_every=0)
    assert info["train_positions"] == len(GAME4)     # 7-ply game dropped


def test_refuses_to_overwrite(tmp_path):
    src = tmp_path / "games.pgn"
    src.write_text(PGN)
    data.build(str(src), str(tmp_path / "out"), min_plies=1, val_every=0, workers=0, log_every=0)
    with pytest.raises(FileExistsError):
        data.build(str(src), str(tmp_path / "out"), workers=0, log_every=0)


def test_max_positions_is_exact(tmp_path):
    src = tmp_path / "games.pgn"
    src.write_text(PGN)
    info = data.build(str(src), str(tmp_path / "out"), min_plies=1, val_every=0,
                      max_positions=9, workers=0, log_every=0)
    assert info["train_positions"] == 9
    assert len(data.load_shards(str(tmp_path / "out" / "train"))["moves"]) == 9
