from pathlib import Path

import chess
import pytest

from vision import advance_board, recognize_piece_map


REFERENCE = Path("vision_assets/reference_board.png")
FANTASY_REFERENCE = Path("vision_assets/fantasy_reference.png")
CIPHER_REFERENCE = Path("vision_assets/cipherchess_reference.png")


def test_reference_theme_recognizes_initial_position():
    pieces, confidence = recognize_piece_map(REFERENCE.read_bytes())
    board = chess.Board(None)
    for square, piece in pieces.items():
        board.set_piece_at(square, piece)
    assert board.board_fen() == chess.STARTING_BOARD_FEN
    assert len(pieces) == 32
    assert confidence > 0.8


def test_fantasy_theme_recognizes_the_supplied_live_position():
    pieces, confidence = recognize_piece_map(FANTASY_REFERENCE.read_bytes())
    board = chess.Board(None)
    for square, piece in pieces.items():
        board.set_piece_at(square, piece)
    assert board.board_fen() == "rnbqkbnr/ppp1pppp/8/3p4/8/3P4/PPP1PPPP/RNBQKBNR"
    assert board.is_valid()
    assert len(pieces) == 32
    assert confidence > 0.8


def test_analysis_board_theme_recognizes_the_supplied_position():
    pieces, confidence = recognize_piece_map(CIPHER_REFERENCE.read_bytes())
    board = chess.Board(None)
    for square, piece in pieces.items():
        board.set_piece_at(square, piece)
    assert board.board_fen() == "r3qrk1/1ppb3n/n2p2p1/3Ppp1p/1PP1P2B/2N1bP2/3NB1PP/1R1Q1RK1"
    assert board.is_valid()
    assert len(pieces) == 30
    assert confidence > 0.8


def test_screen_tracking_infers_move_and_en_passant_state():
    previous = chess.Board()
    current = previous.copy()
    current.push_uci("e2e4")
    recovered, move = advance_board(previous.fen(), current.piece_map())
    assert move == chess.Move.from_uci("e2e4")
    assert recovered.turn == chess.BLACK
    assert recovered.ep_square == chess.E3


def test_screen_tracking_rejects_multiple_untracked_moves():
    previous = chess.Board()
    current = previous.copy()
    current.push_uci("e2e4")
    current.push_uci("e7e5")
    with pytest.raises(ValueError, match="more than one legal move"):
        advance_board(previous.fen(), current.piece_map())
