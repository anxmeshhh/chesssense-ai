import chess
import pytest
import torch

import baseline
import model
import search

MATE_IN_ONE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"   # Qxf7#
POISONED_PAWN = "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1"                                  # Qxd5?? exd5


@pytest.fixture(scope="module")
def mcts():
    torch.manual_seed(0)
    return search.MCTS(model.ChessNet(blocks=1, channels=16), batch_size=8)


def visits_consistent(node):
    if not node.children:
        return True
    child_sum = sum(c.visits for c in node.children)
    return child_sum <= node.visits and all(visits_consistent(c) for c in node.children if c.visits)


def test_root_visits_add_up(mcts):
    root = mcts.search(chess.Board(), nodes=200)
    assert root.visits == 200
    assert root.visits == 1 + sum(c.visits for c in root.children)   # virtual loss fully removed
    assert visits_consistent(root)
    assert all(abs(c.q()) <= 1 for c in root.children if c.visits)


def test_mcts_finds_mate_in_one_with_untrained_net(mcts):
    root = mcts.search(chess.Board(MATE_IN_ONE), nodes=300)
    best = max(root.children, key=lambda c: c.visits)
    assert best.move == chess.Move.from_uci("h5f7")
    assert best.q() > 0.9


def test_mcts_rejects_finished_game(mcts):
    with pytest.raises(ValueError):
        mcts.search(chess.Board("7k/5QQ1/8/8/8/8/8/K7 b - - 0 1"), nodes=10)


def test_engine_wrapper(tmp_path):
    net = model.ChessNet(blocks=1, channels=16)
    ckpt = tmp_path / "net.pt"
    torch.save({"model": net.state_dict(), "config": net.config}, ckpt)
    engine = search.Engine(str(ckpt), nodes=50, device="cpu")
    board = chess.Board()
    info = engine.analyse(board)
    assert info["move"] in board.legal_moves and info["nodes"] == 50
    assert info["pv"][0] == info["move"]
    assert engine.choose(board, temperature=1.0) in board.legal_moves
    assert engine.choose(board, nodes=1) in board.legal_moves         # policy only


def test_value_to_cp():
    assert search.value_to_cp(0) == 0
    assert search.value_to_cp(0.5) > 100 and search.value_to_cp(-0.5) < -100
    assert search.value_to_cp(1.0) == -search.value_to_cp(-1.0)


def test_baseline_mate_and_quiescence():
    assert baseline.choose(chess.Board(MATE_IN_ONE), depth=2) == chess.Move.from_uci("h5f7")
    assert baseline.choose(chess.Board(POISONED_PAWN), depth=1) != chess.Move.from_uci("d1d5")
    hanging = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")          # Rxd5 wins the queen
    assert baseline.choose(hanging, depth=2) == chess.Move.from_uci("d1d5")


def test_baseline_eval_symmetric():
    board = chess.Board()
    assert baseline.evaluate(board) == 0
    board.push_san("e4")
    assert baseline.evaluate(board) < 0          # Black to move, White is better
    assert baseline.evaluate(board.mirror()) == baseline.evaluate(board)   # colours and turn swapped
