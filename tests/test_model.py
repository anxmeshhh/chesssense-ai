import chess
import numpy as np
import pytest
import torch

import data
import model
import train
from test_data import PGN


def test_to_planes_matches_numpy_unpack():
    boards = [chess.Board(), chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")]
    masks, metas = zip(*(data.encode_board(b) for b in boards))
    packed = np.array(masks, dtype=np.uint64)
    planes = model.to_planes(torch.from_numpy(packed.view(np.int64)), torch.tensor(metas, dtype=torch.uint8))
    assert planes.shape == (2, model.IN_PLANES, 8, 8)
    assert torch.equal(planes[:, :12], torch.from_numpy(data.unpack_boards(packed)))
    assert planes[:, 12:].sum() == 2 * 4 * 64        # all castling rights set


def test_forward_shapes_and_value_range():
    net = model.ChessNet(blocks=2, channels=32)
    policy, value = net(torch.rand(3, model.IN_PLANES, 8, 8))
    assert policy.shape == (3, data.POLICY_SIZE) and value.shape == (3,)
    assert (value.abs() <= 1).all()


def test_underpromotion_logits_line_up_with_encoding():
    # Policy index 4096 + file*9 + dir*3 + piece must come from the file's 7th-rank square.
    net = model.ChessNet(blocks=1, channels=16).eval()
    seen = {}
    net.underpromo.register_forward_hook(lambda m, i, o: seen.update(sq=i[0]))
    net(torch.rand(1, model.IN_PLANES, 8, 8))
    assert seen["sq"].shape[1] == 8
    board = chess.Board("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    idx = data.encode_move(chess.Move.from_uci("b7b8n"), board.turn)
    assert (idx - 4096) // 9 == chess.square_file(chess.B7)


def test_evaluate_returns_legal_distribution():
    net = model.ChessNet(blocks=1, channels=16).eval()
    board = chess.Board()
    probs, value = model.evaluate(net, board)
    assert set(probs) == set(board.legal_moves)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4) and -1 <= value <= 1


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    net = model.ChessNet(blocks=2, channels=32)
    x = torch.rand(16, model.IN_PLANES, 8, 8).round()
    moves = torch.randint(0, data.POLICY_SIZE, (16,))
    values = torch.randint(-1, 2, (16,)).float()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(150):
        loss, *_ = train.losses(net, x, moves, values, 1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_train_script_runs_and_resumes(tmp_path):
    src = tmp_path / "games.pgn"
    src.write_text(PGN)
    data.build(str(src), str(tmp_path / "d"), min_plies=1, val_every=2, workers=0, log_every=0)
    args = ["--data", str(tmp_path / "d"), "--out", str(tmp_path / "run"), "--blocks", "1",
            "--channels", "16", "--batch", "2", "--epochs", "2", "--eval-every", "2"]
    train.main(args)
    first = torch.load(tmp_path / "run" / "last.pt")
    assert first["step"] == 6                         # 7 train positions // 2 * 2 epochs
    ckpt = dict(first, step=4)                        # simulate a crash after step 4
    ckpt["sched"].update(last_epoch=4, _step_count=5)
    torch.save(ckpt, tmp_path / "run" / "last.pt")
    train.main(args + ["--resume"])
    assert torch.load(tmp_path / "run" / "last.pt")["step"] == 6
    with pytest.raises(ValueError, match="original --epochs"):
        train.main(args[:-3] + ["4", "--eval-every", "2", "--resume"])
    net = model.load(str(tmp_path / "run" / "best.pt"))
    assert model.evaluate(net, chess.Board())[0]
