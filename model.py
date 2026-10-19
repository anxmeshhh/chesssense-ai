"""Policy/value ResNet over positions encoded by data.py.

Input: 16 planes (12 piece bitboards + 4 constant castling planes), side-to-move view.
Policy: attention-style head. Each square gets a query and key vector; the
from->to logit is query[from] . key[to], giving the 4096 from*64+to moves.
The 72 under-promotion logits come from the 8 seventh-rank squares.
Value: scalar in [-1, 1] (tanh), expected result for the side to move.
"""
import math

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data

IN_PLANES = 16


def to_planes(boards: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """(N, 12) int64 bitboards + (N,) uint8 castling bits -> (N, 16, 8, 8) float."""
    device = boards.device
    bits = (boards.unsqueeze(-1) >> torch.arange(64, device=device)) & 1
    castle = (meta.long().unsqueeze(-1) >> torch.arange(4, device=device)) & 1
    castle = castle[:, :, None, None].expand(-1, -1, 8, 8)
    return torch.cat([bits.view(-1, 12, 8, 8), castle], 1).float()


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(channels, channels // 4), nn.ReLU(),
                                nn.Linear(channels // 4, channels), nn.Sigmoid())

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y * self.se(y)[:, :, None, None]
        return F.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, blocks: int = 10, channels: int = 128, head_dim: int = 64):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels, "head_dim": head_dim}
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.query = nn.Linear(channels, head_dim)
        self.key = nn.Linear(channels, head_dim)
        self.underpromo = nn.Linear(channels, 9)
        self.value = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(32 * 64, 128), nn.ReLU(),
                                   nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        x = self.body(self.stem(x))
        squares = x.flatten(2).transpose(1, 2)                  # (N, 64, C), index = square
        q, k = self.query(squares), self.key(squares)
        from_to = (q @ k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        under = self.underpromo(squares[:, 48:56])              # (N, 8 files, 9)
        policy = torch.cat([from_to.flatten(1), under.flatten(1)], 1)
        return policy, self.value(x).squeeze(-1)


def load(path: str, device="cpu") -> ChessNet:
    ckpt = torch.load(path, map_location=device)
    net = ChessNet(**ckpt["config"]).to(device)
    net.load_state_dict(ckpt["model"])
    return net.eval()


def legal_mask(board: chess.Board) -> tuple[list[chess.Move], torch.Tensor]:
    moves = list(board.legal_moves)
    return moves, torch.tensor([data.encode_move(m, board.turn) for m in moves], dtype=torch.long)


@torch.no_grad()
def evaluate(net: ChessNet, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
    """Legal-move probabilities and value (side to move) for one position."""
    device = next(net.parameters()).device
    masks, meta = data.encode_board(board)
    boards = torch.from_numpy(np.array([masks], dtype=np.uint64).view(np.int64)).to(device)
    policy, value = net(to_planes(boards, torch.tensor([meta], device=device)))
    moves, idx = legal_mask(board)
    if not moves:
        return {}, float(value)
    probs = torch.softmax(policy[0, idx.to(device)].float(), 0).tolist()
    return dict(zip(moves, probs)), float(value)
