"""PUCT Monte-Carlo tree search guided by ChessNet (AlphaZero-style).

Each simulation walks down the tree choosing argmax Q + U, evaluates the leaf
with the network (policy priors + value) and backs the value up. Leaves are
evaluated in batches using virtual loss, so one GPU call serves several
simulations. Checkmate, stalemate, insufficient material, the 50-move rule
and repetitions inside the tree are scored exactly instead of by the network.

Node.value_sum is from the perspective of the player who made the move into
that node, so a parent picks the child with the highest Q directly.
"""
import math
import time

import chess
import numpy as np
import torch

import data
import model


class Node:
    __slots__ = ("move", "prior", "visits", "value_sum", "children", "terminal")

    def __init__(self, move, prior):
        self.move, self.prior = move, prior
        self.visits, self.value_sum = 0, 0.0
        self.children, self.terminal = None, None

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


def value_to_cp(q: float) -> int:
    """Map an expected score in [-1, 1] to centipawns (Lc0's conversion)."""
    q = max(-0.999, min(0.999, q))
    return round(111.714640912 * math.tan(1.5620688421 * q))


class MCTS:
    def __init__(self, net, c_puct=1.5, fpu_reduction=0.3, batch_size=16):
        self.net = net.eval()
        self.device = next(net.parameters()).device
        self.c_puct, self.fpu_reduction, self.batch_size = c_puct, fpu_reduction, batch_size

    # --- network ---------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, leaves):
        """leaves: list of (moves, masks, meta, idx) -> list of (priors, value)."""
        boards = np.array([leaf[1] for leaf in leaves], dtype=np.uint64).view(np.int64)
        meta = torch.tensor([leaf[2] for leaf in leaves], dtype=torch.uint8, device=self.device)
        x = model.to_planes(torch.from_numpy(boards).to(self.device), meta)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            policy, value = self.net(x)
        policy, value = policy.float().cpu().numpy(), value.float().cpu().numpy()
        results = []
        for i, leaf in enumerate(leaves):
            logits = policy[i, leaf[3]]
            p = np.exp(logits - logits.max())
            results.append((p / p.sum(), float(value[i])))
        return results

    @staticmethod
    def _inspect(board, is_root):
        """Return (leaf info, None) for a position to evaluate, or (None, exact value)."""
        moves = list(board.legal_moves)
        if not moves:
            return None, (-1.0 if board.is_check() else 0.0)
        if (board.is_insufficient_material() or board.halfmove_clock >= 100
                or (not is_root and board.is_repetition(2))):
            return None, 0.0
        masks, meta = data.encode_board(board)
        return (moves, masks, meta, [data.encode_move(m, board.turn) for m in moves]), None

    # --- tree ------------------------------------------------------------
    def _select(self, node):
        sqrt_n = math.sqrt(node.visits)
        fpu = -node.q() - self.fpu_reduction       # parent's own value, pessimistically
        best, best_score = None, -math.inf
        for child in node.children:
            q = child.q() if child.visits else fpu
            score = q + self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            if score > best_score:
                best, best_score = child, score
        return best

    @staticmethod
    def _backup(path, value, virtual):
        """value is for the side to move at the leaf; `virtual` removes a pending virtual loss."""
        value = -value
        for node in reversed(path):
            node.visits += 0 if virtual else 1
            node.value_sum += value + (1 if virtual else 0)
            value = -value

    def search(self, board: chess.Board, nodes=800, movetime=None) -> Node:
        board = board.copy()
        info, exact = self._inspect(board, True)
        if info is None:
            raise ValueError("game is already over")
        root = Node(None, 1.0)
        (priors, value), = self._evaluate([info])
        root.children = [Node(m, float(p)) for m, p in zip(info[0], priors)]
        root.visits, root.value_sum = 1, -value
        deadline = time.time() + movetime if movetime else None

        while root.visits < nodes and not (deadline and time.time() > deadline):
            pending, pending_ids = [], set()
            for _ in range(min(self.batch_size, nodes - root.visits)):
                node, path = root, [root]
                while node.children is not None:
                    node = self._select(node)
                    board.push(node.move)
                    path.append(node)
                if node.terminal is None and id(node) not in pending_ids:
                    leaf, node.terminal = self._inspect(board, False)
                else:
                    leaf = None
                for _ in path[1:]:
                    board.pop()
                if node.terminal is not None:
                    self._backup(path, node.terminal, virtual=False)
                elif leaf is None:
                    break                              # leaf already awaiting evaluation
                else:
                    for n in path:                     # virtual loss
                        n.visits += 1
                        n.value_sum -= 1
                    pending.append((path, leaf))
                    pending_ids.add(id(node))
            if pending:
                for (path, leaf), (priors, value) in zip(pending, self._evaluate([l for _, l in pending])):
                    path[-1].children = [Node(m, float(p)) for m, p in zip(leaf[0], priors)]
                    self._backup(path, value, virtual=True)
        return root


def principal_variation(node: Node) -> list[chess.Move]:
    """Follow the most-visited child from `node` (the first step is taken even if unvisited)."""
    pv = []
    while node.children:
        node = max(node.children, key=lambda c: (c.visits, c.prior))
        if not node.visits and pv:
            break
        pv.append(node.move)
    return pv


class Engine:
    """Convenience wrapper: load a checkpoint and pick or analyse moves."""

    def __init__(self, checkpoint: str, nodes=800, device=None, **mcts_kwargs):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mcts = MCTS(model.load(checkpoint, device), **mcts_kwargs)
        self.nodes = nodes

    def analyse(self, board: chess.Board, nodes=None, movetime=None, multipv=1) -> dict:
        root = self.mcts.search(board, nodes or self.nodes, movetime)
        ranked = sorted(root.children, key=lambda c: (c.visits, c.prior), reverse=True)
        value = -root.q()                              # side to move
        # A child's Q is for the player who made the move into it, i.e. the side to move here.
        lines = [{"move": c.move, "value": c.q(), "visits": c.visits, "pv": [c.move] + principal_variation(c)}
                 for c in ranked[:multipv] if c.visits]
        return {"move": ranked[0].move, "value": value, "cp": value_to_cp(value), "pv": principal_variation(root),
                "nodes": root.visits, "lines": lines,
                "top": [{"move": c.move, "visits": c.visits, "prior": c.prior,
                         "value": c.q() if c.visits else None} for c in ranked[:5]]}

    def choose(self, board: chess.Board, nodes=None, movetime=None, temperature=0.0) -> chess.Move:
        if temperature <= 0:
            return self.analyse(board, nodes, movetime)["move"]
        root = self.mcts.search(board, nodes or self.nodes, movetime)
        weights = np.array([c.visits for c in root.children], dtype=np.float64) ** (1 / temperature)
        return root.children[np.random.choice(len(weights), p=weights / weights.sum())].move
