"""Classical baseline engine: negamax alpha-beta + quiescence search.

Evaluation is material plus small piece-square bonuses (centre control, pawn
advancement, king safety in the middlegame). Captures are ordered MVV-LVA.
Used as the reference opponent when measuring the neural engine.
"""
import time

import chess

VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
MATE = 100_000


def _centre(sq):
    file, rank = chess.square_file(sq), chess.square_rank(sq)
    return 3.5 - max(abs(file - 3.5), abs(rank - 3.5))   # 0 at edge .. 3 in centre


def _pst(piece_type, sq):
    """Bonus for a white piece on sq (mirror the square for black)."""
    rank = chess.square_rank(sq)
    if piece_type == chess.PAWN:
        return 5 * rank + (10 if chess.square_file(sq) in (3, 4) and rank in (3, 4) else 0)
    if piece_type in (chess.KNIGHT, chess.BISHOP):
        return 10 * _centre(sq)
    if piece_type == chess.QUEEN:
        return 3 * _centre(sq)
    if piece_type == chess.KING:
        return 20 if sq in (chess.G1, chess.C1, chess.B1, chess.H1) else -5 * rank
    return 10 if rank == 6 else 0                           # rook on the 7th


PST = {pt: [round(_pst(pt, sq)) for sq in chess.SQUARES] for pt in VALUES}


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns from the side to move's point of view."""
    score = 0
    for sq, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            score += VALUES[piece.piece_type] + PST[piece.piece_type][sq]
        else:
            score -= VALUES[piece.piece_type] + PST[piece.piece_type][chess.square_mirror(sq)]
    return score if board.turn == chess.WHITE else -score


def _order_key(board, move):
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or chess.PAWN   # en passant
        return 10 * VALUES[victim] - VALUES[board.piece_type_at(move.from_square)] + 10_000
    return 9_000 if move.promotion else 0


class Searcher:
    def __init__(self):
        self.nodes = 0

    def quiesce(self, board, alpha, beta):
        self.nodes += 1
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        for move in sorted(moves, key=lambda m: _order_key(board, m), reverse=True):
            board.push(move)
            score = -self.quiesce(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material() or (ply and board.is_repetition(2)):
            return 0
        if depth == 0:
            return self.quiesce(board, alpha, beta)
        best = -MATE
        for move in sorted(board.legal_moves, key=lambda m: _order_key(board, m), reverse=True):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def search(self, board: chess.Board, depth=3, movetime=None) -> tuple[chess.Move, int]:
        """Iterative deepening; the best move from the previous depth is searched first."""
        board = board.copy()
        deadline = time.time() + movetime if movetime else None
        best_move, best_score = None, 0
        for d in range(1, depth + 1):
            moves = sorted(board.legal_moves, key=lambda m: _order_key(board, m), reverse=True)
            if best_move:
                moves.remove(best_move)
                moves.insert(0, best_move)
            alpha, iter_best = -MATE - 1, None
            for move in moves:
                board.push(move)
                score = -self.negamax(board, d - 1, -MATE - 1, -alpha, 1)
                board.pop()
                if score > alpha:
                    alpha, iter_best = score, move
            best_move, best_score = iter_best, alpha
            if deadline and time.time() > deadline:
                break
        return best_move, best_score


def choose(board: chess.Board, depth=3, movetime=None) -> chess.Move:
    return Searcher().search(board, depth, movetime)[0]
