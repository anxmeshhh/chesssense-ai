"""Template-based recognition for the supplied chessboard theme."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import chess
import numpy as np
from PIL import Image, UnidentifiedImageError

REFERENCE_BOARD = Path(__file__).parent / "vision_assets" / "reference_board.png"
NORMALIZED_SIZE = 256
SQUARE_SIZE = NORMALIZED_SIZE // 8
EMPTY_THRESHOLD = 0.035
MAX_MATCH_ERROR = 0.06
MIN_MATCH_GAP = 0.0015

REFERENCE_TEMPLATES = {
    "r": [(0, 0), (0, 7)],
    "n": [(0, 1), (0, 6)],
    "b": [(0, 2), (0, 5)],
    "q": [(0, 3)],
    "k": [(0, 4)],
    "p": [(1, file_index) for file_index in range(8)],
    "R": [(7, 0), (7, 7)],
    "N": [(7, 1), (7, 6)],
    "B": [(7, 2), (7, 5)],
    "Q": [(7, 3)],
    "K": [(7, 4)],
    "P": [(6, file_index) for file_index in range(8)],
}


class BoardRecognitionError(ValueError):
    pass


class PositionSyncError(ValueError):
    pass


def _tile_features(tile: np.ndarray) -> tuple[np.ndarray, float]:
    gray = (tile[:, :, 0] * 0.299 + tile[:, :, 1] * 0.587 + tile[:, :, 2] * 0.114).astype(np.float32)
    border = np.concatenate((gray[:4, :4].ravel(), gray[:4, -4:].ravel(),
                             gray[-4:, :4].ravel(), gray[-4:, -4:].ravel()))
    residual = (gray - np.median(border)) / 255.0
    return residual, float(np.mean(np.abs(residual)))


@lru_cache(maxsize=1)
def _templates() -> dict[str, tuple[np.ndarray, ...]]:
    if not REFERENCE_BOARD.is_file():
        raise RuntimeError(f"Piece reference image is missing: {REFERENCE_BOARD}")
    with Image.open(REFERENCE_BOARD) as source:
        image = np.asarray(source.convert("RGB").resize((NORMALIZED_SIZE, NORMALIZED_SIZE)), dtype=np.float32)
    templates = {}
    for piece, locations in REFERENCE_TEMPLATES.items():
        features = []
        for row, column in locations:
            top, left = row * SQUARE_SIZE, column * SQUARE_SIZE
            tile, _ = _tile_features(image[top:top + SQUARE_SIZE, left:left + SQUARE_SIZE])
            features.append(tile)
        templates[piece] = tuple(features)
    return templates


def _read_board_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.width < 160 or source.height < 160:
                raise BoardRecognitionError("The selected board image is too small; select the full board.")
            if abs(source.width - source.height) / max(source.width, source.height) > 0.04:
                raise BoardRecognitionError("The selected area must be a square around the chessboard.")
            image = source.convert("RGB").resize((NORMALIZED_SIZE, NORMALIZED_SIZE))
    except (UnidentifiedImageError, OSError) as error:
        raise BoardRecognitionError("The screen capture could not be read as an image.") from error
    return np.asarray(image, dtype=np.float32)


def recognize_piece_map(image_bytes: bytes, orientation: str = "white") -> tuple[dict[chess.Square, chess.Piece], float]:
    """Recognize the supplied board theme and return its piece placement."""
    if orientation not in {"white", "black"}:
        raise BoardRecognitionError("Choose whether White or Black is at the bottom of the board.")
    image = _read_board_image(image_bytes)
    templates = _templates()
    pieces: dict[chess.Square, chess.Piece] = {}
    confidences = []
    uncertain = []
    for row in range(8):
        for column in range(8):
            top, left = row * SQUARE_SIZE, column * SQUARE_SIZE
            features, energy = _tile_features(image[top:top + SQUARE_SIZE, left:left + SQUARE_SIZE])
            square_name = chess.square_name(chess.square(column, 7 - row))
            if energy < EMPTY_THRESHOLD:
                confidences.append(max(0.0, 1.0 - energy / EMPTY_THRESHOLD))
                continue
            scores = sorted(
                (float(np.mean((features - template) ** 2)), symbol)
                for symbol, variations in templates.items()
                for template in variations
            )
            best_score, symbol = scores[0]
            second_score = next(score for score, candidate in scores if candidate != symbol)
            gap = second_score - best_score
            if best_score > MAX_MATCH_ERROR or gap < MIN_MATCH_GAP:
                uncertain.append(square_name)
                continue
            confidence = max(0.0, min(1.0, 1.0 - best_score / MAX_MATCH_ERROR))
            confidences.append(confidence)
            file_index = column if orientation == "white" else 7 - column
            rank = 8 - row if orientation == "white" else row + 1
            square = chess.square(file_index, rank - 1)
            pieces[square] = chess.Piece.from_symbol(symbol)
    if uncertain:
        squares = ", ".join(uncertain[:8])
        suffix = "…" if len(uncertain) > 8 else ""
        raise BoardRecognitionError(f"Could not confidently recognize squares {squares}{suffix}. Check the crop, theme, and board animation.")
    return pieces, float(np.mean(confidences)) if confidences else 0.0


def build_board(
    pieces: dict[chess.Square, chess.Piece],
    turn: str = "w",
    castling: str = "KQkq",
    en_passant: str = "-",
) -> chess.Board:
    if turn not in {"w", "b"}:
        raise BoardRecognitionError("Choose White or Black to move for the first screen position.")
    board = chess.Board(None)
    for square, piece in pieces.items():
        board.set_piece_at(square, piece)
    board.turn = turn == "w"
    try:
        board.set_castling_fen(castling or "-")
    except ValueError as error:
        raise BoardRecognitionError("Castling rights must use KQkq letters or '-'.") from error
    if en_passant != "-":
        try:
            board.ep_square = chess.parse_square(en_passant)
        except ValueError as error:
            raise BoardRecognitionError("The en-passant target must be a square such as e3 or '-'.") from error
    if not board.is_valid():
        raise BoardRecognitionError("The recognized pieces do not form a valid chess position. Wait for the board to settle or adjust the crop.")
    return board


def advance_board(previous_fen: str, pieces: dict[chess.Square, chess.Piece]) -> tuple[chess.Board, chess.Move | None]:
    """Preserve game state by matching a newly recognized board to one legal move."""
    try:
        previous = chess.Board(previous_fen)
    except ValueError as error:
        raise PositionSyncError("The tracked position is invalid; reset screen tracking and select the side to move.") from error
    if previous.piece_map() == pieces:
        return previous, None
    matches = []
    for move in previous.legal_moves:
        candidate = previous.copy()
        candidate.push(move)
        if candidate.piece_map() == pieces:
            matches.append((candidate, move))
    if len(matches) == 1:
        return matches[0]
    raise PositionSyncError("The screen position changed by more than one legal move. Reset tracking and choose the current side to move.")


def analyze_screen_image(
    image_bytes: bytes,
    orientation: str,
    turn: str,
    castling: str,
    en_passant: str = "-",
    previous_fen: str | None = None,
) -> tuple[chess.Board, chess.Move | None, float]:
    pieces, confidence = recognize_piece_map(image_bytes, orientation)
    if previous_fen:
        board, played_move = advance_board(previous_fen, pieces)
    else:
        board, played_move = build_board(pieces, turn, castling, en_passant), None
    return board, played_move, confidence
