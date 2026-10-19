#!/usr/bin/env python
"""Stream a Lichess PGN dump into compact training shards.

Positions are stored from the side-to-move's perspective: when Black is to
move the board is mirrored (ranks flipped, colours swapped), so the 12 planes
are always [own P N B R Q K, opponent P N B R Q K] and the network never needs
a side-to-move plane. Moves are encoded in the same mirrored coordinates.

Shard format (.npz, ~100 bytes/position):
    boards  (N, 12) uint64  one bitboard per plane, bit i = square i (a1 = 0)
    meta    (N,)    uint8   castling rights, bits: own K, own Q, opp K, opp Q
    moves   (N,)    uint16  policy index of the move played (see encode_move)
    values  (N,)    int8    game result for the side to move: +1 / 0 / -1

Usage:
    python data.py --month 2026-09 --out data/2026-09 --max-positions 5000000
    python data.py --input games.pgn.zst --out data/local
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import signal
import sys
import time
import urllib.request
from multiprocessing import Pool

import chess
import chess.pgn
import numpy as np
import zstandard

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
FIELDS = ("boards", "meta", "moves", "values")

# --- Move encoding --------------------------------------------------------
# 0..4095     from*64 + to, in side-to-move coordinates (queen promotions too)
# 4096..4167  under-promotions: (from_file, direction -1/0/+1, piece N/B/R)
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
POLICY_SIZE = 64 * 64 + 8 * 3 * 3


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    from_sq, to_sq = move.from_square, move.to_square
    if turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    if move.promotion and move.promotion != chess.QUEEN:
        file = chess.square_file(from_sq)
        direction = chess.square_file(to_sq) - file + 1
        return 4096 + (file * 3 + direction) * 3 + UNDERPROMOTIONS.index(move.promotion)
    return from_sq * 64 + to_sq


def decode_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of encode_move. Does not check legality."""
    promotion = None
    if index >= 4096:
        file, rest = divmod(index - 4096, 9)
        direction, piece = divmod(rest, 3)
        from_sq = chess.square(file, 6)
        to_sq = chess.square(file + direction - 1, 7)
        promotion = UNDERPROMOTIONS[piece]
    else:
        from_sq, to_sq = divmod(index, 64)
    if board.turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    if (promotion is None and board.piece_type_at(from_sq) == chess.PAWN
            and chess.square_rank(to_sq) in (0, 7)):
        promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion)


# --- Board encoding -------------------------------------------------------
def encode_board(board: chess.Board) -> tuple[list[int], int]:
    """Return (12 bitboards, castling meta) from the side-to-move's view."""
    us, them = board.turn, not board.turn
    masks = ([board.pieces_mask(pt, us) for pt in chess.PIECE_TYPES]
             + [board.pieces_mask(pt, them) for pt in chess.PIECE_TYPES])
    if us == chess.BLACK:
        masks = [chess.flip_vertical(m) for m in masks]
    meta = (board.has_kingside_castling_rights(us)
            | board.has_queenside_castling_rights(us) << 1
            | board.has_kingside_castling_rights(them) << 2
            | board.has_queenside_castling_rights(them) << 3)
    return masks, meta


def unpack_boards(boards: np.ndarray) -> np.ndarray:
    """(N, 12) uint64 bitboards -> (N, 12, 8, 8) float32 planes [plane, rank, file]."""
    raw = np.ascontiguousarray(boards, dtype="<u8").view(np.uint8)
    return np.unpackbits(raw, axis=-1, bitorder="little").reshape(-1, 12, 8, 8).astype(np.float32)


# --- PGN streaming and filtering -----------------------------------------
HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$', re.M)
RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
BAD_TERMINATIONS = {"Abandoned", "Rules infraction"}


def open_source(source: str) -> io.TextIOBase:
    """Open a local path or URL (.pgn or .pgn.zst) as a text stream, without buffering it to disk."""
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "chess-engine-data/0.1"})
        raw = urllib.request.urlopen(request)
    else:
        raw = open(source, "rb")
    if source.endswith(".zst"):
        # Lichess dumps use a long zstd window; the default limit rejects them.
        raw = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(raw, read_size=1 << 20)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def iter_games(lines):
    """Split a PGN text stream into one string per game, without parsing moves."""
    buf, in_moves = [], False
    for line in lines:
        if line.startswith("["):
            if in_moves:
                yield "".join(buf)
                buf, in_moves = [], False
        elif line.strip():
            in_moves = True
        buf.append(line)
    if in_moves:
        yield "".join(buf)


def estimated_seconds(time_control: str) -> int | None:
    """Lichess-style estimated duration (base + 40 * increment); None if unknown/correspondence."""
    base, _, inc = time_control.partition("+")
    try:
        return int(base) + 40 * int(inc or 0)
    except ValueError:
        return None


def keep_game(headers: dict, min_elo: int, min_seconds: int) -> bool:
    try:
        if min(int(headers["WhiteElo"]), int(headers["BlackElo"])) < min_elo:
            return False
    except (KeyError, ValueError):
        return False
    if headers.get("Result") not in RESULTS:
        return False
    if headers.get("Variant", "Standard") != "Standard" or "FEN" in headers:
        return False
    if headers.get("Termination") in BAD_TERMINATIONS:
        return False
    if min_seconds:
        seconds = estimated_seconds(headers.get("TimeControl", "-"))
        if seconds is not None and seconds < min_seconds:
            return False
    return True


def convert_games(texts: list[str], min_plies: int) -> tuple[np.ndarray, ...]:
    """Parse games and return (boards, meta, moves, values) arrays for every position."""
    boards, meta, moves, values = [], [], [], []
    for text in texts:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None or game.errors:
            continue
        result = RESULTS[game.headers["Result"]]
        board = game.board()
        g_boards, g_meta, g_moves, g_values = [], [], [], []
        for move in game.mainline_moves():
            masks, m = encode_board(board)
            g_boards.append(masks)
            g_meta.append(m)
            g_moves.append(encode_move(move, board.turn))
            g_values.append(result if board.turn == chess.WHITE else -result)
            board.push(move)
        if len(g_moves) < min_plies:
            continue
        boards += g_boards
        meta += g_meta
        moves += g_moves
        values += g_values
    return (np.array(boards, dtype=np.uint64).reshape(-1, 12),
            np.array(meta, dtype=np.uint8),
            np.array(moves, dtype=np.uint16),
            np.array(values, dtype=np.int8))


# --- Shard output ---------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir: str, shard_size: int, limit: int | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.shard_size, self.limit = out_dir, shard_size, limit
        self.parts, self.pending, self.total, self.index = [], 0, 0, 0

    def add(self, arrays):
        if self.limit is not None:
            arrays = tuple(a[:self.limit - self.total] for a in arrays)
        n = len(arrays[2])
        if n == 0:
            return
        self.parts.append(arrays)
        self.pending += n
        self.total += n
        if self.pending >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.parts:
            return
        columns = dict(zip(FIELDS, (np.concatenate(c) for c in zip(*self.parts))))
        tmp = os.path.join(self.out_dir, f".tmp_{self.index:05d}.npz")
        np.savez(tmp, **columns)
        os.replace(tmp, os.path.join(self.out_dir, f"shard_{self.index:05d}.npz"))
        self.parts, self.pending, self.index = [], 0, self.index + 1


def load_shards(directory: str) -> dict[str, np.ndarray]:
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no shards in {directory}")
    parts = [np.load(p) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


class _Done:
    """Stand-in for AsyncResult when running without a worker pool."""
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build(source: str, out_dir: str, *, min_elo=2000, min_seconds=180, min_plies=10,
          max_positions=None, max_games=None, val_every=50, shard_size=1_000_000,
          workers=None, batch_games=256, log_every=10.0) -> dict:
    """Stream `source`, filter games, and write train/ and val/ shards under `out_dir`.

    Every `val_every`-th kept game goes to val/ (0 disables), so the split is by game.
    max_positions caps train positions exactly (in-flight batches are truncated).
    """
    if glob.glob(os.path.join(out_dir, "*", "shard_*.npz")):
        raise FileExistsError(f"{out_dir} already contains shards; use a new --out")
    workers = max(1, (os.cpu_count() or 2) - 2) if workers is None else workers
    writers = {False: ShardWriter(os.path.join(out_dir, "train"), shard_size, max_positions),
               True: ShardWriter(os.path.join(out_dir, "val"), shard_size)}
    batches = {False: [], True: []}
    pending = collections.deque()
    stats = collections.Counter()
    pool = Pool(workers, initializer=_ignore_sigint) if workers > 0 else None
    start = last_log = time.time()

    def submit(is_val):
        texts, batches[is_val] = batches[is_val], []
        if texts:
            job = (pool.apply_async(convert_games, (texts, min_plies)) if pool
                   else _Done(convert_games(texts, min_plies)))
            pending.append((is_val, job))

    def drain(limit):
        while len(pending) > limit:
            is_val, job = pending.popleft()
            writers[is_val].add(job.get())

    try:
        with open_source(source) as stream:
            for text in iter_games(stream):
                stats["seen"] += 1
                if not keep_game(dict(HEADER_RE.findall(text)), min_elo, min_seconds):
                    continue
                stats["kept"] += 1
                is_val = val_every > 0 and stats["kept"] % val_every == 0
                batches[is_val].append(text)
                if len(batches[is_val]) >= batch_games:
                    submit(is_val)
                    drain(max(1, workers) * 4)
                if max_positions and writers[False].total >= max_positions:
                    break
                if max_games and stats["kept"] >= max_games:
                    break
                if log_every and time.time() - last_log > log_every:
                    last_log = time.time()
                    print(f"games {stats['seen']:,}  kept {stats['kept']:,} "
                          f"({stats['kept'] / stats['seen']:.1%})  "
                          f"train {writers[False].total:,}  val {writers[True].total:,}  "
                          f"[{stats['seen'] / (last_log - start):,.0f} games/s]",
                          file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("interrupted, writing what we have...", file=sys.stderr)
    finally:
        submit(False)
        submit(True)
        drain(0)
        if pool:
            pool.close()
            pool.join()
        for w in writers.values():
            w.flush()

    info = {"source": source, "min_elo": min_elo, "min_seconds": min_seconds,
            "min_plies": min_plies, "val_every": val_every,
            "games_seen": stats["seen"], "games_kept": stats["kept"],
            "train_positions": writers[False].total, "val_positions": writers[True].total,
            "seconds": round(time.time() - start, 1), "policy_size": POLICY_SIZE}
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--month", help="Lichess month to stream, e.g. 2026-09")
    src.add_argument("--input", help="local path or URL to a .pgn or .pgn.zst")
    p.add_argument("--out", required=True, help="output directory (gets train/ and val/)")
    p.add_argument("--min-elo", type=int, default=2000, help="both players at least this (default 2000)")
    p.add_argument("--min-seconds", type=int, default=180,
                   help="min base+40*inc seconds; 180 drops bullet, 0 keeps all (default 180)")
    p.add_argument("--min-plies", type=int, default=10)
    p.add_argument("--max-positions", type=int, help="stop after ~this many train positions")
    p.add_argument("--max-games", type=int, help="stop after this many kept games")
    p.add_argument("--val-every", type=int, default=50, help="every Nth kept game -> val (default 50 = 2%%)")
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, help="parser processes (default cpu_count-2; 0 = in-process)")
    args = p.parse_args()

    source = LICHESS_URL.format(args.month) if args.month else args.input
    info = build(source, args.out, min_elo=args.min_elo, min_seconds=args.min_seconds,
                 min_plies=args.min_plies, max_positions=args.max_positions,
                 max_games=args.max_games, val_every=args.val_every,
                 shard_size=args.shard_size, workers=args.workers)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
