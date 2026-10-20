#!/usr/bin/env python
"""Local web app: play the engine, study any position on an analysis board, or review a finished game.

    python app.py                          # http://localhost:5000
    python app.py --host 0.0.0.0           # friends on the same Wi-Fi use http://<your-ip>:5000
    python app.py --checkpoint runs/r1/best.pt
"""
import argparse
import io
import threading

import chess
import chess.pgn
from flask import Flask, jsonify, request, send_from_directory

from search import Engine, value_to_cp
from vision import BoardRecognitionError, PositionSyncError, analyze_screen_image

# Drop in expected score (on the [-1, 1] value scale) for the mover, as on Lichess.
LABELS = [(0.3, "blunder"), (0.2, "mistake"), (0.1, "inaccuracy")]
FINISHED = {"1-0", "0-1", "1/2-1/2"}
# Why python-chess rejects a set-up position, in the order worth reporting.
POSITION_PROBLEMS = [
    (chess.STATUS_NO_WHITE_KING, "White has no king"),
    (chess.STATUS_NO_BLACK_KING, "Black has no king"),
    (chess.STATUS_TOO_MANY_KINGS, "each side needs exactly one king"),
    (chess.STATUS_PAWNS_ON_BACKRANK, "pawns can't stand on the first or last rank"),
    (chess.STATUS_OPPOSITE_CHECK, "the side not to move is in check"),
    (chess.STATUS_BAD_CASTLING_RIGHTS, "castling rights don't match the king and rook squares"),
    (chess.STATUS_INVALID_EP_SQUARE, "the en-passant square isn't possible here"),
]


def white_cp(value: float, turn: chess.Color) -> int:
    cp = value_to_cp(value)
    return cp if turn == chess.WHITE else -cp


def create_app(engine) -> Flask:
    app = Flask(__name__, static_folder="static")
    lock = threading.Lock()                      # one search at a time on the GPU

    def nodes_arg(body, default):
        return max(1, min(int(body.get("nodes", default)), 10_000))

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/move")
    def api_move():
        body = request.get_json(force=True)
        board = chess.Board()
        try:
            for uci in body.get("moves", []):
                board.push_uci(uci)
        except ValueError as e:
            return jsonify(error=f"illegal move: {e}"), 400
        if board.is_game_over():
            return jsonify(error="game is over"), 400
        with lock:
            info = engine.analyse(board, nodes=nodes_arg(body, 800))
        return jsonify(move=info["move"].uci(), san=board.san(info["move"]),
                       cp=white_cp(info["value"], board.turn), pv=board.variation_san(info["pv"][:8]))

    @app.post("/api/analyse")
    def api_analyse():
        body = request.get_json(force=True)
        try:
            board = chess.Board(body.get("fen") or chess.STARTING_FEN)
        except (ValueError, TypeError) as e:
            return jsonify(error=f"invalid FEN: {e}"), 400
        status = board.status()
        if status != chess.STATUS_VALID:
            reason = next((text for flag, text in POSITION_PROBLEMS if status & flag), "the position is not legal")
            return jsonify(error=f"illegal position: {reason}"), 400
        try:
            for uci in body.get("moves", []):
                board.push_uci(uci)
        except (ValueError, TypeError) as e:
            return jsonify(error=f"invalid move history: {e}"), 400
        if board.is_game_over():
            return jsonify(error="game is over"), 400
        multipv = max(1, min(int(body.get("multipv", 1)), 5))
        with lock:
            info = engine.analyse(board, nodes=nodes_arg(body, 200), multipv=multipv)
        move = info["move"]
        lines = [{"move": line["move"].uci(), "san": board.san(line["move"]),
                  "from_square": chess.square_name(line["move"].from_square),
                  "to_square": chess.square_name(line["move"].to_square),
                  "cp": white_cp(line["value"], board.turn), "visits": line["visits"],
                  "pv": board.variation_san(line["pv"][:8]), "pv_uci": [m.uci() for m in line["pv"][:8]]}
                 for line in info["lines"]]
        return jsonify(move=move.uci(), san=board.san(move),
                       from_square=chess.square_name(move.from_square),
                       to_square=chess.square_name(move.to_square), fen=board.fen(),
                       cp=white_cp(info["value"], board.turn),
                       pv=board.variation_san(info["pv"][:8]), nodes=info["nodes"], lines=lines)

    @app.post("/api/screen-recognize")
    def api_screen_recognize():
        capture = request.files.get("board")
        if capture is None:
            return jsonify(error="missing selected board image"), 400
        image_bytes = capture.read(4 * 1024 * 1024 + 1)
        if len(image_bytes) > 4 * 1024 * 1024:
            return jsonify(error="board capture is too large"), 413
        previous_fen = request.form.get("previous_fen") or None
        try:
            board, played_move, confidence = analyze_screen_image(
                image_bytes,
                orientation=request.form.get("orientation", "white"),
                turn=request.form.get("turn", "w"),
                castling=request.form.get("castling", "KQkq"),
                en_passant=request.form.get("en_passant", "-"),
                previous_fen=previous_fen,
            )
        except PositionSyncError as error:
            return jsonify(error=str(error), reset_required=True), 409
        except BoardRecognitionError as error:
            return jsonify(error=str(error)), 422
        return jsonify(
            fen=board.fen(),
            confidence=round(confidence, 3),
            played_move=previous_fen and played_move.uci() if played_move else None,
        )

    @app.post("/api/review")
    def api_review():
        body = request.get_json(force=True)
        game = chess.pgn.read_game(io.StringIO(body.get("pgn", "")))
        if game is None or game.errors or not any(True for _ in game.mainline_moves()):
            return jsonify(error="could not parse that PGN"), 400
        if game.headers.get("Result") not in FINISHED:
            return jsonify(error="review only works on finished games (Result must be 1-0, 0-1 or 1/2-1/2)"), 400
        nodes = nodes_arg(body, 300)

        board = game.board()
        analyses = []                                    # (board before move, analysis) per ply
        for move in game.mainline_moves():
            with lock:
                analyses.append((board.copy(), engine.analyse(board, nodes=nodes)))
            board.push(move)
        if board.is_checkmate():
            final = -1.0
        elif board.is_game_over():
            final = 0.0
        else:
            with lock:
                final = engine.analyse(board, nodes=nodes)["value"]
        values = [a["value"] for _, a in analyses] + [final]   # each for its side to move

        rows, summary = [], {"white": collections_counter(), "black": collections_counter()}
        for i, (move, (before, info)) in enumerate(zip(game.mainline_moves(), analyses)):
            loss = max(0.0, values[i] + values[i + 1]) if move != info["move"] else 0.0
            label = next((name for limit, name in LABELS if loss >= limit), "")
            side = "white" if before.turn == chess.WHITE else "black"
            summary[side]["moves"] += 1
            summary[side]["loss"] += loss
            if label:
                summary[side][label] += 1
            rows.append({"ply": i + 1, "san": before.san(move), "best": before.san(info["move"]),
                         "fen_before": before.fen(), "cp": white_cp(values[i], before.turn),
                         "loss": round(loss, 3), "label": label})
        for s in summary.values():
            s["avg_loss"] = round(s.pop("loss") / max(1, s["moves"]), 3)
        return jsonify(rows=rows, summary=summary, final_fen=board.fen(),
                       white=game.headers.get("White", "?"), black=game.headers.get("Black", "?"),
                       result=game.headers["Result"])

    return app


def collections_counter():
    return {"moves": 0, "loss": 0.0, "blunder": 0, "mistake": 0, "inaccuracy": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs/r1/best.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    create_app(Engine(args.checkpoint)).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
