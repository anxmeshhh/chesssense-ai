#!/usr/bin/env python
"""Visible desktop coach for permitted chess practice and analysis boards."""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import threading
import time
from dataclasses import dataclass
from io import BytesIO

import chess
import mss
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from search import Engine, value_to_cp
from vision import BoardRecognitionError, PositionSyncError, analyze_screen_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("chesssense.desktop")


@dataclass(frozen=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int


class WorkerSignals(QObject):
    finished = Signal(int, object, object)


class AnalysisWorker(QRunnable):
    def __init__(self, task_id: int, image: bytes, options: dict, engine: Engine,
                 engine_lock: threading.Lock, nodes: int):
        super().__init__()
        self.task_id, self.image, self.options = task_id, image, options
        self.engine, self.engine_lock, self.nodes = engine, engine_lock, nodes
        self.signals = WorkerSignals()

    def run(self):
        try:
            board, played_move, confidence = analyze_screen_image(self.image, **self.options)
            if board.is_game_over():
                raise BoardRecognitionError("The recognized position is already over.")
            with self.engine_lock:
                info = self.engine.analyse(board, nodes=self.nodes)
            move = info["move"]
            result = {
                "fen": board.fen(), "uci": move.uci(), "san": board.san(move),
                "from_square": chess.square_name(move.from_square),
                "to_square": chess.square_name(move.to_square),
                "cp": value_to_cp(info["value"]) * (1 if board.turn else -1),
                "pv": board.variation_san(info["pv"][:6]), "confidence": confidence,
                "played": board.san(played_move) if played_move else None,
                "turn": board.turn,
                "candidates": [
                    {"san": board.san(candidate["move"]),
                     "from_square": chess.square_name(candidate["move"].from_square),
                     "to_square": chess.square_name(candidate["move"].to_square),
                     "visits": candidate["visits"]}
                    for candidate in info.get("top", [])[:5]
                ],
                "reason": ("Gives check" if board.gives_check(move) else
                           "Wins material" if board.is_capture(move) else
                           "Castles to improve king safety" if board.is_castling(move) else
                           "Top recommendation from your trained model"),
            }
            self.signals.finished.emit(self.task_id, result, None)
        except (BoardRecognitionError, PositionSyncError, RuntimeError, ValueError) as error:
            self.signals.finished.emit(self.task_id, None, str(error))
        except Exception as error:
            LOGGER.exception("Unexpected screen-analysis failure")
            self.signals.finished.emit(self.task_id, None, f"Analysis failed: {error}")


class BoardPreview(QLabel):
    def __init__(self):
        super().__init__("Select a practice board to preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background:#091118; color:#8497a5; border:1px solid #2b3c49; border-radius:10px;")
        self.move_uci = None
        self.orientation = "white"

    def set_move(self, move_uci, orientation):
        self.move_uci, self.orientation = move_uci, orientation
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap() or not self.move_uci:
            return
        pixmap = self.pixmap()
        bounds = self.contentsRect()
        scale = min(bounds.width() / pixmap.width(), bounds.height() / pixmap.height())
        side = min(pixmap.width(), pixmap.height()) * scale
        left = bounds.x() + (bounds.width() - side) / 2
        top = bounds.y() + (bounds.height() - side) / 2
        square_size = side / 8
        move = chess.Move.from_uci(self.move_uci)

        def center(square):
            file_index, rank = chess.square_file(square), chess.square_rank(square)
            if self.orientation == "black":
                file_index, rank = 7 - file_index, 7 - rank
            return left + (file_index + .5) * square_size, top + (7 - rank + .5) * square_size

        start, end = center(move.from_square), center(move.to_square)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(25, 225, 135, 225), max(4, square_size * .10),
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(*start, *end)
        painter.setBrush(QColor(25, 225, 135, 235))
        painter.setPen(Qt.PenStyle.NoPen)
        import math
        from PySide6.QtCore import QPointF
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        length = square_size * .4
        painter.drawPolygon([QPointF(*end),
                             QPointF(end[0] - length * math.cos(angle - .55), end[1] - length * math.sin(angle - .55)),
                             QPointF(end[0] - length * math.cos(angle + .55), end[1] - length * math.sin(angle + .55))])


class BoardSelector(QWidget):
    selected = Signal(object)

    def __init__(self):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.origin = None
        self.current = None
        virtual = QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(virtual)
        self.show()
        self.raise_()
        self.activateWindow()

    def mousePressEvent(self, event):
        self.origin = event.globalPosition().toPoint()
        self.current = self.origin
        self.update()

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            self.current = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.origin is None:
            return
        self.current = event.globalPosition().toPoint()
        rect = self._square_rect()
        self.hide()
        if rect.width() >= 160:
            screen = QApplication.screenAt(rect.center()) or QApplication.primaryScreen()
            scale = screen.devicePixelRatio()
            geometry = screen.geometry()
            self.selected.emit(ScreenRegion(
                round((rect.left() - geometry.left()) * scale + screen.geometry().left() * scale),
                round((rect.top() - geometry.top()) * scale + screen.geometry().top() * scale),
                round(rect.width() * scale), round(rect.height() * scale),
            ))
        self.deleteLater()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def _square_rect(self):
        from PySide6.QtCore import QRect
        if self.origin is None or self.current is None:
            return QRect()
        dx, dy = self.current.x() - self.origin.x(), self.current.y() - self.origin.y()
        side = min(abs(dx), abs(dy))
        return QRect(self.origin.x() if dx >= 0 else self.origin.x() - side,
                     self.origin.y() if dy >= 0 else self.origin.y() - side, side, side)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(3, 10, 18, 135))
        painter.setPen(QPen(QColor("#56e39f"), 3))
        if self.origin is not None:
            painter.drawRect(self._square_rect().translated(-self.x(), -self.y()))
        else:
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Drag a square around the chessboard · Esc to cancel")


class ArrowOverlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                         Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.move_uci = None
        self.orientation = "white"

    def set_move(self, move, orientation):
        self.move_uci, self.orientation = move, orientation
        self.update()

    def paintEvent(self, event):
        if not self.move_uci:
            return
        move = chess.Move.from_uci(self.move_uci)
        size = min(self.width(), self.height()) / 8
        def center(square):
            file_index, rank = chess.square_file(square), chess.square_rank(square)
            if self.orientation == "black":
                file_index, rank = 7 - file_index, 7 - rank
            return ((file_index + .5) * size, (7 - rank + .5) * size)
        start, end = center(move.from_square), center(move.to_square)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(45, 220, 125, 205), max(5, size * .11), Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(*start, *end)
        painter.setBrush(QColor(45, 220, 125, 220))
        painter.setPen(Qt.PenStyle.NoPen)
        from PySide6.QtCore import QPointF
        import math
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        length = size * .35
        points = [QPointF(*end),
                  QPointF(end[0] - length * math.cos(angle - .55), end[1] - length * math.sin(angle - .55)),
                  QPointF(end[0] - length * math.cos(angle + .55), end[1] - length * math.sin(angle + .55))]
        painter.drawPolygon(points)


class CoachWindow(QMainWindow):
    def __init__(self, engine: Engine, nodes: int, interval_ms: int):
        super().__init__()
        self.engine, self.nodes = engine, nodes
        self.engine_lock = threading.Lock()
        self.screen_capture = mss.mss()
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(1)
        self.region = None
        self.previous_fen = None
        self.last_digest = None
        self.analyzed_digest = None
        self.pending_since = 0.0
        self.task_id = 0
        self.busy = False
        self.running = False
        self.orientation = "white"
        self.selector = None
        self.arrow = ArrowOverlay()
        self.setWindowTitle("ChessSense AI · Practice Coach")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(820, 540)
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#101820; color:#eaf1f7; font-family:'Segoe UI'; }
            QFrame#card { background:#192630; border:1px solid #2b3c49; border-radius:14px; }
            QLabel#title { font-size:20px; font-weight:700; }
            QLabel#move { font-size:28px; font-weight:700; color:#60e69f; }
            QLabel#route { font-size:18px; font-weight:700; color:#eaf1f7; }
            QLabel#muted { color:#a7b6c2; }
            QLabel#section { color:#60e69f; font-size:12px; font-weight:700; }
            QPushButton { background:#233745; border:0; border-radius:9px; padding:10px; font-weight:600; }
            QPushButton:hover { background:#304d5e; }
            QPushButton#primary { background:#1d9c67; color:white; }
            QComboBox, QSpinBox { background:#233745; border:0; padding:7px; border-radius:7px; }
        """)
        root = QWidget(); layout = QVBoxLayout(root); layout.setContentsMargins(18, 16, 18, 18); layout.setSpacing(12)
        header = QHBoxLayout(); title = QLabel("ChessSense AI"); title.setObjectName("title"); header.addWidget(title)
        header.addStretch(); hide = QPushButton("—"); hide.setFixedWidth(38); hide.clicked.connect(self.showMinimized); header.addWidget(hide)
        layout.addLayout(header)
        columns = QHBoxLayout(); columns.setSpacing(16)
        self.preview_label = BoardPreview()
        columns.addWidget(self.preview_label, 5)
        panel = QVBoxLayout(); panel.setSpacing(9)
        self.status = QLabel("Ready · select a permitted practice board")
        self.status.setObjectName("muted"); self.status.setWordWrap(True); panel.addWidget(self.status)
        self.select_button = QPushButton("Select board region"); self.select_button.clicked.connect(self.select_region); panel.addWidget(self.select_button)
        controls = QHBoxLayout()
        self.orientation_box = QComboBox(); self.orientation_box.addItems(["White at bottom", "Black at bottom"])
        self.orientation_box.currentIndexChanged.connect(self.orientation_changed)
        controls.addWidget(self.orientation_box)
        self.turn_box = QComboBox(); self.turn_box.addItems(["White to move", "Black to move"]); controls.addWidget(self.turn_box)
        panel.addLayout(controls)
        self.nodes_box = QSpinBox(); self.nodes_box.setRange(16, 10000); self.nodes_box.setValue(nodes); self.nodes_box.setPrefix("Model search nodes: "); panel.addWidget(self.nodes_box)
        self.analyze_now_button = QPushButton("My turn - analyze now")
        self.analyze_now_button.setObjectName("primary")
        self.analyze_now_button.clicked.connect(self.analyze_my_turn)
        self.analyze_now_button.setToolTip("Analyze now for the side shown at the bottom of the board")
        panel.addWidget(self.analyze_now_button)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start live coach"); self.start_button.setObjectName("primary"); self.start_button.clicked.connect(self.toggle_capture); buttons.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop"); self.stop_button.clicked.connect(self.stop_capture); buttons.addWidget(self.stop_button); panel.addLayout(buttons)
        card = QFrame(); card.setObjectName("card"); card_layout = QVBoxLayout(card); card_layout.setContentsMargins(16, 14, 16, 14)
        eyebrow = QLabel("MODEL RECOMMENDATION"); eyebrow.setObjectName("section"); card_layout.addWidget(eyebrow)
        self.move_label = QLabel("—"); self.move_label.setObjectName("move"); card_layout.addWidget(self.move_label)
        self.route_label = QLabel("From —    →    To —"); self.route_label.setObjectName("route"); card_layout.addWidget(self.route_label)
        self.eval_label = QLabel("Model evaluation · waiting for board"); self.eval_label.setWordWrap(True); card_layout.addWidget(self.eval_label)
        self.reason_label = QLabel("Select a board region, then press My turn."); self.reason_label.setWordWrap(True); self.reason_label.setObjectName("muted"); card_layout.addWidget(self.reason_label)
        self.pv_label = QLabel(""); self.pv_label.setWordWrap(True); card_layout.addWidget(self.pv_label)
        panel.addWidget(card)
        candidates_title = QLabel("TOP MODEL CANDIDATES"); candidates_title.setObjectName("section"); panel.addWidget(candidates_title)
        self.candidates_label = QLabel("No analysis yet"); self.candidates_label.setWordWrap(True); self.candidates_label.setObjectName("muted"); panel.addWidget(self.candidates_label)
        panel.addStretch(1)
        disclaimer = QLabel("For permitted practice and analysis only. Recommendations are from your local trained model.")
        disclaimer.setWordWrap(True); disclaimer.setObjectName("muted"); panel.addWidget(disclaimer)
        columns.addLayout(panel, 4); layout.addLayout(columns, 1)
        self.setCentralWidget(root)
        self.timer = QTimer(self); self.timer.timeout.connect(self.capture_tick); self.timer.start(max(250, interval_ms))

    def select_region(self):
        self.selector = BoardSelector()
        self.selector.selected.connect(self.set_region)

    def set_region(self, region):
        self.region = region
        self.previous_fen = self.last_digest = self.analyzed_digest = None
        self.status.setText(f"Board selected · {region.width} × {region.height} capture pixels")

    def orientation_changed(self, index):
        self.orientation = "white" if index == 0 else "black"
        self.preview_label.set_move(self.preview_label.move_uci, self.orientation)

    def toggle_capture(self):
        if not self.region:
            self.select_region(); self.status.setText("Select a square board region first."); return
        self.running = not self.running
        self.start_button.setText("Pause live coach" if self.running else "Resume live coach")
        self.status.setText("Live capture active" if self.running else "Capture paused")

    def stop_capture(self):
        self.running = False; self.previous_fen = self.last_digest = self.analyzed_digest = None
        self.arrow.hide(); self.start_button.setText("Start live coach")
        self.status.setText("Stopped · tracking state reset")

    def analyze_my_turn(self):
        self.turn_box.setCurrentIndex(0 if self.orientation == "white" else 1)
        self.capture_tick(force=True)

    def capture_tick(self, force=False):
        if force and not self.region:
            self.status.setText("Select the full square board region first.")
            return
        if (not self.running and not force) or not self.region or self.busy:
            return
        try:
            monitor = {"left": self.region.left, "top": self.region.top,
                       "width": self.region.width, "height": self.region.height}
            arrow_was_visible = self.arrow.isVisible()
            if arrow_was_visible:
                self.arrow.hide()
                QApplication.processEvents()
            try:
                raw = self.screen_capture.grab(monitor)
            finally:
                if arrow_was_visible and self.running:
                    self.arrow.show()
                    self.arrow.raise_()
            frame = QImage(raw.rgb, raw.width, raw.height, raw.width * 3,
                           QImage.Format.Format_RGB888).copy()
            self.preview_label.setPixmap(QPixmap.fromImage(frame))
            digest = hashlib.blake2b(raw.rgb, digest_size=16).digest()
            if digest != self.last_digest:
                self.last_digest = digest; self.pending_since = time.monotonic()
                self.arrow.hide()
                self.status.setText("Board changed · waiting for animation to settle…")
                if not force:
                    return
            if digest == self.analyzed_digest and not force:
                return
            if not force and time.monotonic() - self.pending_since < .3:
                return
            image = Image.frombytes("RGB", raw.size, raw.rgb)
            buffer = BytesIO(); image.save(buffer, format="PNG")
            self.busy = True; self.task_id += 1
            self.analyzed_digest = digest
            options = {"orientation": self.orientation,
                       "turn": ("w" if self.turn_box.currentIndex() == 0 else "b"),
                       "castling": "KQkq", "previous_fen": self.previous_fen}
            worker = AnalysisWorker(self.task_id, buffer.getvalue(), options, self.engine,
                                    self.engine_lock, self.nodes_box.value())
            worker.signals.finished.connect(self.analysis_finished)
            self.pool.start(worker)
            self.status.setText("Recognizing position and searching your trained model…")
        except Exception as error:
            LOGGER.exception("Screen capture failed")
            self.status.setText(f"Capture error: {error}")

    def analysis_finished(self, task_id, result, error):
        self.busy = False
        if task_id != self.task_id:
            return
        if error:
            self.arrow.hide()
            self.status.setText(f"{error} Reselect tightly around all 64 squares, then press My turn.")
            return
        self.previous_fen = result["fen"]
        self.move_label.setText(result["san"])
        self.route_label.setText(f"From {result['from_square']}  →  To {result['to_square']}")
        score = result["cp"] / 100
        self.eval_label.setText(f"Evaluation · {score:+.2f} pawns (White perspective) · {result['confidence']:.0%} recognition")
        self.reason_label.setText(result["reason"] + (f" · saw {result['played']}" if result["played"] else ""))
        self.pv_label.setText("Variation · " + ("  ".join(result["pv"]) or result["san"]))
        self.candidates_label.setText("\n".join(
            f"{index}. {candidate['san']}  ·  {candidate['from_square']} → {candidate['to_square']}  ·  {candidate['visits']} visits"
            for index, candidate in enumerate(result["candidates"], 1)
        ) or "No candidate lines returned")
        self.preview_label.set_move(result["uci"], self.orientation)
        self.status.setText("Recommendation ready · press My turn to refresh" if not self.running
                            else "Position analyzed · watching for the next stable move")
        if self.region:
            self.arrow.setGeometry(self.region.left, self.region.top, self.region.width, self.region.height)
            self.arrow.set_move(result["uci"], self.orientation); self.arrow.show(); self.arrow.raise_()

    def closeEvent(self, event):
        self.timer.stop(); self.arrow.close(); self.pool.waitForDone(3000)
        self.screen_capture.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/r1/best.pt")
    parser.add_argument("--nodes", type=int, default=96, help="MCTS nodes per new position; higher is stronger but slower")
    parser.add_argument("--interval", type=int, default=250, help="screen polling interval in milliseconds")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("ChessSense AI")
    try:
        LOGGER.info("Loading trained checkpoint %s", args.checkpoint)
        engine = Engine(args.checkpoint, nodes=args.nodes)
    except Exception as error:
        LOGGER.exception("Could not initialize trained model")
        print(f"Could not load trained checkpoint: {error}", file=sys.stderr)
        return 1
    window = CoachWindow(engine, args.nodes, args.interval)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
