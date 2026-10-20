# ChessSense AI

A neural chess engine that learned from 400k games between 2000+ rated Lichess players.
A small ResNet predicts moves and evaluates positions, Monte-Carlo tree search turns that
into play, and a local web board lets you play it, review your games, or run it as a Lichess bot.

## How it works

| Stage | File | What it does |
|---|---|---|
| Data | `data.py` | Streams a monthly Lichess dump (no 30 GB download), keeps games where both players are 2000+ and the clock is blitz or slower, and packs each position into 100 bytes |
| Model | `model.py` | 10-block, 128-channel SE-ResNet (3.3M params). Policy head scores 4168 moves; value head predicts the result |
| Training | `train.py` | bf16 on GPU, one-cycle LR, checkpoints + resume, logs top-1/top-5 move accuracy |
| Search | `search.py` | AlphaZero-style PUCT MCTS with batched leaf evaluation, about 1000 nodes/s on an RTX 4050 |
| Baseline | `baseline.py` | Classical alpha-beta + quiescence engine, used as a sparring partner |
| Play | `app.py` | Flask + chessboard.js: play the engine with a live best-move suggestion, study any position on an analysis board, or review a finished game's PGN |
| Bot | `lichess_bot.py` | Runs the engine on a Lichess BOT account |
| Eval | `match.py` | Engine-vs-engine matches (baseline, Stockfish) with Elo estimates |

## Quick start

```bash
pip install -r requirements.txt

python data.py --month 2026-09 --out data/2026-09 --max-positions 30000000   # ~10 min
python train.py --data data/2026-09 --out runs/r1 --epochs 3                 # ~2.5 h on an RTX 4050
python app.py                                                                # http://localhost:5000
```

### Play

The **Play** tab has two modes:

- **vs Engine**: you play your trained model at the strength you choose.
- **1v1 with a friend**: two people share one board and take turns (pass and play). Enter both names; **You play** sets your colour and which side faces you.

**Recommendations for** picks whose moves get your model's top three lines, each with an eval, a "g1 → f3" label and its own arrow: **My side**, **Both sides** (1v1 only) or **Off**. Hover a line to isolate its arrow, or click it to play the move. Search work runs in Flask's request thread, so the board stays responsive.

### Analysis board

The **Analysis** tab is a study board for any position:

- **Set up a position**: press **Setup** under the board to drag pieces from the trays above and below the board, or drag a piece off the board to remove it. Choose the side to move, castling rights (impossible rights are dropped automatically) and the en-passant square. **Clear** empties the board and **Start position** resets it.
- **Load a position**: paste a FEN and press **Load FEN**, or open **Import PGN** and step through the game with the arrow buttons or by clicking a move.
- **Ask the model**: **My turn — analyze now** runs your trained model and lists its top three moves, each with its evaluation, a "g1 → f3" square label, the expected line and how many search visits it got. Each line has its own coloured arrow on the board (gold for the best move, then magenta and pink), and the eval bar beside the board shows who is better. Hover over a line to show only its arrow, or click it to play the move. With **Live updates** on, every move or edit is analyzed automatically.
- **Arrows**: **Engine** shows the model's moves, **Moves** shows the next move of a loaded game, and **Both** shows them together. The last move played is highlighted. Use the arrow keys, Home and End to step through a game.

Illegal set-ups (missing king, pawns on the back rank, the side not to move in check, and so on) are reported instead of analyzed.

On Windows (PowerShell):

```powershell
cd $HOME\Desktop\chess.com
python app.py --checkpoint runs/r1/best.pt     # then open http://localhost:5000 and pick the Analysis tab
```

The **Screen Share** tab captures a board using the browser's screen-sharing prompt and reconstructs its position for display only. It does not request engine recommendations or sync captured positions into **Analysis**. For completed games, use **Review** with the game's PGN. Enable **Recognize after board changes** for live recognition updates. Set board orientation, side to move, castling rights, and any en-passant target for the first captured position; subsequent single legal moves are tracked from successive frames. The screenshot crop is sent only to the local Flask app. Recognition uses local templates for the supplied green-and-cream Staunton, fantasy, and CipherChess themes; other piece sets are not currently supported. When starting mid-game, the full repetition history cannot be recovered from pixels alone; reset tracking if the recognized board jumps multiple moves.

For recommendations without switching back to a browser tab, use the native Windows coach:

```powershell
python -m pip install -r requirements.txt
python desktop_app.py --checkpoint runs/r1/best.pt
```

The compact, always-on-top coach captures a manually selected square board region locally with MSS, previews the exact crop, waits for the image to settle, recognizes one of the supplied board themes, then runs the existing trained ChessNet/MCTS model off the UI thread. The window shows the recommended SAN move, explicit source and destination squares, evaluation, confidence, a principal variation, and a transparent move arrow over the selected board. Click **My turn - analyze now** for a one-shot recommendation, or start live coaching to analyze stable board changes automatically; use **Pause live coach** or **Stop** at any time. For the first position, orientation sets the player's side; later single legal moves are tracked from successive positions. Castling rights are retained only when the corresponding king and rook are visibly on their starting squares. The screen recognizer currently supports the supplied themes and consecutive single legal moves. Search strength can be adjusted with the node control (default 96) or `--nodes`; more nodes take longer. For bot, practice, and analysis boards where engine assistance is permitted—not live human games.

Other commands:

```bash
python app.py --host 0.0.0.0                                  # friends on your Wi-Fi can play
python match.py net:runs/r1/best.pt:800 baseline:3 --games 40
python match.py net:runs/r1/best.pt:800 stockfish:1500 --stockfish path/to/stockfish.exe
python -m pytest -q
```

## Results

Training is in progress. Final move accuracy and Elo estimates will be added here.

## Fair play

ChessSense AI is for local practice, reviewing finished games, and Lichess BOT accounts, where bots
are allowed and labelled. Don't use its coaching during live games against people on any site.

## License

MIT, see [LICENSE](LICENSE).
