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
| Play | `app.py` | Flask + chessboard.js: play the engine with a live best-move suggestion, or review a finished game's PGN |
| Bot | `lichess_bot.py` | Runs the engine on a Lichess BOT account |
| Eval | `match.py` | Engine-vs-engine matches (baseline, Stockfish) with Elo estimates |

## Quick start

```bash
pip install -r requirements.txt

python data.py --month 2026-09 --out data/2026-09 --max-positions 30000000   # ~10 min
python train.py --data data/2026-09 --out runs/r1 --epochs 3                 # ~2.5 h on an RTX 4050
python app.py                                                                # http://localhost:5000
```

During a local practice game, the live coach analyses the current position when it is your turn and shows the engine's top move, evaluation, and principal variation. Search work runs in Flask's request thread, so board interaction remains responsive.

The Play page also has a **Screen board** panel. Choose a screen or window in the browser's screen-sharing prompt, then drag a square over the board. The app recognizes the supplied green-and-cream outlined Staunton theme and the fantasy green theme in the supplied screen example using local image templates, builds a chess position, analyzes it with the existing engine, and draws a move arrow over the screen preview. Enable **Analyze after board changes** for automatic updates. Set the board orientation, side to move, castling rights, and any en-passant target for the first captured position; later single legal moves are tracked from successive positions. The screenshot crop is sent only to the local Flask app. Other piece sets are not supported by these templates. When starting mid-game, repetition history cannot be recovered from pixels alone; reset tracking if the recognized board jumps multiple moves.

For recommendations without switching back to a browser tab, use the native Windows coach:

```powershell
python -m pip install -r requirements.txt
python desktop_app.py --checkpoint runs/r1/best.pt
```

The compact, always-on-top coach captures a manually selected square board region locally with MSS, waits for the image to settle, recognizes either supplied board theme, then runs the existing trained ChessNet/MCTS model off the UI thread. The window shows the recommended SAN/UCI move, evaluation, confidence, a principal variation, and a transparent arrow over the selected board. It polls only while live coaching is enabled and skips unchanged frames; use **Pause live coach** or **Stop** at any time. The initial side to move and board orientation must be correct for the first position; castling rights default to all available and are tracked after a recognized legal move. The screen recognizer currently supports the two supplied themes and consecutive single legal moves. Search strength can be adjusted with the node control (default 96) or `--nodes`; more nodes take longer. For manual practice/analysis boards where engine assistance is permitted only—not live games against people.

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
