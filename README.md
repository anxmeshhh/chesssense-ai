# Knightshade

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
| Play | `app.py` | Flask + chessboard.js: play the engine, or paste a finished game's PGN for a move-by-move review |
| Bot | `lichess_bot.py` | Runs the engine on a Lichess BOT account |
| Eval | `match.py` | Engine-vs-engine matches (baseline, Stockfish) with Elo estimates |

## Quick start

```bash
pip install -r requirements.txt

python data.py --month 2026-09 --out data/2026-09 --max-positions 30000000   # ~10 min
python train.py --data data/2026-09 --out runs/r1 --epochs 3                 # ~2.5 h on an RTX 4050
python app.py                                                                # http://localhost:5000
```

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

Knightshade is for local play, reviewing finished games, and Lichess BOT accounts, where bots
are allowed and labelled. Don't use it for help during live games against people on any site.
