# ML Blackjack

This project is a reusable blackjack engine built for three jobs:

- simulate blackjack hands locally
- train a Q-table agent
- power a live portfolio page through a small Python API server

The code now supports:

- a dedicated `Shoe` class
- split hands
- double downs
- a heuristic baseline player
- a Q-table-backed player
- a training script
- a persistent API worker for the website

## Project Layout

- `blackjack.py`
  - core engine
  - scoring helpers
  - `Shoe`
  - round/session state
  - event log used by the API
- `player.py`
  - heuristic `Player`
  - `QPlayer`
- `q_table.py`
  - Q-value storage
  - epsilon-greedy action selection
  - JSON save/load
- `train_q_agent.py`
  - offline Q-table training entry point
- `server.py`
  - persistent background worker
  - HTTP API for live state
- `main.py`
  - short local smoke-test runner

## Blackjack Rules

Implemented action rules:

- `hit`
- `stay`
- `double_down`
- `split`

Current split and double rules:

- double down is only allowed on the first action of a hand
- split is only allowed on exact opening pairs
- no re-splitting
- no double after split
- split aces receive one additional card each

## Engine Design

The old prompt-driven game loop has been replaced with a reusable engine API.

Key pieces:

- `BlackjackGame.play_round(agent)`
- `Shoe.deal_card()`
- `BlackjackGame.get_state_snapshot()`

Agents now receive structured contexts instead of raw prompt strings:

- `BetContext`
- `ActionContext`

That makes the same engine usable for:

- heuristic simulation
- Q-table training
- a live website feed

## Q-Table State

The Q-player currently encodes each decision state with:

- player score
- dealer up-card value
- usable ace flag
- pair value if the hand is splittable
- double-down eligibility
- split-hand flag

The Q-table stores action values for:

- `hit`
- `stay`
- `double_down`
- `split`

## Training

Run a short training session:

```bash
python train_q_agent.py --episodes 5000
```

Useful options:

```bash
python train_q_agent.py \
  --episodes 10000 \
  --model-path runtime/blackjack_q_table.json \
  --summary-path runtime/training_summary.json
```

Training outputs:

- saved Q-table JSON
- training summary JSON
- recent round samples in the summary output

## Live API Server

The live server runs a blackjack worker continuously in the background and exposes live state over HTTP.

Start the API:

```bash
python server.py --mode q --model-path runtime/blackjack_q_table.json
```

Useful modes:

- `heuristic`
- `q`
- `q-train`

Default endpoint:

- `/api/blackjack/state`

Health endpoint:

- `/api/blackjack/health`

Example:

```bash
python server.py --host 127.0.0.1 --port 8765 --mode q-train
```

## Front-End Integration

The portfolio page lives at:

- `actual-website/techscapades/ml/blackjack.html`

That page is designed to poll the blackjack API and render:

- bankroll
- time at table
- wins, losses, draws
- live hand state
- recent prompts and decisions
- recent resolved round details

Recommended deployment setup:

- run `server.py` as a persistent process
- proxy `/api/blackjack/*` to that process from the main site
- point the page at the same-origin API path

## Local Smoke Test

Run a short heuristic session and print the resulting snapshot:

```bash
python main.py
```

## Notes

- The shoe still uses the project's simplified rank-based deck representation.
- The Q-player supports online updates, but offline training with `train_q_agent.py` is the intended path before serving a live model.
- The API keeps a recent event log so the website can show prompts and decisions without scraping terminal output.
