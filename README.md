# ML Blackjack

This directory contains a small blackjack environment and a baseline simulation that can be used as the starting point for a Q-table blackjack agent.

Right now the project has two main pieces:

- `blackjack.py`: the game environment, card dealing logic, scoring, bankroll tracking, and game loop
- `player.py`: a simple automated player that bets a fixed amount and decides whether to hit based on the remaining cards in the shoe

## Current Status

Implemented:

- Multi-deck shoe setup
- Random card dealing from the shoe
- Hand score calculation
- Interactive blackjack loop
- Agent hook for automated decisions
- Basic experiment runner in `player.py`
- End-of-run stats such as wins, busts, bankroll, and win rate

Planned next:

- Q-learning agent
- Q-table state/action design
- Split-hand support in the game loop
- Cleaner training and evaluation scripts

## Project Files

### `blackjack.py`

Core environment logic:

- Builds and refreshes the shoe
- Deals cards
- Calculates hand scores
- Runs the full blackjack loop
- Accepts either human input or an agent object with a `decide(...)` method

The environment currently supports:

- Betting choices of `50`, `100`, `250`, `500`, or `1000`
- Player actions `hit` and `stay`
- A visible prompt for `split`, although split behavior has not been implemented yet

### `player.py`

Contains a small baseline agent:

- `betting_algorithm()` always places the minimum bet
- `hitting_algorithm(agent)` estimates the probability of drawing a safe card based on the remaining shoe
- `Player` stores the current shoe, hand, and score so decision logic can inspect the environment

This is not a learned policy yet. It is a hand-written heuristic that can be used as a comparison point once the Q-table agent is added.

## How It Works

### Shoe

The shoe is represented as a list of decks. Each deck contains:

`['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']`

By default, the game uses a 6-deck shoe.

### Agent Interface

An agent can be passed into `blackjack(...)` as long as it provides:

- `set_shoe(shoe)`
- `set_hand(hand)`
- `set_score(score)`
- `decide(prompt)`

The environment calls `decide(prompt)` for:

- bet selection
- hit/stay decisions during a hand

### Baseline Hitting Strategy

The current automated player:

1. Computes the highest card it can safely draw without busting
2. Counts how many cards remaining in the shoe are "good"
3. Estimates the probability of a safe draw
4. Hits if that probability is above a confidence threshold

This makes `player.py` a useful smoke test for the environment before adding reinforcement learning.

## Running the Project

### Requirements

- Python 3
- `readchar`

Install the dependency with:

```bash
pip install readchar
```

### Run the interactive game

```bash
python3 blackjack.py
```

### Run the automated baseline simulation

```bash
python3 player.py
```

## Current Limitations

- Split is shown in the prompt but not implemented in game state or payout handling
- The baseline player is heuristic, not learned
- Dealer information is currently fully visible during play, which is convenient for testing but differs from standard blackjack
- Ace scoring logic is still simple and may need refinement for edge cases with multiple aces
- There is no dedicated training loop, persistence format, or Q-table module yet

## Suggested Next Steps

When extending this into a Q-learning project, a reasonable order is:

1. Define the state representation
2. Define the action space
3. Add split support to the environment
4. Create a Q-table structure
5. Implement training episodes and update rules
6. Compare the learned policy against the current heuristic player

Possible state features:

- player total
- usable ace or not
- dealer up-card
- pair/splittable hand or not
- shoe information, if card counting is meant to be part of the state

Possible actions:

- hit
- stay
- split
- optionally bet sizing as a separate decision process

## Goal of This Directory

The short-term goal is to turn this simplified blackjack simulator into a reinforcement learning playground:

- a reusable blackjack environment
- a Q-table agent
- a training script
- evaluation against the current rule-based player

This README documents the current foundation before those pieces are added.
