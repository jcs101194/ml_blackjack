"""Training entry point for the blackjack Q-table agent."""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

from blackjack import BlackjackGame
from player import QPlayer
from q_table import QTable


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a blackjack Q-table agent.")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of rounds to simulate.")
    parser.add_argument("--shoe-size", type=int, default=6, help="Number of decks in the shoe.")
    parser.add_argument("--starting-cash", type=int, default=10000, help="Starting bankroll.")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Exploration rate.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--preferred-bet", type=int, default=50, help="Default bet for the agent.")
    parser.add_argument(
        "--model-path",
        default="runtime/blackjack_q_table.json",
        help="Path where the trained Q-table should be saved.",
    )
    parser.add_argument(
        "--summary-path",
        default="runtime/training_summary.json",
        help="Path where the training summary should be saved.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save the Q-table every N episodes.",
    )
    parser.add_argument(
        "--sample-rounds",
        type=int,
        default=10,
        help="How many recent round summaries to keep in the saved training report.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the blackjack Q-table training loop."""
    args = parse_args()
    model_path = Path(args.model_path)
    summary_path = Path(args.summary_path)

    q_table = QTable.load(model_path)
    agent = QPlayer(
        q_table=q_table,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        preferred_bet=args.preferred_bet,
        training_enabled=True,
        name="Q Trainer",
    )
    game = BlackjackGame(
        shoe_size=args.shoe_size,
        starting_cash=args.starting_cash,
        auto_rebuy=True,
    )

    started_at = time.time()
    recent_rounds: list[dict[str, object]] = []

    for episode in range(1, args.episodes + 1):
        round_summary = game.play_round(agent)
        recent_rounds.append(round_summary)
        if len(recent_rounds) > args.sample_rounds:
            recent_rounds.pop(0)

        if episode % args.save_every == 0:
            q_table.save(model_path)

    q_table.save(model_path)

    summary = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": args.episodes,
        "duration_seconds": round(time.time() - started_at, 2),
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "preferred_bet": args.preferred_bet,
        "q_table_states": len(q_table),
        "stats": game.stats.to_dict(),
        "recent_rounds": recent_rounds,
        "model_path": str(model_path),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
