"""Training entry point for the blackjack betting-policy Q-table."""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

from blackjack import BlackjackGame
from player import BET_ACTIONS, QPlayer
from q_table import QTable


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for bet-policy training."""
    parser = argparse.ArgumentParser(description="Train a blackjack betting-policy Q-table.")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of rounds to simulate.")
    parser.add_argument("--shoe-size", type=int, default=6, help="Number of decks in the shoe.")
    parser.add_argument("--starting-cash", type=int, default=10000, help="Starting bankroll.")
    parser.add_argument(
        "--action-model-path",
        default="checkpoints/blackjack_q_table.json",
        help="Path to the already-trained action Q-table.",
    )
    parser.add_argument(
        "--bet-model-path",
        default="checkpoints/blackjack_bet_q_table.json",
        help="Path where the betting-policy Q-table should be saved.",
    )
    parser.add_argument(
        "--summary-path",
        default="runtime/bet_training_summary.json",
        help="Path where the betting-policy training summary should be saved.",
    )
    parser.add_argument("--bet-epsilon", type=float, default=0.15, help="Exploration rate for betting.")
    parser.add_argument("--bet-alpha", type=float, default=0.10, help="Learning rate for betting.")
    parser.add_argument("--bet-gamma", type=float, default=0.0, help="Discount factor for betting updates.")
    parser.add_argument(
        "--action-epsilon",
        type=float,
        default=0.0,
        help="Exploration rate for the frozen action policy. Use 0 to run it greedily.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save the betting-policy Q-table every N episodes.",
    )
    parser.add_argument(
        "--sample-rounds",
        type=int,
        default=10,
        help="How many recent round summaries to keep in the saved training report.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the blackjack betting-policy training loop."""
    args = parse_args()
    action_model_path = Path(args.action_model_path)
    bet_model_path = Path(args.bet_model_path)
    summary_path = Path(args.summary_path)

    action_q_table = QTable.load(action_model_path)
    if len(action_q_table) == 0:
        raise RuntimeError(
            "The action Q-table is empty. Train the action policy first before training the betting policy."
        )

    bet_q_table = QTable.load(bet_model_path, actions=BET_ACTIONS)
    agent = QPlayer(
        action_q_table=action_q_table,
        bet_q_table=bet_q_table,
        action_epsilon=args.action_epsilon,
        bet_epsilon=args.bet_epsilon,
        action_alpha=0.0,
        bet_alpha=args.bet_alpha,
        action_gamma=0.95,
        bet_gamma=args.bet_gamma,
        train_action_policy=False,
        train_bet_policy=True,
        use_bet_policy=True,
        name="Bet Policy Trainer",
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
            bet_q_table.save(bet_model_path)

    bet_q_table.save(bet_model_path)

    summary = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": args.episodes,
        "duration_seconds": round(time.time() - started_at, 2),
        "action_model_path": str(action_model_path),
        "bet_model_path": str(bet_model_path),
        "bet_epsilon": args.bet_epsilon,
        "bet_alpha": args.bet_alpha,
        "bet_gamma": args.bet_gamma,
        "action_epsilon": args.action_epsilon,
        "bet_q_table_states": len(bet_q_table),
        "action_q_table_states": len(action_q_table),
        "stats": game.stats.to_dict(),
        "recent_rounds": recent_rounds,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
