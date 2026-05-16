"""Evaluation script for comparing fixed bets against a learned bet policy."""

from __future__ import annotations

import argparse
import json
import random
import time

from pathlib import Path
from typing import Any

from blackjack import BET_OPTIONS, BlackjackGame
from player import QPlayer
from q_table import QTable


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for bet-policy evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the learned blackjack bet policy.")
    parser.add_argument("--episodes", type=int, default=10000, help="Rounds per trial.")
    parser.add_argument("--trials", type=int, default=5, help="How many seeded trials to run.")
    parser.add_argument("--shoe-size", type=int, default=6, help="Number of decks in the shoe.")
    parser.add_argument("--starting-cash", type=int, default=10000, help="Starting bankroll.")
    parser.add_argument(
        "--action-model-path",
        default="checkpoints/blackjack_q_table.json",
        help="Path to the trained action Q-table.",
    )
    parser.add_argument(
        "--bet-model-path",
        default="checkpoints/blackjack_bet_q_table.json",
        help="Path to the trained betting Q-table.",
    )
    parser.add_argument(
        "--fixed-bets",
        nargs="*",
        type=int,
        default=list(BET_OPTIONS),
        help="Fixed bet baselines to compare against the learned bet policy.",
    )
    parser.add_argument(
        "--summary-path",
        default="runtime/bet_policy_evaluation.json",
        help="Path where the evaluation summary should be written.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed for paired trials.")
    return parser.parse_args()


def run_trial(
    *,
    policy_name: str,
    episodes: int,
    seed: int,
    shoe_size: int,
    starting_cash: int,
    action_q_table: QTable,
    bet_q_table: QTable,
    preferred_bet: int,
    use_bet_policy: bool,
) -> dict[str, Any]:
    """Runs one seeded evaluation trial for a single betting policy."""
    game = BlackjackGame(
        shoe_size=shoe_size,
        starting_cash=starting_cash,
        rng=random.Random(seed),
        auto_rebuy=True,
    )
    agent = QPlayer(
        action_q_table=action_q_table,
        bet_q_table=bet_q_table,
        action_epsilon=0.0,
        bet_epsilon=0.0,
        train_action_policy=False,
        train_bet_policy=False,
        preferred_bet=preferred_bet,
        use_bet_policy=use_bet_policy,
        name=policy_name,
        rng=random.Random(seed),
    )

    total_round_reward = 0.0
    total_bet = 0
    bet_counts = {str(bet): 0 for bet in BET_OPTIONS}

    for _ in range(episodes):
        round_summary = game.play_round(agent)
        total_round_reward += float(round_summary["round_net_reward"])
        total_bet += int(round_summary["bet"])
        bet_counts[str(round_summary["bet"])] += 1

    stats = game.stats.to_dict()
    return {
        "seed": seed,
        "policy_name": policy_name,
        "episodes": episodes,
        "final_cash": stats["cash"],
        "most_cash": stats["most_cash"],
        "win_rate": stats["win_rate"],
        "bankroll_resets": stats["bankroll_resets"],
        "average_round_reward": round(total_round_reward / episodes, 4),
        "total_round_reward": round(total_round_reward, 2),
        "average_bet": round(total_bet / episodes, 2),
        "bet_counts": bet_counts,
        "stats": stats,
    }


def aggregate_trials(policy_name: str, trial_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates trial metrics into one comparison block."""
    trial_count = len(trial_summaries)
    bet_distribution = {str(bet): 0 for bet in BET_OPTIONS}

    for summary in trial_summaries:
        for bet, count in summary["bet_counts"].items():
            bet_distribution[bet] += count

    total_bets = sum(bet_distribution.values()) or 1

    return {
        "policy_name": policy_name,
        "trials": trial_count,
        "episodes_per_trial": trial_summaries[0]["episodes"] if trial_summaries else 0,
        "average_final_cash": round(sum(item["final_cash"] for item in trial_summaries) / trial_count, 2),
        "average_most_cash": round(sum(item["most_cash"] for item in trial_summaries) / trial_count, 2),
        "average_win_rate": round(sum(item["win_rate"] for item in trial_summaries) / trial_count, 4),
        "average_round_reward": round(
            sum(item["average_round_reward"] for item in trial_summaries) / trial_count,
            4,
        ),
        "average_bet": round(sum(item["average_bet"] for item in trial_summaries) / trial_count, 2),
        "average_bankroll_resets": round(
            sum(item["bankroll_resets"] for item in trial_summaries) / trial_count,
            2,
        ),
        "bet_distribution": {
            bet: round(count / total_bets, 4)
            for bet, count in bet_distribution.items()
        },
        "trial_summaries": trial_summaries,
    }


def main() -> None:
    """Runs the full betting-policy evaluation suite."""
    args = parse_args()
    action_model_path = Path(args.action_model_path)
    bet_model_path = Path(args.bet_model_path)
    summary_path = Path(args.summary_path)

    action_q_table = QTable.load(action_model_path)
    if len(action_q_table) == 0:
        raise RuntimeError("The action Q-table is empty. Train the action policy before evaluation.")

    bet_q_table = QTable.load(bet_model_path)
    if len(bet_q_table) == 0:
        raise RuntimeError("The betting Q-table is empty. Train the betting policy before evaluation.")

    started_at = time.time()
    comparisons: list[dict[str, Any]] = []

    for fixed_bet in args.fixed_bets:
        trial_summaries = []
        for trial_index in range(args.trials):
            trial_seed = args.seed + trial_index
            trial_summaries.append(
                run_trial(
                    policy_name=f"Fixed Bet {fixed_bet}",
                    episodes=args.episodes,
                    seed=trial_seed,
                    shoe_size=args.shoe_size,
                    starting_cash=args.starting_cash,
                    action_q_table=action_q_table,
                    bet_q_table=bet_q_table,
                    preferred_bet=fixed_bet,
                    use_bet_policy=False,
                )
            )
        comparisons.append(aggregate_trials(f"fixed_bet_{fixed_bet}", trial_summaries))

    learned_trial_summaries = []
    for trial_index in range(args.trials):
        trial_seed = args.seed + trial_index
        learned_trial_summaries.append(
            run_trial(
                policy_name="Learned Bet Policy",
                episodes=args.episodes,
                seed=trial_seed,
                shoe_size=args.shoe_size,
                starting_cash=args.starting_cash,
                action_q_table=action_q_table,
                bet_q_table=bet_q_table,
                preferred_bet=min(BET_OPTIONS),
                use_bet_policy=True,
            )
        )
    comparisons.append(aggregate_trials("learned_bet_policy", learned_trial_summaries))

    best_by_reward = max(comparisons, key=lambda item: item["average_round_reward"])
    summary = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - started_at, 2),
        "episodes_per_trial": args.episodes,
        "trials": args.trials,
        "action_model_path": str(action_model_path),
        "bet_model_path": str(bet_model_path),
        "comparisons": comparisons,
        "best_policy_by_average_round_reward": {
            "policy_name": best_by_reward["policy_name"],
            "average_round_reward": best_by_reward["average_round_reward"],
            "average_win_rate": best_by_reward["average_win_rate"],
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
