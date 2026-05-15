"""Player implementations for blackjack simulation and training."""

from __future__ import annotations

import random

from typing import Any

from blackjack import ActionContext, BetContext, card_value
from q_table import QTable


class Player:
    """Baseline heuristic blackjack player."""

    def __init__(
        self,
        confidence: float = 0.60,
        preferred_bet: int = 50,
        name: str = "Heuristic Player",
    ):
        """Initializes the heuristic player.

        Args:
            confidence: Minimum safe-draw probability required to hit.
            preferred_bet: Default bet to place when it is affordable.
            name: Display name used in logs and API responses.
        """
        self.confidence_threshold = confidence
        self.preferred_bet = preferred_bet
        self.name = name

    def choose_bet(self, context: BetContext) -> int:
        """Chooses the round's bet amount."""
        if self.preferred_bet in context.available_bets:
            return self.preferred_bet
        return context.available_bets[0]

    def choose_action(self, context: ActionContext) -> str:
        """Chooses a blackjack action using simple heuristics."""
        cards = context.hand_cards

        if context.can_split and len(cards) == 2 and cards[0] == cards[1]:
            if cards[0] in ("A", 8):
                return "split"

        if context.can_double and context.hand_score in (10, 11) and context.dealer_up_value <= 9:
            return "double_down"

        if "hit" in context.valid_actions and context.safe_draw_probability >= self.confidence_threshold:
            return "hit"

        return "stay"

    def on_round_complete(self, round_summary: dict[str, Any]) -> None:
        """Receives the round summary after resolution."""


class QPlayer(Player):
    """Blackjack player that selects actions from a Q-table."""

    def __init__(
        self,
        q_table: QTable | None = None,
        epsilon: float = 0.10,
        alpha: float = 0.10,
        gamma: float = 0.95,
        preferred_bet: int = 50,
        training_enabled: bool = True,
        name: str = "Q Player",
        rng: random.Random | None = None,
    ):
        """Initializes a Q-table-driven player.

        Args:
            q_table: Existing Q-table to use for action selection.
            epsilon: Exploration rate for epsilon-greedy decisions.
            alpha: Learning rate for Q-value updates.
            gamma: Discount factor for future rewards.
            preferred_bet: Default bet to place when it is affordable.
            training_enabled: Whether the player should update the Q-table.
            name: Display name used in logs and API responses.
            rng: Optional random number generator for exploration decisions.
        """
        super().__init__(confidence=0.0, preferred_bet=preferred_bet, name=name)
        self.q_table = q_table if q_table is not None else QTable()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training_enabled = training_enabled
        self.rng = rng or random.Random()
        self.pending_traces: dict[str, list[dict[str, Any]]] = {}

    def encode_state(self, context: ActionContext) -> list[int]:
        """Encodes the current hand into a compact Q-table state."""
        pair_value = 0
        if len(context.hand_cards) == 2 and context.hand_cards[0] == context.hand_cards[1]:
            pair_value = card_value(context.hand_cards[0])

        return [
            context.hand_score,
            context.dealer_up_value,
            int(context.usable_ace),
            pair_value,
            int(context.can_double),
            int(context.is_split_hand),
        ]

    def choose_action(self, context: ActionContext) -> str:
        """Chooses the next action using epsilon-greedy exploration."""
        state = self.encode_state(context)
        action = self.q_table.choose_action(
            state=state,
            valid_actions=context.valid_actions,
            epsilon=self.epsilon,
            rng=self.rng,
        )

        self.pending_traces.setdefault(context.hand_id, []).append(
            {
                "state": state,
                "action": action,
                "valid_actions": list(context.valid_actions),
            }
        )
        return action

    def get_action_values(self, context: ActionContext) -> dict[str, float]:
        """Returns the Q-values for the current decision context."""
        return self.q_table.get_action_values(self.encode_state(context))

    def on_round_complete(self, round_summary: dict[str, Any]) -> None:
        """Updates the Q-table from the completed round's outcomes."""
        if not self.training_enabled:
            self.pending_traces.clear()
            return

        training_rewards = round_summary.get("training_rewards", {})

        for hand_id, steps in self.pending_traces.items():
            if not steps:
                continue

            final_reward = float(training_rewards.get(hand_id, 0.0))
            for index, step in enumerate(steps):
                next_step = steps[index + 1] if index + 1 < len(steps) else None
                done = next_step is None
                reward = final_reward if done else 0.0
                next_state = next_step["state"] if next_step else None
                next_actions = next_step["valid_actions"] if next_step else None

                self.q_table.update(
                    state=step["state"],
                    action=step["action"],
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    valid_next_actions=next_actions,
                )

        self.pending_traces.clear()
