"""Player implementations for blackjack simulation and training."""

from __future__ import annotations

import random

from typing import Any

from blackjack import ACTIONS, BET_OPTIONS, ActionContext, BetContext, card_value
from q_table import QTable

BET_ACTIONS = tuple(f"bet_{bet}" for bet in BET_OPTIONS)


def bet_amount_to_action(amount: int) -> str:
    """Converts a numeric bet into a bet-policy action label."""
    return f"bet_{amount}"


def bet_action_to_amount(action: str) -> int:
    """Converts a bet-policy action label back into a numeric bet."""
    return int(action.removeprefix("bet_"))


def bankroll_bucket(cash: int) -> int:
    """Buckets bankroll values so the bet-policy state stays compact."""
    if cash < 250:
        return 0
    if cash < 500:
        return 1
    if cash < 1000:
        return 2
    if cash < 2500:
        return 3
    if cash < 5000:
        return 4
    if cash < 10000:
        return 5
    return 6


def shoe_remaining_bucket(remaining_cards: int, total_cards: int) -> int:
    """Buckets how deep into the shoe the table currently is."""
    if total_cards <= 0:
        return 0

    ratio = remaining_cards / total_cards
    if ratio < 0.2:
        return 0
    if ratio < 0.4:
        return 1
    if ratio < 0.6:
        return 2
    if ratio < 0.8:
        return 3
    return 4


def count_bucket(value: float, clip: int = 8) -> int:
    """Rounds and clips count-based signals into small discrete buckets."""
    return max(-clip, min(clip, int(round(value))))


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
        action_q_table: QTable | None = None,
        bet_q_table: QTable | None = None,
        q_table: QTable | None = None,
        action_epsilon: float = 0.10,
        bet_epsilon: float = 0.10,
        action_alpha: float = 0.10,
        bet_alpha: float = 0.10,
        action_gamma: float = 0.95,
        bet_gamma: float = 0.0,
        preferred_bet: int = 50,
        training_enabled: bool = True,
        train_action_policy: bool | None = None,
        train_bet_policy: bool = False,
        use_bet_policy: bool = False,
        name: str = "Q Player",
        rng: random.Random | None = None,
    ):
        """Initializes a Q-table-driven player.

        Args:
            action_q_table: Existing Q-table to use for hand-action selection.
            bet_q_table: Existing Q-table to use for bet selection.
            q_table: Backward-compatible alias for `action_q_table`.
            action_epsilon: Exploration rate for action-policy decisions.
            bet_epsilon: Exploration rate for bet-policy decisions.
            action_alpha: Learning rate for action-policy updates.
            bet_alpha: Learning rate for bet-policy updates.
            action_gamma: Discount factor for action-policy updates.
            bet_gamma: Discount factor for bet-policy updates.
            preferred_bet: Default bet to place when it is affordable.
            training_enabled: Backward-compatible flag for action-policy training.
            train_action_policy: Whether to update the action Q-table.
            train_bet_policy: Whether to update the bet Q-table.
            use_bet_policy: Whether to choose bets from the bet Q-table.
            name: Display name used in logs and API responses.
            rng: Optional random number generator for exploration decisions.
        """
        super().__init__(confidence=0.0, preferred_bet=preferred_bet, name=name)
        self.action_q_table = (
            action_q_table
            if action_q_table is not None
            else q_table
            if q_table is not None
            else QTable(actions=ACTIONS)
        )
        self.bet_q_table = bet_q_table if bet_q_table is not None else QTable(actions=BET_ACTIONS)
        self.action_epsilon = action_epsilon
        self.bet_epsilon = bet_epsilon
        self.action_alpha = action_alpha
        self.bet_alpha = bet_alpha
        self.action_gamma = action_gamma
        self.bet_gamma = bet_gamma
        self.train_action_policy = training_enabled if train_action_policy is None else train_action_policy
        self.train_bet_policy = train_bet_policy
        self.use_bet_policy = use_bet_policy
        self.rng = rng or random.Random()
        self.pending_traces: dict[str, list[dict[str, Any]]] = {}
        self.pending_bet_trace: dict[str, Any] | None = None

    @property
    def q_table(self) -> QTable:
        """Backwards-compatible alias for the action Q-table."""
        return self.action_q_table

    def encode_bet_state(self, context: BetContext) -> list[int]:
        """Encodes the table state used by the betting policy."""
        return [
            bankroll_bucket(context.cash),
            shoe_remaining_bucket(context.shoe_cards_remaining, context.total_shoe_cards),
            count_bucket(context.shoe_advantage_count),
            count_bucket(context.shoe_true_count),
        ]

    def choose_bet(self, context: BetContext) -> int:
        """Chooses the next bet using the bet policy when enabled."""
        if not self.use_bet_policy:
            return super().choose_bet(context)

        state = self.encode_bet_state(context)
        valid_actions = [bet_amount_to_action(bet) for bet in context.available_bets]
        selected_action = self.bet_q_table.choose_action(
            state=state,
            valid_actions=valid_actions,
            epsilon=self.bet_epsilon,
            rng=self.rng,
        )
        self.pending_bet_trace = {
            "state": state,
            "action": selected_action,
        }
        return bet_action_to_amount(selected_action)

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
        action = self.action_q_table.choose_action(
            state=state,
            valid_actions=context.valid_actions,
            epsilon=self.action_epsilon,
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
        return self.action_q_table.get_action_values(self.encode_state(context))

    def on_round_complete(self, round_summary: dict[str, Any]) -> None:
        """Updates the Q-table from the completed round's outcomes."""
        if not self.train_action_policy:
            self.pending_traces.clear()
        else:
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

                    self.action_q_table.update(
                        state=step["state"],
                        action=step["action"],
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        alpha=self.action_alpha,
                        gamma=self.action_gamma,
                        valid_next_actions=next_actions,
                    )

            self.pending_traces.clear()

        if self.train_bet_policy and self.pending_bet_trace is not None:
            round_reward = float(round_summary.get("round_net_reward", 0.0))
            self.bet_q_table.update(
                state=self.pending_bet_trace["state"],
                action=self.pending_bet_trace["action"],
                reward=round_reward,
                next_state=None,
                done=True,
                alpha=self.bet_alpha,
                gamma=self.bet_gamma,
            )

        self.pending_bet_trace = None
