"""Tabular policy storage for blackjack action selection."""

from __future__ import annotations

import json
import random

from pathlib import Path
from typing import Any

DEFAULT_ACTIONS = ("hit", "stay", "double_down", "split")


class QTable:
    """Stores and updates Q-values for blackjack states."""

    def __init__(self, actions: tuple[str, ...] = DEFAULT_ACTIONS):
        """Initializes an empty Q-table.

        Args:
            actions: The action names supported by the table.
        """
        self.actions = tuple(actions)
        self._table: dict[str, dict[str, float]] = {}

    def _state_key(self, state: Any) -> str:
        """Converts a state object into a stable string key."""
        return json.dumps(state, sort_keys=True)

    def _ensure_state(self, state: Any) -> dict[str, float]:
        """Ensures a state entry exists before reading or writing it."""
        state_key = self._state_key(state)
        if state_key not in self._table:
            self._table[state_key] = {action: 0.0 for action in self.actions}
        return self._table[state_key]

    def get_action_values(self, state: Any) -> dict[str, float]:
        """Returns the action values for a given state."""
        return dict(self._ensure_state(state))

    def get_q_value(self, state: Any, action: str) -> float:
        """Reads one Q-value from the table."""
        return self._ensure_state(state)[action]

    def best_action(
        self,
        state: Any,
        valid_actions: tuple[str, ...] | list[str] | None = None,
        rng: random.Random | None = None,
    ) -> str:
        """Returns the highest-valued action for the supplied state."""
        action_values = self._ensure_state(state)
        candidate_actions = tuple(valid_actions) if valid_actions else self.actions
        best_value = max(action_values[action] for action in candidate_actions)
        best_actions = [action for action in candidate_actions if action_values[action] == best_value]
        random_source = rng or random
        return random_source.choice(best_actions)

    def choose_action(
        self,
        state: Any,
        valid_actions: tuple[str, ...] | list[str] | None = None,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ) -> str:
        """Chooses an action using epsilon-greedy exploration."""
        random_source = rng or random
        candidate_actions = tuple(valid_actions) if valid_actions else self.actions
        self._ensure_state(state)

        if random_source.random() < epsilon:
            return random_source.choice(candidate_actions)

        return self.best_action(state, candidate_actions, rng=random_source)

    def update(
        self,
        state: Any,
        action: str,
        reward: float,
        next_state: Any | None = None,
        done: bool = False,
        alpha: float = 0.1,
        gamma: float = 0.95,
        valid_next_actions: tuple[str, ...] | list[str] | None = None,
    ) -> float:
        """Updates one Q-value with a temporal-difference target.

        Args:
            state: The state where the action was taken.
            action: The selected action.
            reward: The observed reward for the transition.
            next_state: The state reached after taking the action.
            done: Whether the transition ended the episode.
            alpha: Learning rate.
            gamma: Discount factor.
            valid_next_actions: Optional subset of legal actions in the next state.

        Returns:
            The new Q-value for the state-action pair.
        """
        action_values = self._ensure_state(state)
        current_q = action_values[action]

        if done or next_state is None:
            target = reward
        else:
            next_values = self._ensure_state(next_state)
            next_actions = tuple(valid_next_actions) if valid_next_actions else self.actions
            target = reward + gamma * max(next_values[next_action] for next_action in next_actions)

        new_q = current_q + alpha * (target - current_q)
        action_values[action] = new_q
        return new_q

    def to_dict(self) -> dict[str, Any]:
        """Serializes the full Q-table."""
        return {
            "actions": list(self.actions),
            "states": self._table,
        }

    def save(self, output_path: str | Path) -> None:
        """Writes the Q-table to disk as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_path: str | Path) -> "QTable":
        """Loads a Q-table from disk if it exists."""
        path = Path(input_path)
        if not path.exists():
            return cls()

        payload = json.loads(path.read_text(encoding="utf-8"))
        q_table = cls(actions=tuple(payload.get("actions", DEFAULT_ACTIONS)))
        q_table._table = {
            state_key: {action: float(value) for action, value in action_values.items()}
            for state_key, action_values in payload.get("states", {}).items()
        }
        return q_table

    def __len__(self) -> int:
        """Returns the number of states currently stored."""
        return len(self._table)
