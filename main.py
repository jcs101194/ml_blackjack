"""Simple local runner for the blackjack engine."""

from __future__ import annotations

import json

from blackjack import BlackjackGame
from player import Player


def main() -> None:
    """Runs a short heuristic session and prints the final snapshot."""
    game = BlackjackGame()
    agent = Player()

    for _ in range(50):
        game.play_round(agent)

    print(json.dumps(game.get_state_snapshot(), indent=2))


if __name__ == "__main__":
    main()
