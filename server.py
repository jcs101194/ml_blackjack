"""Persistent blackjack API server with a background simulation worker."""

from __future__ import annotations

import argparse
import json
import threading
import time

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from blackjack import BlackjackGame
from player import BET_ACTIONS, Player, QPlayer
from q_table import QTable


class BlackjackService:
    """Owns the live game session and background agent worker."""

    def __init__(
        self,
        model_path: str,
        bet_model_path: str = "checkpoints/blackjack_bet_q_table.json",
        mode: str = "q",
        shoe_size: int = 6,
        starting_cash: int = 10000,
        preferred_bet: int = 50,
        q_epsilon: float = 0.02,
        training_epsilon: float = 0.12,
        alpha: float = 0.10,
        gamma: float = 0.95,
        round_delay: float = 0.35,
        save_every: int = 25,
    ):
        """Initializes the live blackjack service.

        Args:
            model_path: Location of the persisted action Q-table on disk.
            bet_model_path: Location of the persisted betting Q-table on disk.
            mode: Player mode: `heuristic`, `q`, or `q-train`.
            shoe_size: Number of decks in the live shoe.
            starting_cash: Starting bankroll for the live session.
            preferred_bet: Default bet size for the active player.
            q_epsilon: Exploration rate when running the live Q-player.
            training_epsilon: Exploration rate when the live service is training.
            alpha: Learning rate for online Q updates.
            gamma: Discount factor for future Q rewards.
            round_delay: Delay between rounds so front-end updates remain readable.
            save_every: Persist the Q-table every N rounds.
        """
        self.action_model_path = Path(model_path)
        self.bet_model_path = Path(bet_model_path)
        self.mode = mode
        self.q_epsilon = q_epsilon
        self.training_epsilon = training_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.round_delay = round_delay
        self.save_every = save_every
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.mode_note = ""
        self.rounds_since_save = 0

        self.game = BlackjackGame(
            shoe_size=shoe_size,
            starting_cash=starting_cash,
            auto_rebuy=True,
        )
        self.player = self._build_player(preferred_bet)

    def _build_player(self, preferred_bet: int) -> Player:
        """Builds the player used by the live background worker."""
        if self.mode == "heuristic":
            self.mode_note = "Running the heuristic player."
            return Player(preferred_bet=preferred_bet)

        action_q_table = QTable.load(self.action_model_path)
        bet_q_table = QTable.load(self.bet_model_path, actions=BET_ACTIONS)

        if self.mode == "q" and len(action_q_table) == 0:
            self.mode_note = "No saved Q-table found. Falling back to the heuristic player."
            return Player(preferred_bet=preferred_bet, name="Heuristic Fallback")

        if self.mode == "q-train":
            if len(bet_q_table) > 0:
                self.mode_note = "Running the Q-player with online learning enabled for actions and bets."
            else:
                self.mode_note = "Running the Q-player with online learning enabled. Betting stays fixed until a bet policy is trained."
            return QPlayer(
                action_q_table=action_q_table,
                bet_q_table=bet_q_table,
                action_epsilon=self.training_epsilon,
                bet_epsilon=self.training_epsilon,
                action_alpha=self.alpha,
                bet_alpha=self.alpha,
                action_gamma=self.gamma,
                bet_gamma=0.0,
                preferred_bet=preferred_bet,
                training_enabled=True,
                train_bet_policy=True,
                use_bet_policy=len(bet_q_table) > 0,
                name="Live Q Trainer",
            )

        if len(bet_q_table) > 0:
            self.mode_note = "Running the Q-player in live inference mode with the trained bet policy."
        else:
            self.mode_note = "Running the Q-player in live inference mode with a fixed preferred bet."
        return QPlayer(
            action_q_table=action_q_table,
            bet_q_table=bet_q_table,
            action_epsilon=self.q_epsilon,
            bet_epsilon=0.0,
            action_alpha=self.alpha,
            bet_alpha=self.alpha,
            action_gamma=self.gamma,
            bet_gamma=0.0,
            preferred_bet=preferred_bet,
            training_enabled=False,
            train_bet_policy=False,
            use_bet_policy=len(bet_q_table) > 0,
            name="Live Q Player",
        )

    def start(self) -> None:
        """Starts the background worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        """Signals the background worker to stop."""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

    def _save_model_if_needed(self) -> None:
        """Persists the Q-table when the active player supports it."""
        if not isinstance(self.player, QPlayer):
            return

        self.rounds_since_save += 1
        if self.rounds_since_save >= self.save_every:
            self.player.action_q_table.save(self.action_model_path)
            self.player.bet_q_table.save(self.bet_model_path)
            self.rounds_since_save = 0

    def _run(self) -> None:
        """Continuously advances the live blackjack session."""
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    self.game.play_round(self.player)
                    self._save_model_if_needed()
                    self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
            finally:
                self.stop_event.wait(self.round_delay)

    def get_state(self) -> dict[str, Any]:
        """Returns the latest live blackjack state for API consumers."""
        with self.lock:
            snapshot = self.game.get_state_snapshot()
            snapshot["service"] = {
                "mode": self.mode,
                "note": self.mode_note,
                "action_model_path": str(self.action_model_path),
                "bet_model_path": str(self.bet_model_path),
                "action_q_table_states": len(self.player.action_q_table) if isinstance(self.player, QPlayer) else 0,
                "bet_q_table_states": len(self.player.bet_q_table) if isinstance(self.player, QPlayer) else 0,
                "q_table_states": len(self.player.action_q_table) if isinstance(self.player, QPlayer) else 0,
                "worker_alive": bool(self.worker_thread and self.worker_thread.is_alive()),
                "last_error": self.last_error,
            }
            return snapshot

    def get_health(self) -> dict[str, Any]:
        """Returns a compact health response."""
        return {
            "status": "ok" if self.last_error is None else "degraded",
            "worker_alive": bool(self.worker_thread and self.worker_thread.is_alive()),
            "mode": self.mode,
            "last_error": self.last_error,
        }


def build_handler(service: BlackjackService, api_prefix: str):
    """Builds an HTTP request handler bound to the blackjack service."""

    class BlackjackHandler(BaseHTTPRequestHandler):
        """HTTP handler for blackjack API endpoints."""

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            """Handles browser preflight requests."""
            self._send_json({"status": "ok"})

        def do_GET(self) -> None:  # noqa: N802
            """Routes GET requests to the blackjack API."""
            parsed = urlparse(self.path)
            if parsed.path == f"{api_prefix}/state":
                self._send_json(service.get_state())
                return

            if parsed.path == f"{api_prefix}/health":
                self._send_json(service.get_health())
                return

            self._send_json(
                {
                    "error": "Not found.",
                    "available_endpoints": [
                        f"{api_prefix}/state",
                        f"{api_prefix}/health",
                    ],
                },
                status=HTTPStatus.NOT_FOUND,
            )

        def log_message(self, format: str, *args: Any) -> None:
            """Silences default request logging."""

    return BlackjackHandler


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the live blackjack server."""
    parser = argparse.ArgumentParser(description="Run the live blackjack API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the API server.")
    parser.add_argument(
        "--api-prefix",
        default="/api/blackjack",
        help="URL prefix for the blackjack API.",
    )
    parser.add_argument(
        "--mode",
        choices=("heuristic", "q", "q-train"),
        default="q",
        help="Player mode for the background worker.",
    )
    parser.add_argument("--shoe-size", type=int, default=6, help="Number of decks in the shoe.")
    parser.add_argument("--starting-cash", type=int, default=10000, help="Starting bankroll.")
    parser.add_argument("--preferred-bet", type=int, default=50, help="Default player bet.")
    parser.add_argument(
        "--model-path",
        default="checkpoints/blackjack_q_table.json",
        help="Path to the saved action Q-table model.",
    )
    parser.add_argument(
        "--bet-model-path",
        default="checkpoints/blackjack_bet_q_table.json",
        help="Path to the saved betting Q-table model.",
    )
    parser.add_argument("--q-epsilon", type=float, default=0.02, help="Exploration rate for live Q mode.")
    parser.add_argument(
        "--training-epsilon",
        type=float,
        default=0.12,
        help="Exploration rate for live Q training mode.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--round-delay", type=float, default=0.35, help="Delay between rounds.")
    parser.add_argument("--save-every", type=int, default=25, help="Persist the model every N rounds.")
    return parser.parse_args()


def main() -> None:
    """Starts the blackjack service and HTTP server."""
    args = parse_args()
    service = BlackjackService(
        model_path=args.model_path,
        bet_model_path=args.bet_model_path,
        mode=args.mode,
        shoe_size=args.shoe_size,
        starting_cash=args.starting_cash,
        preferred_bet=args.preferred_bet,
        q_epsilon=args.q_epsilon,
        training_epsilon=args.training_epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        round_delay=args.round_delay,
        save_every=args.save_every,
    )
    service.start()

    server = ThreadingHTTPServer((args.host, args.port), build_handler(service, args.api_prefix))
    print(f"Blackjack API listening on http://{args.host}:{args.port}{args.api_prefix}/state")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.stop()
        if isinstance(service.player, QPlayer):
            service.player.action_q_table.save(service.action_model_path)
            service.player.bet_q_table.save(service.bet_model_path)


if __name__ == "__main__":
    main()
