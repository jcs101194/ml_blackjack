"""Reusable blackjack engine for training, simulation, and API serving."""

from __future__ import annotations

import random
import time

from collections import deque
from dataclasses import dataclass, field
from typing import Any

CARD_TEMPLATE = ("A", 2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K")
FACE_CARDS = {"J", "Q", "K"}
HIGH_CARDS = {"A", 10, "J", "Q", "K"}
LOW_CARDS = {2, 3, 4, 5, 6}
BET_OPTIONS = (50, 100, 250, 500, 1000)
ACTIONS = ("hit", "stay", "double_down", "split")


def card_value(card: Any) -> int:
    """Converts a card rank into its blackjack value."""
    if card == "A":
        return 11
    if card in FACE_CARDS:
        return 10
    return int(card)


def calculate_score(cards: list[Any]) -> int:
    """Calculates the best blackjack score for a set of cards.

    Args:
        cards: The cards in the hand being evaluated.

    Returns:
        The highest non-busting score for the hand. If every interpretation
        busts, the returned score will be greater than 21.
    """
    total = 0
    ace_count = 0

    for card in cards:
        if card == "A":
            ace_count += 1
        elif card in FACE_CARDS:
            total += 10
        else:
            total += int(card)

    total += ace_count
    if ace_count and total + 10 <= 21:
        total += 10

    return total


def has_usable_ace(cards: list[Any]) -> bool:
    """Checks whether the hand currently contains an ace counted as 11."""
    non_ace_total = 0
    ace_count = 0

    for card in cards:
        if card == "A":
            ace_count += 1
        elif card in FACE_CARDS:
            non_ace_total += 10
        else:
            non_ace_total += int(card)

    return ace_count > 0 and non_ace_total + ace_count + 10 <= 21


def is_blackjack(cards: list[Any]) -> bool:
    """Checks whether a hand is a natural blackjack."""
    return len(cards) == 2 and calculate_score(cards) == 21


def is_pair(cards: list[Any]) -> bool:
    """Checks whether the opening hand is an exact pair."""
    return len(cards) == 2 and cards[0] == cards[1]


@dataclass(frozen=True)
class BetContext:
    """Context passed to a player when choosing a bet."""

    round_number: int
    cash: int
    available_bets: tuple[int, ...]
    total_shoe_cards: int
    shoe_cards_remaining: int
    shoe_high_cards_remaining: int
    shoe_low_cards_remaining: int
    shoe_advantage_count: int
    shoe_true_count: float
    prompt: str
    stats: dict[str, Any]


@dataclass(frozen=True)
class ActionContext:
    """Context passed to a player when choosing a hand action."""

    round_number: int
    hand_id: str
    hand_index: int
    hand_cards: tuple[Any, ...]
    hand_score: int
    hand_bet: int
    dealer_up_card: Any
    dealer_up_value: int
    cash: int
    valid_actions: tuple[str, ...]
    can_double: bool
    can_split: bool
    is_split_hand: bool
    split_from_aces: bool
    usable_ace: bool
    safe_card_count: int
    safe_draw_probability: float
    shoe_cards_remaining: int
    prompt: str
    stats: dict[str, Any]


@dataclass
class Hand:
    """Mutable state for an in-progress blackjack hand."""

    hand_id: str
    cards: list[Any]
    bet: int
    parent_hand_id: str | None = None
    is_split_hand: bool = False
    split_from_aces: bool = False
    doubled_down: bool = False
    finished: bool = False
    actions_taken: list[str] = field(default_factory=list)
    resolution: str | None = None
    outcome: str | None = None
    payout: int = 0
    net_reward: int = 0

    @property
    def score(self) -> int:
        """Returns the current blackjack score for the hand."""
        return calculate_score(self.cards)

    @property
    def is_bust(self) -> bool:
        """Checks whether the hand has busted."""
        return self.score > 21

    @property
    def can_double(self) -> bool:
        """Checks whether the hand is still eligible to double down."""
        return len(self.actions_taken) == 0

    @property
    def can_split(self) -> bool:
        """Checks whether the hand can currently be split."""
        return len(self.actions_taken) == 0 and is_pair(self.cards)


@dataclass
class SessionStats:
    """Long-lived aggregate stats for a blackjack session."""

    starting_cash: int = 10000
    cash: int = 10000
    most_cash: int = 10000
    rounds_played: int = 0
    hands_played: int = 0
    hands_won: int = 0
    house_won: int = 0
    draws: int = 0
    player_busts: int = 0
    player_has_more: int = 0
    player_blackjack_count: int = 0
    dealer_busts: int = 0
    dealer_has_more: int = 0
    dealer_blackjack_count: int = 0
    double_downs: int = 0
    splits: int = 0
    bankroll_resets: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serializes stats for API responses."""
        total_decisions = self.hands_won + self.house_won + self.draws
        win_rate = round(self.hands_won / total_decisions, 4) if total_decisions else 0.0

        return {
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "most_cash": self.most_cash,
            "rounds_played": self.rounds_played,
            "hands_played": self.hands_played,
            "hands_won": self.hands_won,
            "house_won": self.house_won,
            "draws": self.draws,
            "player_busts": self.player_busts,
            "player_has_more": self.player_has_more,
            "player_blackjack_count": self.player_blackjack_count,
            "dealer_busts": self.dealer_busts,
            "dealer_has_more": self.dealer_has_more,
            "dealer_blackjack_count": self.dealer_blackjack_count,
            "double_downs": self.double_downs,
            "splits": self.splits,
            "bankroll_resets": self.bankroll_resets,
            "time_at_table_seconds": round(time.time() - self.started_at, 2),
            "win_rate": win_rate,
        }


class Shoe:
    """Represents the card shoe and all shoe-related behaviors."""

    def __init__(self, shoe_size: int = 6, rng: random.Random | None = None):
        """Initializes a shoe with the requested number of decks.

        Args:
            shoe_size: The number of decks in the shoe.
            rng: Optional random number generator used for dealing cards.
        """
        self.shoe_size = shoe_size
        self.rng = rng or random.Random()
        self.shuffle_count = 0
        self.total_cards = self.shoe_size * len(CARD_TEMPLATE)
        self._decks: list[list[Any]] = []
        self.reset()

    def reset(self) -> None:
        """Resets the shoe to a fresh set of decks."""
        self._decks = [list(CARD_TEMPLATE) for _ in range(self.shoe_size)]
        self.shuffle_count += 1

    def is_empty(self) -> bool:
        """Checks whether every deck in the shoe has been exhausted."""
        return not any(self._decks)

    def cards_remaining(self) -> int:
        """Counts how many cards remain in the shoe."""
        return sum(len(deck) for deck in self._decks)

    def iter_cards(self) -> list[Any]:
        """Returns a flat list of every remaining card in the shoe."""
        cards: list[Any] = []
        for deck in self._decks:
            cards.extend(deck)
        return cards

    def count_safe_cards(self, highest_value: int) -> int:
        """Counts draws that would not immediately bust the current hand.

        Args:
            highest_value: The highest card value that can be safely drawn.

        Returns:
            The number of remaining cards that are safe to draw.
        """
        if highest_value <= 0:
            return 0

        safe_count = 0
        for card in self.iter_cards():
            if card == "A":
                safe_count += 1
            elif card in FACE_CARDS:
                if highest_value >= 10:
                    safe_count += 1
            elif int(card) <= highest_value:
                safe_count += 1

        return safe_count

    def count_high_cards(self) -> int:
        """Counts the remaining high-value cards in the shoe."""
        return sum(1 for card in self.iter_cards() if card in HIGH_CARDS)

    def count_low_cards(self) -> int:
        """Counts the remaining low-value cards in the shoe."""
        return sum(1 for card in self.iter_cards() if card in LOW_CARDS)

    def player_advantage_count(self) -> int:
        """Measures whether the shoe is high-card heavy or low-card heavy.

        A positive value means more high cards than low cards remain, which is
        usually better for the player when deciding how much to bet.
        """
        return self.count_high_cards() - self.count_low_cards()

    def player_true_count(self) -> float:
        """Normalizes the shoe advantage by the estimated decks remaining."""
        decks_remaining = self.cards_remaining() / len(CARD_TEMPLATE)
        if decks_remaining <= 0:
            return 0.0
        return self.player_advantage_count() / decks_remaining

    def deal_card(self) -> Any:
        """Deals a random card from a non-empty deck in the shoe."""
        if self.is_empty():
            self.reset()

        nonempty_decks = [deck for deck in self._decks if deck]
        selected_deck = self.rng.choice(nonempty_decks)
        selected_index = self.rng.randrange(len(selected_deck))
        card = selected_deck.pop(selected_index)

        if self.is_empty():
            self.reset()

        return card

    def snapshot(self) -> dict[str, Any]:
        """Serializes the current shoe for API responses."""
        return {
            "shoe_size": self.shoe_size,
            "cards_remaining": self.cards_remaining(),
            "shuffle_count": self.shuffle_count,
        }


class BlackjackGame:
    """Runs blackjack rounds and captures live table state."""

    def __init__(
        self,
        shoe_size: int = 6,
        starting_cash: int = 10000,
        bet_options: tuple[int, ...] = BET_OPTIONS,
        event_log_size: int = 200,
        rng: random.Random | None = None,
        auto_rebuy: bool = True,
    ):
        """Initializes a blackjack game session.

        Args:
            shoe_size: Number of decks in the shoe.
            starting_cash: Initial bankroll for the player.
            bet_options: Allowed betting options for each round.
            event_log_size: Maximum number of recent events to retain.
            rng: Optional random number generator used by the shoe.
            auto_rebuy: Whether the bankroll should reset after bankruptcy.
        """
        self.bet_options = tuple(sorted(bet_options))
        self.auto_rebuy = auto_rebuy
        self.shoe = Shoe(shoe_size=shoe_size, rng=rng)
        self.stats = SessionStats(
            starting_cash=starting_cash,
            cash=starting_cash,
            most_cash=starting_cash,
        )
        self.event_log: deque[dict[str, Any]] = deque(maxlen=event_log_size)
        self.event_counter = 0
        self.round_number = 0
        self._current_hand_sequence = 0
        self.current_phase = "idle"
        self.current_prompt = ""
        self.current_agent_name = ""
        self.current_bet = 0
        self.current_player_hands: list[Hand] = []
        self.current_dealer_hand: list[Any] = []
        self.last_round_summary: dict[str, Any] | None = None

    def _make_hand_id(self) -> str:
        """Generates a unique hand identifier for the current round."""
        self._current_hand_sequence += 1
        return f"round-{self.round_number}-hand-{self._current_hand_sequence}"

    def log_event(self, event_type: str, message: str, **payload: Any) -> None:
        """Appends a timestamped event to the recent table log."""
        self.event_counter += 1
        event = {
            "id": self.event_counter,
            "timestamp": round(time.time(), 3),
            "round_number": self.round_number,
            "event_type": event_type,
            "message": message,
        }
        event.update(payload)
        self.event_log.append(event)

    def _reset_bankroll(self) -> None:
        """Restores the bankroll so automated sessions can keep running."""
        self.stats.cash = self.stats.starting_cash
        self.stats.bankroll_resets += 1
        self.current_bet = 0
        self.log_event(
            "bankroll_reset",
            f"Player rebought for ${self.stats.starting_cash}.",
            stats=self.stats.to_dict(),
        )

    def _available_bets(self) -> tuple[int, ...]:
        """Returns the betting options that are currently affordable."""
        return tuple(bet for bet in self.bet_options if bet <= self.stats.cash)

    def _build_bet_prompt(self, available_bets: tuple[int, ...]) -> str:
        """Formats the current bet-selection prompt."""
        options = "\n".join(f"- ${bet}" for bet in available_bets)
        return f"Cash available: ${self.stats.cash}\nChoose a bet:\n{options}"

    def _build_action_prompt(self, valid_actions: tuple[str, ...]) -> str:
        """Formats the current hand-action prompt."""
        labels = {
            "hit": "hit",
            "stay": "stay",
            "double_down": "double down",
            "split": "split",
        }
        action_lines = "\n".join(f"- {labels[action]}" for action in valid_actions)
        return f"Choose an action:\n{action_lines}"

    def _build_bet_context(self) -> BetContext:
        """Builds the context object for a player's betting decision."""
        available_bets = self._available_bets()
        prompt = self._build_bet_prompt(available_bets)
        self.current_prompt = prompt
        return BetContext(
            round_number=self.round_number,
            cash=self.stats.cash,
            available_bets=available_bets,
            total_shoe_cards=self.shoe.total_cards,
            shoe_cards_remaining=self.shoe.cards_remaining(),
            shoe_high_cards_remaining=self.shoe.count_high_cards(),
            shoe_low_cards_remaining=self.shoe.count_low_cards(),
            shoe_advantage_count=self.shoe.player_advantage_count(),
            shoe_true_count=round(self.shoe.player_true_count(), 4),
            prompt=prompt,
            stats=self.stats.to_dict(),
        )

    def _build_action_context(self, hand: Hand, hand_index: int) -> ActionContext:
        """Builds the context object for a player's action decision."""
        score_limit = 21 - hand.score
        safe_card_count = self.shoe.count_safe_cards(score_limit)
        cards_remaining = self.shoe.cards_remaining()
        safe_draw_probability = safe_card_count / cards_remaining if cards_remaining else 0.0
        valid_actions = self.get_valid_actions(hand)
        prompt = self._build_action_prompt(valid_actions)
        self.current_prompt = prompt
        dealer_up_card = self.current_dealer_hand[0]

        return ActionContext(
            round_number=self.round_number,
            hand_id=hand.hand_id,
            hand_index=hand_index,
            hand_cards=tuple(hand.cards),
            hand_score=hand.score,
            hand_bet=hand.bet,
            dealer_up_card=dealer_up_card,
            dealer_up_value=card_value(dealer_up_card),
            cash=self.stats.cash,
            valid_actions=valid_actions,
            can_double="double_down" in valid_actions,
            can_split="split" in valid_actions,
            is_split_hand=hand.is_split_hand,
            split_from_aces=hand.split_from_aces,
            usable_ace=has_usable_ace(hand.cards),
            safe_card_count=safe_card_count,
            safe_draw_probability=round(safe_draw_probability, 4),
            shoe_cards_remaining=cards_remaining,
            prompt=prompt,
            stats=self.stats.to_dict(),
        )

    def get_valid_actions(self, hand: Hand) -> tuple[str, ...]:
        """Returns the actions currently allowed for a hand."""
        if hand.finished:
            return tuple()

        actions = ["hit", "stay"]
        if hand.can_double and self.stats.cash >= hand.bet and not hand.is_split_hand:
            actions.append("double_down")
        if (
            hand.can_split
            and not hand.is_split_hand
            and self.stats.cash >= hand.bet
        ):
            actions.append("split")

        return tuple(actions)

    def _serialize_hand(self, hand: Hand) -> dict[str, Any]:
        """Converts a hand into a JSON-safe dictionary."""
        return {
            "hand_id": hand.hand_id,
            "parent_hand_id": hand.parent_hand_id,
            "cards": list(hand.cards),
            "score": hand.score,
            "bet": hand.bet,
            "is_split_hand": hand.is_split_hand,
            "split_from_aces": hand.split_from_aces,
            "doubled_down": hand.doubled_down,
            "finished": hand.finished,
            "actions_taken": list(hand.actions_taken),
            "resolution": hand.resolution,
            "outcome": hand.outcome,
            "payout": hand.payout,
            "net_reward": hand.net_reward,
        }

    def _snapshot_round_state(self) -> dict[str, Any]:
        """Captures the live state of the current round for API consumers."""
        reveal_dealer = self.current_phase in {"dealer_turn", "round_complete"}
        dealer_cards = list(self.current_dealer_hand)
        visible_cards = dealer_cards if reveal_dealer else dealer_cards[:1]

        return {
            "round_number": self.round_number,
            "phase": self.current_phase,
            "prompt": self.current_prompt,
            "current_bet": self.current_bet,
            "dealer": {
                "visible_cards": visible_cards,
                "score": calculate_score(visible_cards) if visible_cards else 0,
                "full_cards": dealer_cards if reveal_dealer else [],
                "full_score": calculate_score(dealer_cards) if dealer_cards else 0,
            },
            "player_hands": [self._serialize_hand(hand) for hand in self.current_player_hands],
            "shoe": self.shoe.snapshot(),
        }

    def get_state_snapshot(self, recent_event_limit: int = 40) -> dict[str, Any]:
        """Builds the latest public state for APIs or front-end polling."""
        events = list(self.event_log)[-recent_event_limit:]
        return {
            "agent_name": self.current_agent_name,
            "stats": self.stats.to_dict(),
            "table": self._snapshot_round_state(),
            "last_round": self.last_round_summary,
            "recent_events": events,
        }

    def _deal_initial_cards(self, hands: list[Hand]) -> None:
        """Deals the opening cards for the round."""
        self.current_dealer_hand = []
        for _ in range(2):
            for hand in hands:
                hand.cards.append(self.shoe.deal_card())
            self.current_dealer_hand.append(self.shoe.deal_card())

    def _deal_to_hand(self, hand: Hand) -> None:
        """Deals one card to a hand and logs the result."""
        card = self.shoe.deal_card()
        hand.cards.append(card)
        self.log_event(
            "card_dealt",
            f"Dealt {card} to {hand.hand_id}.",
            hand_id=hand.hand_id,
            cards=list(hand.cards),
            score=hand.score,
        )

    def _split_hand(self, hands: list[Hand], hand_index: int) -> None:
        """Splits one pair into two child hands using the configured rules."""
        parent_hand = hands[hand_index]
        first_card, second_card = parent_hand.cards
        self.stats.cash -= parent_hand.bet
        self.stats.splits += 1

        child_one = Hand(
            hand_id=self._make_hand_id(),
            cards=[first_card],
            bet=parent_hand.bet,
            parent_hand_id=parent_hand.hand_id,
            is_split_hand=True,
            split_from_aces=first_card == "A",
        )
        child_two = Hand(
            hand_id=self._make_hand_id(),
            cards=[second_card],
            bet=parent_hand.bet,
            parent_hand_id=parent_hand.hand_id,
            is_split_hand=True,
            split_from_aces=second_card == "A",
        )

        hands[hand_index:hand_index + 1] = [child_one, child_two]
        self.current_player_hands = hands

        self._deal_to_hand(child_one)
        self._deal_to_hand(child_two)

        if child_one.split_from_aces:
            child_one.finished = True
        if child_two.split_from_aces:
            child_two.finished = True

        self.log_event(
            "hand_split",
            f"Split {parent_hand.hand_id} into {child_one.hand_id} and {child_two.hand_id}.",
            parent_hand_id=parent_hand.hand_id,
            child_hand_ids=[child_one.hand_id, child_two.hand_id],
            remaining_cash=self.stats.cash,
        )

    def _play_hand(self, agent: Any, hands: list[Hand], hand_index: int) -> int:
        """Executes the action loop for one player hand."""
        hand = hands[hand_index]
        if hand.finished:
            return hand_index + 1

        if is_blackjack(hand.cards):
            hand.finished = True
            return hand_index + 1

        while not hand.finished and not hand.is_bust:
            action_context = self._build_action_context(hand, hand_index)
            self.current_phase = "player_turn"
            self.log_event(
                "prompt_shown",
                action_context.prompt,
                hand_id=hand.hand_id,
                valid_actions=list(action_context.valid_actions),
                hand=self._serialize_hand(hand),
                dealer_up_card=action_context.dealer_up_card,
            )

            action = agent.choose_action(action_context)
            if action not in action_context.valid_actions:
                action = "stay"

            hand.actions_taken.append(action)
            self.log_event(
                "player_action",
                f"{self.current_agent_name} chose {action} on {hand.hand_id}.",
                hand_id=hand.hand_id,
                action=action,
            )

            if action == "hit":
                self._deal_to_hand(hand)
                if hand.is_bust or hand.score == 21:
                    hand.finished = True
            elif action == "stay":
                hand.finished = True
            elif action == "double_down":
                self.stats.cash -= hand.bet
                hand.bet *= 2
                hand.doubled_down = True
                hand.finished = True
                self.stats.double_downs += 1
                self._deal_to_hand(hand)
            elif action == "split":
                self._split_hand(hands, hand_index)
                return hand_index

        self.current_player_hands = hands
        return hand_index + 1

    def _dealer_should_play(self, hands: list[Hand]) -> bool:
        """Checks whether the dealer needs to draw additional cards."""
        return any(not hand.is_bust for hand in hands)

    def _play_dealer(self) -> None:
        """Runs the dealer draw logic."""
        self.current_phase = "dealer_turn"
        self.log_event(
            "dealer_turn_started",
            "Dealer begins drawing cards.",
            dealer_cards=list(self.current_dealer_hand),
            dealer_score=calculate_score(self.current_dealer_hand),
        )

        while calculate_score(self.current_dealer_hand) < 17:
            card = self.shoe.deal_card()
            self.current_dealer_hand.append(card)
            self.log_event(
                "dealer_action",
                f"Dealer drew {card}.",
                dealer_cards=list(self.current_dealer_hand),
                dealer_score=calculate_score(self.current_dealer_hand),
            )

    def _resolve_hand(self, hand: Hand, dealer_score: int, dealer_blackjack: bool) -> None:
        """Applies payout and stats for a finished hand."""
        player_score = hand.score
        player_blackjack = is_blackjack(hand.cards) and not hand.is_split_hand

        if hand.is_bust:
            hand.resolution = "player_bust"
            hand.outcome = "loss"
            hand.net_reward = -hand.bet
            self.stats.player_busts += 1
            self.stats.house_won += 1
        elif dealer_score > 21:
            hand.resolution = "dealer_bust"
            hand.outcome = "win"
            hand.payout = 2 * hand.bet
            hand.net_reward = hand.bet
            self.stats.cash += hand.payout
            self.stats.hands_won += 1
            self.stats.dealer_busts += 1
        elif player_blackjack and not dealer_blackjack:
            hand.resolution = "player_blackjack"
            hand.outcome = "win"
            hand.payout = 2 * hand.bet
            hand.net_reward = hand.bet
            self.stats.cash += hand.payout
            self.stats.hands_won += 1
            self.stats.player_blackjack_count += 1
        elif dealer_blackjack and not player_blackjack:
            hand.resolution = "dealer_blackjack"
            hand.outcome = "loss"
            hand.net_reward = -hand.bet
            self.stats.house_won += 1
            self.stats.dealer_blackjack_count += 1
        elif player_score > dealer_score:
            hand.resolution = "player_score"
            hand.outcome = "win"
            hand.payout = 2 * hand.bet
            hand.net_reward = hand.bet
            self.stats.cash += hand.payout
            self.stats.hands_won += 1
            self.stats.player_has_more += 1
        elif player_score < dealer_score:
            hand.resolution = "dealer_score"
            hand.outcome = "loss"
            hand.net_reward = -hand.bet
            self.stats.house_won += 1
            self.stats.dealer_has_more += 1
        else:
            hand.resolution = "push"
            hand.outcome = "draw"
            hand.payout = hand.bet
            hand.net_reward = 0
            self.stats.cash += hand.payout
            self.stats.draws += 1

        self.stats.hands_played += 1
        self.stats.most_cash = max(self.stats.most_cash, self.stats.cash)

    def _build_training_rewards(self, hands: list[Hand]) -> dict[str, int]:
        """Builds reward assignments for Q-table updates."""
        direct_rewards = {hand.hand_id: hand.net_reward for hand in hands}
        aggregated_rewards = dict(direct_rewards)

        for hand in hands:
            parent_id = hand.parent_hand_id
            while parent_id:
                aggregated_rewards[parent_id] = aggregated_rewards.get(parent_id, 0) + hand.net_reward
                parent_lookup = next(
                    (
                        candidate.parent_hand_id
                        for candidate in hands
                        if candidate.hand_id == parent_id
                    ),
                    None,
                )
                parent_id = parent_lookup

        return aggregated_rewards

    def play_round(self, agent: Any) -> dict[str, Any]:
        """Runs one round of blackjack for the supplied player.

        Args:
            agent: Any object that exposes `choose_bet(context)` and
                `choose_action(context)`.

        Returns:
            A round summary containing the resolved hand outcomes, stats, and
            training rewards.
        """
        if self.stats.cash < min(self.bet_options):
            if not self.auto_rebuy:
                raise RuntimeError("Player cannot afford the minimum bet.")
            self._reset_bankroll()

        self.round_number += 1
        self._current_hand_sequence = 0
        self.current_phase = "betting"
        self.current_agent_name = getattr(agent, "name", agent.__class__.__name__)
        self.current_player_hands = []
        self.current_dealer_hand = []
        self.current_bet = 0

        bet_context = self._build_bet_context()
        self.log_event(
            "prompt_shown",
            bet_context.prompt,
            available_bets=list(bet_context.available_bets),
            cash=bet_context.cash,
        )

        chosen_bet = agent.choose_bet(bet_context)
        if chosen_bet not in bet_context.available_bets:
            chosen_bet = bet_context.available_bets[0]

        self.current_bet = chosen_bet
        self.stats.cash -= chosen_bet
        self.log_event(
            "bet_placed",
            f"{self.current_agent_name} bet ${chosen_bet}.",
            bet=chosen_bet,
            remaining_cash=self.stats.cash,
        )

        hands = [Hand(hand_id=self._make_hand_id(), cards=[], bet=chosen_bet)]
        self.current_player_hands = hands
        self._deal_initial_cards(hands)
        self.log_event(
            "initial_deal",
            "Initial cards dealt.",
            player_hands=[self._serialize_hand(hand) for hand in hands],
            dealer_up_card=self.current_dealer_hand[0],
        )

        hand_index = 0
        while hand_index < len(hands):
            hand_index = self._play_hand(agent, hands, hand_index)

        if self._dealer_should_play(hands):
            self._play_dealer()

        dealer_score = calculate_score(self.current_dealer_hand)
        dealer_blackjack = is_blackjack(self.current_dealer_hand)

        for hand in hands:
            self._resolve_hand(hand, dealer_score, dealer_blackjack)

        self.stats.rounds_played += 1
        training_rewards = self._build_training_rewards(hands)
        self.current_phase = "round_complete"

        round_summary = {
            "round_number": self.round_number,
            "agent_name": self.current_agent_name,
            "bet": chosen_bet,
            "dealer_hand": {
                "cards": list(self.current_dealer_hand),
                "score": dealer_score,
                "blackjack": dealer_blackjack,
            },
            "player_hands": [self._serialize_hand(hand) for hand in hands],
            "training_rewards": training_rewards,
            "round_net_reward": sum(hand.net_reward for hand in hands),
            "stats": self.stats.to_dict(),
            "shoe": self.shoe.snapshot(),
        }

        self.last_round_summary = round_summary
        self.log_event(
            "round_complete",
            f"Round {self.round_number} finished with net reward {round_summary['round_net_reward']}.",
            round_summary=round_summary,
        )

        if hasattr(agent, "on_round_complete"):
            agent.on_round_complete(round_summary)

        return round_summary


def run_sample_session(rounds: int = 25, agent: Any | None = None) -> dict[str, Any]:
    """Runs a finite simulation and returns the resulting state snapshot."""
    if agent is None:
        raise ValueError("An agent is required to run a sample session.")

    game = BlackjackGame()
    for _ in range(rounds):
        game.play_round(agent)

    return game.get_state_snapshot()
