import blackjack as b
from player import Player, betting_algorithm, hitting_algorithm

def main():
    """Runs a sample blackjack simulation with the baseline agent."""

    player_1 = Player(betting_algorithm=betting_algorithm,
                   hitting_algorithm=hitting_algorithm,
                   confidence=.60)
    b.game_loop(shoe_size=6, agent=player_1)

if __name__ == "__main__":
    main()