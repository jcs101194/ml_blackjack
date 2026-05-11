import blackjack as b

from pprint import pprint

class Player:
	"""Baseline blackjack agent driven by simple betting and hit logic."""

	def __init__(self, betting_algorithm, hitting_algorithm, confidence):
		"""Initializes the player with strategy callbacks.

		Args:
			betting_algorithm: A callable that returns the selected betting option.
			hitting_algorithm: A callable that decides whether to hit or stay.
			confidence: The minimum safe-draw probability required to hit.
		"""
		self.betting_algorithm = betting_algorithm
		self.hitting_algorithm = hitting_algorithm
		self.confidence_threshold = confidence

	def set_shoe(self, shoe):
		"""Stores the current shoe for later decision making.

		Args:
			shoe: A list of decks representing the remaining cards in play.
		"""
		self.shoe = shoe

	def set_hand(self, hand):
		"""Stores the player's current hand.

		Args:
			hand: A list of cards currently held by the player.
		"""
		self.hand = hand

	def set_score(self, hand_score):
		"""Stores the player's current hand score.

		Args:
			hand_score: The numeric score of the player's current hand.
		"""
		self.score = hand_score

	def calculate_shoe_size(self):
		"""Counts how many cards remain in the shoe.

		Returns:
			The total number of cards left across all decks.
		"""
		size = 0
		for deck in self.shoe:
			size += len(deck)

		return size

	def get_good_card_count(self, highest_card):
		"""Counts the cards that can be drawn without immediately busting.

		Args:
			highest_card: The largest card value the player can safely draw.

		Returns:
			The number of remaining cards in the shoe considered safe draws.
		"""
		possible_card_count = 0
		for deck in self.shoe:
			for card in deck:
				if card == 'A':
					possible_card_count += 1
				elif card == 'J' or card == 'Q' or card == 'K':
					if 10 <= highest_card:
						possible_card_count += 1
				elif card <= highest_card:
					possible_card_count += 1

		return possible_card_count

	def decide(self, prompt):
		"""Routes a game prompt to the appropriate strategy callback.

		Args:
			prompt: The prompt string emitted by the blackjack environment.

		Returns:
			The action selected by the configured strategy callback.
		"""
		if 'Cash available' in prompt:
			return self.betting_algorithm()
		elif 'hit' in prompt:
			return self.hitting_algorithm(self)

def betting_algorithm():
	"""Selects the baseline betting action.

	Returns:
		The menu option for the minimum bet.
	"""
	return '1'

def hitting_algorithm(agent):
	"""Chooses whether to hit based on safe-draw probability.

	Args:
		agent: The player instance requesting a decision.

	Returns:
		'h' if the probability of a safe draw meets the confidence threshold.
		Otherwise, 'p' to stay.
	"""
	score_limit = 21 - agent.score
	shoe_size = agent.calculate_shoe_size()

	card_count = agent.get_good_card_count(score_limit)
	good_card_prob = card_count / shoe_size
	
	pprint(agent.shoe)
	print(score_limit)
	print(good_card_prob)

	if good_card_prob >= agent.confidence_threshold:
		return 'h'
	else:
		return 'p'


def run_experiment():
	"""Runs a sample blackjack simulation with the baseline agent."""

	player_1 = Player(betting_algorithm, hitting_algorithm, .60)
	b.blackjack(6, player_1)


if __name__ == "__main__":
	run_experiment()
