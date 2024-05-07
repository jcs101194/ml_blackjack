import blackjack as b

from pprint import pprint

class Player:

	def __init__(self, betting_algorithm, hitting_algorithm, confidence):
		self.betting_algorithm = betting_algorithm
		self.hitting_algorithm = hitting_algorithm
		self.confidence_threshold = confidence

	def set_shoe(self, shoe):
		self.shoe = shoe

	def set_hand(self, hand):
		self.hand = hand

	def set_score(self, hand_score):
		self.score = hand_score

	def calculate_shoe_size(self):
		size = 0
		for deck in self.shoe:
			size += len(deck)

		return size

	"""
		@param highest_card The biggest possible card acceptable in current hand

		@return The number of possible, good cards
	"""
	
	def get_good_card_count(self, highest_card):
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
		if 'Cash available' in prompt:
			return self.betting_algorithm()
		elif 'hit' in prompt:
			return self.hitting_algorithm(self)

def betting_algorithm():
	return '1'

def hitting_algorithm(agent):
	# If card count is simply too low
	#if agent.score <= 14:
	#	return 'h'
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

	player_1 = Player(betting_algorithm, hitting_algorithm, .60)
	b.blackjack(6, player_1)


if __name__ == "__main__":
	run_experiment()
