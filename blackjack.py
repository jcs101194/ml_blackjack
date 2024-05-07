import random
import readchar

from pprint import pprint

SHOE_SIZE = 6

def deal_card(shoe):

	# Remove card from the shoe, don't return its value
	random_deck_index = random.randint(0, len(shoe)-1)
	# Make sure the selected deck has cards in it
	while len(shoe[random_deck_index]) == 0:
		random_deck_index = random.randint(0, len(shoe)-1)

	random_card_index = random.randint(0, len(shoe[random_deck_index])-1)

	random_deck = shoe[random_deck_index]
	random_card = random_deck[random_card_index]
	random_deck.remove(random_card)

	# Null case, renew shoe if empty
	if is_shoe_empty(shoe):
		print("Renewing shoe...\n")
		initialize_shoe(SHOE_SIZE, shoe)

	return random_card

	
def calculate_score(hand):
	total = 0
	ace_count = 0
	for card in hand:
		if card == 'A':
			# Ace cards will be calculated on a what-makes-more-sense basis
			# So count the aces in the mean time
			ace_count += 1
		elif card == 'J' or card == 'Q' or card == 'K':
			total += 10
		else:
			total += card

	# Handle the needed aces
	if ace_count > 0:
		if total+ace_count*11 <= 21:
			total = total + ace_count*11
		else:
			# You can't have two 11 count aces because then you will bust
			total = total + 11 + (ace_count-1)*1

	return total

def is_shoe_empty(shoe):
	for deck in shoe:
		if deck:
			return False

	return True

def initialize_shoe(shoe_size, shoe):

	# Remove empty decks inside the shoe
	if shoe:
		while not any(shoe):
			if not shoe:
				break
			#pprint(shoe)
			shoe.remove([])	

	for i in range(shoe_size):
		shoe.append(['A',2,3,4,5,6,7,8,9,10,'J','Q','K'])

	return shoe

def blackjack(shoe_size = 6, agent = None):

	player_money = 10000
	most_money = player_money
	hands_won = 0
	player_busts = 0
	player_has_more = 0
	player_blackjack_count = 0
	house_won = 0
	dealer_busts = 0
	dealer_has_more = 0
	dealer_blackjack_count = 0
	draws = 0
	total_hands = 0
	shoe = []
	SHOE_SIZE = shoe_size
	initialize_shoe(SHOE_SIZE, shoe)
	agent.set_shoe(shoe)
	
	print("Welcome to ML Blackjack!\n")

	while 1:

		# Game loop
		player_hand = []
		dealer_hand = []
		game_over = False

		prompt_1 = f"Cash available ${player_money}\nChoose an option:\n\n(1) 50\n(2) 100\n(3) 250\n(4) 500\n(5) 1000\n(q) quit\n\n"
		print(prompt_1)
		if agent == None: 
			game_option = readchar.readchar()
		else:
			game_option = agent.decide(prompt_1)
			print(f"{game_option}\n")

		if player_money < 50:
			# The terminating case
			game_over = True
			print(f"You ran out of money...\n")
		
		if game_option == '1': 
			hand_value = 50
		elif game_option == '2': hand_value = 100
		elif game_option == '3': hand_value = 250
		elif game_option == '4': hand_value = 500
		elif game_option == '5': hand_value = 1000
		elif game_option == 'p':
			pprint(shoe)
			continue
		elif game_option == 'q':
			# The terminating case
			game_over = True

		if game_over:
			# End the program
			win_pct = round(hands_won / total_hands * 100, 2)
			quit(f"Take care!\n"+
				f"Stats:\n"+
			   	f"Final Player's money: ${player_money}\n"+
			   	f"Most money carried: ${most_money}\n"+
				f"Total hands played: {total_hands}\n"+
				f"Hands won: {hands_won}\n"+
				f"Player Busts: {player_busts}\n"+
				f"Player wins by score: {player_has_more}\n"+
				f"Player wins by blackjack: {player_blackjack_count}\n"+
				f"Draws: {draws}\n"+
				f"House wins: {house_won}\n"+
				f"Dealer Busts: {dealer_busts}\n"+
				f"Dealer wins by score: {dealer_has_more}\n"+
				f"Dealer wins by blackjack: {dealer_blackjack_count}\n"+
				f"Win Rate: {win_pct}%")

		# Take money from player
		player_money -= hand_value

		# Hand loop
		for _ in range(2):
			player_hand.append(deal_card(shoe))
			dealer_hand.append(deal_card(shoe))


		game_over = False
		while not game_over:
			player_score = calculate_score(player_hand)
			dealer_score = calculate_score(dealer_hand)

			if agent is not None:
				agent.set_hand(player_hand)
				agent.set_score(player_score)

			print(f"Cash available: ${player_money}")
			print(f"Your cards: {player_hand}, current score: {player_score}")
			print(f"Dealer's cards: {dealer_hand}, current_score: {dealer_score}")

			if player_score > 21 or dealer_score == 21:
				game_over = True
			else:
				prompt_2 = "Choose your option:\n\n(h) hit \n(p) stay \n(v) split\n\n"
				print(prompt_2)

				if agent == None: 
					should_continue = readchar.readchar()
				else:
					should_continue = agent.decide(prompt_2)
					print(f"{should_continue}")

				if should_continue == 'h':
					player_hand.append(deal_card(shoe))
				else:
					game_over = True

			print("\n")


		# Dealer's algorithm
		while dealer_score != 0 and dealer_score < 17:
			dealer_hand.append(deal_card(shoe))
			dealer_score = calculate_score(dealer_hand)

		print(f"Your final hand: {player_hand}, final score: {player_score}")
		print(f"Computer's final hand: {dealer_hand}, final score: {dealer_score}")

		if player_score > 21:
			player_busts += 1
			house_won += 1
			print("You went over. You lose!")
		elif dealer_score > 21:
			hands_won += 1
			dealer_busts += 1
			player_money = player_money + 2*hand_value
			print("Dealer went over. You win!")
		elif player_score == dealer_score:
			player_money += hand_value
			draws += 1
			print("It's a draw!")
		elif player_score == 21:
			hands_won += 1
			player_blackjack_count += 1
			player_money = player_money + 2*hand_value
			print("Blackjack! You win!")
		elif dealer_score == 21:
			dealer_blackjack_count += 1
			house_won += 1
			print("Dealer got a Blackjack. You lose!")
		elif player_score > dealer_score:
			hands_won += 1
			player_has_more += 1
			player_money = player_money + 2*hand_value
			print("You have the bigger score. You win!")
		elif player_score < dealer_score:
			dealer_has_more += 1
			house_won += 1
			print("Dealer had the bigger score. You lost!")
		else:
			exit("Generic Error!")

		# Calculate some aggregates
		total_hands += 1
		if player_money > most_money:
			most_money = player_money

if __name__ == "__main__":
	blackjack(6)
