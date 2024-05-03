import random
import readchar

from pprint import pprint

def deal_card(shoe):
	# remove card from the shoe, don't return its value
	random_deck_index = random.randint(0, len(shoe)-1)
	random_card_index = random.randint(0, len(shoe[random_deck_index])-1)

	random_deck = shoe[random_deck_index]
	random_card = random_deck[random_card_index]
	random_deck.remove(random_card)
	
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

	# Handle the needed acces
	if ace_count > 0:
		if total+ace_count*11 <= 21:
			total = total + ace_count*11
		else:
			# You can't have two 11 count aces because then you will bust
			total = total + 11 + (ace_count-1)*1

	return total

def initialize_shoe(shoe_size):
	shoe = []
	for i in range(shoe_size):
		shoe.append(['A',2,3,4,5,6,7,8,9,10,'J','Q','K'])

	return shoe

def blackjack(shoe_size = 1):

	player_money = 1000
	hands_won = 0
	total_hands = 0
	shoe = initialize_shoe(shoe_size)
	
	print("Welcome to ML Blackjack!\n")

	while 1:

		# Game loop
		player_hand = []
		dealer_hand = []

		#game_option = input(f"Cash available ${player_money}\nChoose an option:\n\n(1) 50\n(2) 100\n(3) 250\n(4) 500\n(5) 1000\n(q) quit\n\n")
		print(f"Cash available ${player_money}\nChoose an option:\n\n(1) 50\n(2) 100\n(3) 250\n(4) 500\n(5) 1000\n(q) quit\n\n")
		game_option = readchar.readchar()
		if game_option == '1':
			hand_value = 50
		elif game_option == '2':
			hand_value = 100
		elif game_option == '3':
			hand_value = 250
		elif game_option == '4':
			hand_value = 500
		elif game_option == '5':
			hand_value = 1000
		elif game_option == 'p':
			pprint(shoe)
			continue
		elif game_option == 'q':
			# The terminating case
			win_pct = round(hands_won / total_hands * 100, 2)
			if player_money < 50:
				print(f"You ran out of money...\n")

			quit(f"Take care!\nStats:\nFinal Player's Money: ${player_money}\nTotal hands played: {total_hands}\nHands won: {hands_won}\nWin %: {win_pct}%")

		# Take money from player
		player_money -= hand_value

		# Hand loop
		for _ in range(2):
			player_hand.append(deal_card(shoe))
			#print(shoe);
			dealer_hand.append(deal_card(shoe))
			#print(shoe);

		game_over = False

		while not game_over:
			player_score = calculate_score(player_hand)
			dealer_score = calculate_score(dealer_hand)

			print(f"Cash available: ${player_money}")
			print(f"Your cards: {player_hand}, current score: {player_score}")
			print(f"Dealer's cards: {dealer_hand}, current_score: {dealer_score}")

			if player_score > 21 or dealer_score == 21:
				game_over = True
			else:
				#should_continue = input("Choose your option:\n\n(h) hit \n(p) stay \n(v) split\n\n")
				should_continue = print("Choose your option:\n\n(h) hit \n(p) stay \n(v) split\n\n")
				should_continue = readchar.readchar()
				if should_continue == 'h':
					player_hand.append(deal_card(shoe))
				elif should_continue == 'q':
					game_over = True
				else:
					game_over = True

			print("\n")


		# Dealer's play
		while dealer_score != 0 and dealer_score < 17:
			dealer_hand.append(deal_card(shoe))
			dealer_score = calculate_score(dealer_hand)

		print(f"Your final hand: {player_hand}, final score: {player_score}")
		print(f"Computer's final hand: {dealer_hand}, final score: {dealer_score}")

		if player_score > 21:
			print("You went over. You lose!")
		elif dealer_score > 21:
			hands_won += 1
			player_money = player_money + 2*hand_value
			print("Computer went over. You win!")
		elif player_score == dealer_score:
			player_money = player_money + hand_value
			print("It's a draw!")
		elif player_score == 0:
			hands_won += 1
			player_money = player_money + 2*hand_value
			print("Blackjack! You win!")
		elif dealer_score == 0:
			print("Computer got a Blackjack. You lose!")
		elif player_score > dealer_score:
			hands_won += 1
			player_money = player_money + 2*hand_value
			print("You have the bigger score.You win!")
		elif player_score < dealer_score:
			print("Computer had the bigger score. You lost!")
		else:
			exit("Generic Error!")

		total_hands += 1

print(blackjack(6))
