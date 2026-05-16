#!/usr/bin/env bash
python server.py --host 127.0.0.1 --mode q --model-path checkpoints/blackjack_q_table.json --bet-model-path checkpoints/blackjack_bet_q_table.json
