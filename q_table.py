import random

from collections import defaultdict

class QTable:
    def __init__(self):
        self.q_table = defaultdict(lambda: [0.0, 0.0])          # Initialize Q-values for each state
        self.ACTIONS = ['hit', 'stay', 'double-down', 'split']  # Define possible actions

    def get_q_values(self, state):
        return self.q_table[state]

    def update_q_value(self, state, action, reward, alpha=0.1):
        current_q_values = self.q_table[state]
        current_q_value = current_q_values[action]
        new_q_value = current_q_value + alpha * (reward - current_q_value)
        self.q_table[state][action] = new_q_value

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(self.ACTIONS)
        else:
            return max(iterable=self.q_table[state], 
                       key=lambda a: self.q_table[state][a])