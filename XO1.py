import numpy as np
import random
from collections import defaultdict
import pickle

# Initialize game variables and state
def initialize_board():
    return np.full((3, 3), ' ')

def get_state(board):
    return tuple(board.flatten())

def available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == ' ']

def make_move(board, row, col, player):
    board[row, col] = player
    return board

def check_winner(board, row, col, player):
    # Check row, column, and diagonals
    if all([board[row, c] == player for c in range(3)]) or \
       all([board[r, col] == player for r in range(3)]) or \
       (row == col and all([board[i, i] == player for i in range(3)])) or \
       (row + col == 2 and all([board[i, 2 - i] == player for i in range(3)])):
        return True
    return False

def is_full(board):
    return ' ' not in board


def choose_action(state, q_table, available_moves, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(available_moves)
    q_values = [q_table[(state, move)] for move in available_moves]
    max_q = max(q_values)
    return random.choice([move for move, q in zip(available_moves, q_values) if q == max_q])

def update_q_table(q_table, state, action, reward, next_state, next_available_moves, alpha=0.1, gamma=0.9):
    old_q = q_table[(state, action)]
    next_max_q = max([q_table[(next_state, move)] for move in next_available_moves], default=0)
    q_table[(state, action)] = old_q + alpha * (reward + gamma * next_max_q - old_q)


# Add Q-table save and load functions
def save_q_table(q_table, filename='q_table.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(q_table, file)
    print("Q-table saved to", filename)

def load_q_table(filename='q_table.pkl'):
    try:
        with open(filename, 'rb') as file:
            q_table = pickle.load(file)
        print("Q-table loaded from", filename)
        return q_table
    except FileNotFoundError:
        print("Q-table file not found, starting with an empty Q-table.")
        return defaultdict(float)

# Training function
def train(q_table, episodes=15000):
    print('Training start...')
    for _ in range(episodes):
        board = initialize_board()
        state = get_state(board)
        done = False
        while not done:
            moves = available_moves(board)
            action = choose_action(state, q_table, moves)
            board = make_move(board, action[0], action[1], 'X')
            next_state = get_state(board)

            if check_winner(board, action[0], action[1], 'X'):
                update_q_table(q_table, state, action, 1, next_state, [])
                done = True
            elif is_full(board):
                update_q_table(q_table, state, action, 0.5, next_state, [])
                done = True
            else:
                # Opponent (random) move
                opp_action = random.choice(available_moves(board))
                board = make_move(board, opp_action[0], opp_action[1], 'O')
                if check_winner(board, opp_action[0], opp_action[1], 'O'):
                    update_q_table(q_table, state, action, -1, next_state, [])
                    done = True
                elif is_full(board):
                    update_q_table(q_table, state, action, 0.5, next_state, [])
                    done = True
                else:
                    update_q_table(q_table, state, action, 0, next_state, available_moves(board))
            state = next_state
    print('Training Complete.')

# Play function
def play(q_table):
    print('Play Phase: ')
    board = initialize_board()
    state = get_state(board)
    while True:
        moves = available_moves(board)
        action = choose_action(state, q_table, moves, epsilon=0)  # Choose best action
        board = make_move(board, action[0], action[1], 'X')
        next_state = get_state(board)
        print_board(board)
        if check_winner(board, action[0], action[1], 'X'):
            print("Agent wins!")
            break
        elif is_full(board):
            print("It's a tie!")
            break

        row, col = map(int, input("Enter your move (row col, 0-2): ").split())
        board = make_move(board, row, col, 'O')
        print_board(board)
        if check_winner(board, row, col, 'O'):
            print("You win!")
            break
        elif is_full(board):
            print("It's a tie!")
            break
        state = get_state(board)

# Print the board
def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('---------')

q_table = load_q_table()  # Load existing Q-table if available
train(q_table)            # Train the agent
save_q_table(q_table)     # Save the trained Q-table
play(q_table)             # Play against the agent
