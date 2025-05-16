import random
import copy,math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F

class SnortGame:
    ONGOING = "ongoing"
    R_WINS = "R_wins"
    B_WINS = "B_wins"

    def __init__(self, size=5, debug=True):
        self.size = size
        self.board = np.full((size, size), ' ', dtype=str)
        self.current_player = 'R'
        self.blocked_positions = set()
        self.debug = debug
        self.game_status = SnortGame.ONGOING

        self._legal_moves_cache = None
        self._has_legal_moves_cache = None  

        self.initialize_blocked_positions()

    def initialize_blocked_positions(self):
        while len(self.blocked_positions) < 3:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (x, y) not in self.blocked_positions:
                self.blocked_positions.add((x, y))
                self.board[x, y] = 'X'

    def make_move(self, move):
        x, y = move
        if move in self.legal_moves():
            self.board[x, y] = self.current_player
            self.current_player = 'B' if self.current_player == 'R' else 'R'
            self._legal_moves_cache = None
            self._has_legal_moves_cache = None
            if not self.has_legal_moves():
                if self.debug:
                    print(f"No more legal moves for {self.current_player}. Checking board...")
                    print(f"Remaining legal moves: {self.legal_moves()}")
                self.game_status = SnortGame.R_WINS if self.current_player == 'B' else SnortGame.B_WINS
            
            return True
        return False

    def unmake_move(self, move):
        x, y = move
        if self.board[x, y] in ['R', 'B']:
            self.current_player = self.board[x, y] 
            self.board[x, y] = ' '
            self.game_status = SnortGame.ONGOING
            self._legal_moves_cache = None
            self._has_legal_moves_cache = None

    def is_valid_move(self, x, y):
        if (x, y) in self.blocked_positions or self.board[x, y] != ' ':
            return False
        opponent = 'B' if self.current_player == 'R' else 'R'
        for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == opponent:
                return False
        return True

    def legal_moves(self):
        if self._legal_moves_cache is None:
            self._legal_moves_cache = [
                (x, y) for x in range(self.size) for y in range(self.size) if self.is_valid_move(x, y)
            ]
        return self._legal_moves_cache

    def has_legal_moves(self):
        if self._has_legal_moves_cache is None:
            self._has_legal_moves_cache = len(self.legal_moves()) > 0 
        return self._has_legal_moves_cache

    def encode(self):
        R_mask = (self.board == 'R').astype(int)
        B_mask = (self.board == 'B').astype(int)
        blocked_mask = (self.board == 'X').astype(int)
        player_turn = np.full((self.size, self.size), 1 if self.current_player == 'R' else 0, dtype=int)
        return np.stack([R_mask, B_mask, blocked_mask, player_turn], axis=0)

    def decode(self, encoded_state):
        self.board = np.full((self.size, self.size), ' ', dtype=str)
        self.blocked_positions.clear()

        R_mask, B_mask, blocked_mask, player_turn = encoded_state
        self.board[R_mask == 1] = 'R'
        self.board[B_mask == 1] = 'B'
        self.board[blocked_mask == 1] = 'X'
        self.blocked_positions = set(zip(*np.where(blocked_mask == 1)))
        self.current_player = 'R' if player_turn[0, 0] == 1 else 'B'
        self._legal_moves_cache = None
        self._has_legal_moves_cache = None

    def status(self):
        return self.game_status

    def display_board(self):
        print("  " + " ".join(str(i) for i in range(self.size)))
        for idx, row in enumerate(self.board):
            print(f"{idx} " + " ".join(row))
        print()

    def play(self, player):
        print("Welcome to Snort!")
        self.display_board()
        while self.status() == SnortGame.ONGOING:
            print(f"Current player: {'Red' if self.current_player == 'R' else 'Blue'}")
            move_selection = player.select_move(self)
            if isinstance(move_selection, tuple) and len(move_selection) == 2:
                move = move_selection
            else:
                print(f"Invalid move received: {move_selection}. Skipping turn.")
                continue
            if self.make_move(move):
                self.display_board()
            else:
                print("Invalid move attempted.")
        winner = 'Red' if self.game_status == SnortGame.R_WINS else 'Blue'
        print(f"Game over! {winner} wins!")

    def play(self, player1, player2):
        print("Welcome to Snort!")
        self.display_board()
        
        while self.status() == SnortGame.ONGOING:
            print(f"Current player: {'Red' if self.current_player == 'R' else 'Blue'}")

            # Determine  player
            current_player = player1 if self.current_player == 'R' else player2

            # legal moves for the current player
            legal_moves = self.legal_moves()
            print(f"Legal moves for {self.current_player}: {legal_moves}" if legal_moves else "No legal moves left!")

            # Player selects a move
            move_selection = current_player.select_move(self)
            
            if isinstance(move_selection, tuple) and len(move_selection) == 2:
                move = move_selection
            else:
                print(f"Invalid move received: {move_selection}. Skipping turn.")
                continue

            if self.make_move(move):
                self.display_board()
            else:
                print("Invalid move attempted.")
        winner = 'Red' if self.game_status == SnortGame.R_WINS else 'Blue'
        print(f"Game over! {winner} wins!")


class MCTSNode:
    def __init__(self, game, parent=None, prior_prob=1.0):
        self.game = copy.deepcopy(game)
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior_prob = prior_prob  

    def is_fully_expanded(self):
        return len(self.children) == len(self.game.legal_moves())

    def best_child(self, c_param=1.4):
        return max(self.children.items(), key=lambda item: self.uct_score(item[1], c_param))[0]

    def uct_score(self, child, c_param):
        if child.visit_count == 0:
            return float("inf")  

        q_value = child.total_value / child.visit_count
        exploration = c_param * math.sqrt(math.log(self.visit_count) / (1 + child.visit_count))
        return q_value + exploration

    def expand(self):
        for move in self.game.legal_moves():
            if move not in self.children:
                new_game = copy.deepcopy(self.game)
                new_game.make_move(move)
                self.children[move] = MCTSNode(new_game, self)

    def simulate(self):
        sim_game = copy.deepcopy(self.game)
        
        while sim_game.status() == SnortGame.ONGOING:
            legal_moves = sim_game.legal_moves()
            if not legal_moves:
                break  # no moves left

            move = random.choice(legal_moves)  #  random move
            sim_game.make_move(move)

        return 1 if sim_game.status() == SnortGame.R_WINS else -1

    def backpropagate(self, result):
        self.visit_count += 1
        self.total_value += result
        if self.parent:
            self.parent.backpropagate(-result) 

class MCTSPlayer:
    def __init__(self, simulations=25):
        self.simulations = simulations

    def select_move(self, game):
        root = MCTSNode(game)

        for _ in range(self.simulations):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.children[node.best_child()]
            if node.game.status() == SnortGame.ONGOING:
                node.expand()
            result = node.simulate()
            node.backpropagate(result)

        return root.best_child()
class PUCTNode:
    def __init__(self, game, game_network, parent=None, prior_prob=1.0):
        self.game = copy.deepcopy(game)
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior_prob = prior_prob
        self.game_network = game_network 

    def select_action(self, c_puct=1.0):
        return max(
            self.children.items(),
            key=lambda item: (item[1].total_value / (1 + item[1].visit_count)) +
                             c_puct * item[1].prior_prob *
                             math.sqrt(self.visit_count) / (1 + item[1].visit_count)
        )[0]

    def expand(self, policy_probs):
        for move, prob in policy_probs:
            if move not in self.children:
                new_game = copy.deepcopy(self.game)
                new_game.make_move(move)
                self.children[move] = PUCTNode(new_game, self.game_network, self, prob) 

    def simulate(self):
        state = torch.tensor(self.game.encode(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value, _ = self.game_network(state)  
        return value.item()

    def backpropagate(self, result):
        self.visit_count += 1
        self.total_value += result
        if self.parent:
            self.parent.backpropagate(-result)

class PUCTPlayer:
    def __init__(self, game_network, simulations=100, c_puct=1.0):
        self.game_network = game_network
        self.simulations = simulations
        self.c_puct = c_puct

    def select_move(self, game):
        root = PUCTNode(game, self.game_network)  

        state = torch.tensor(game.encode(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, policy = self.game_network(state)
        policy = policy.squeeze().numpy()

        legal_moves = game.legal_moves()
        if not legal_moves:
            print("No legal moves available!")
            return None  

        policy_probs = [(m, policy[m[0] * game.size + m[1]]) for m in legal_moves]
        root.expand(policy_probs)

        for _ in range(self.simulations):
            node = root
            while node.children:
                node = node.children[node.select_action(self.c_puct)]

            if node.game.status() == SnortGame.ONGOING:
                state = torch.tensor(node.game.encode(), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, policy = self.game_network(state)
                policy = policy.squeeze().numpy()

                legal_moves = node.game.legal_moves()
                if legal_moves:
                    policy_probs = [(m, policy[m[0] * game.size + m[1]]) for m in legal_moves]
                    node.expand(policy_probs)

            result = node.simulate()
            node.backpropagate(result)

        return root.select_action(c_puct=0)

class CNNGameNetwork(nn.Module):
    def __init__(self, board_size):
        super(CNNGameNetwork, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.value_head = nn.Linear(256, 1)
        self.policy_head = nn.Linear(128 * board_size * board_size, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        value = F.relu(self.fc1(x))
        value = torch.tanh(self.value_head(value))
        policy = F.softmax(self.policy_head(x), dim=-1)
        return value, policy
    def save_model(self, filepath='game_model.pth'):
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='game_model_v1.pth'):
        self.load_state_dict(torch.load(filepath))
        self.eval() 
        print(f"Model loaded from {filepath}")

def generate_self_play_games(num_games=10000):
    dataset = []

    for game_num in range(num_games):
        print(f"Generating game {game_num + 1}/{num_games}...")
        game = SnortGame(size=5, debug=False)
        mcts_player = MCTSPlayer(simulations=100) 

        states, values, policies = [], [], []

        while game.status() == SnortGame.ONGOING:
            state = game.encode()
            move = mcts_player.select_move(game)
            print(f"Game {game_num + 1}: Move selected {move}")

            root = MCTSNode(game)
            root.expand()
            visit_counts = np.zeros(25) 
            total_visits = sum(child.visit_count for child in root.children.values())

            if total_visits > 0:
                for m, child in root.children.items():
                    visit_counts[m[0] * 5 + m[1]] = child.visit_count / total_visits  

            states.append(state)
            policies.append(visit_counts)
            game.make_move(move)

        outcome = 1 if game.status() == SnortGame.R_WINS else -1
        print(f"Game {game_num + 1} finished. Winner: {'Red' if outcome == 1 else 'Blue'}")

        values = [outcome] * len(states)
        for i in range(len(states)):
            dataset.append((states[i], values[i], policies[i]))

    print("Self-play game generation complete.")
    return dataset


def train_neural_network(dataset, epochs=5, batch_size=32):
    game_network = CNNGameNetwork(board_size=5)
    game_network.train()
    optimizer = optim.Adam(game_network.parameters(), lr=0.001)
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    dataset_size = len(dataset)
    for epoch in range(epochs):  
        total_loss = 0
        random.shuffle(dataset) 

        for i in range(0, dataset_size, batch_size):
            batch = dataset[i:i + batch_size]
            state_batch = torch.tensor([d[0] for d in batch], dtype=torch.float32)
            outcome_batch = torch.tensor([d[1] for d in batch], dtype=torch.float32).unsqueeze(1)
            policy_batch = torch.tensor([d[2] for d in batch], dtype=torch.float32)

            optimizer.zero_grad()
            predicted_value, predicted_policy = game_network(state_batch)

            value_loss = value_loss_fn(predicted_value, outcome_batch)
            policy_loss = torch.sum(-policy_batch * torch.log(predicted_policy + 1e-7)) / batch_size

            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    game_network.save_model()


def generate_puct_self_play_games(game_network, num_games=1000, simulations=100, alpha=0.3, epsilon=0.25):
    dataset = []
    for game_num in range(num_games):
        print(f"Generating game {game_num + 1}/{num_games}...")
        game = SnortGame(size=5, debug=False)
        puct_player = PUCTPlayer(game_network, simulations=simulations)
        states, values, policies = [], [], []
        while game.status() == SnortGame.ONGOING:
            state = game.encode()
            move = puct_player.select_move(game)
            root = PUCTNode(game, game_network)
            root.expand([(m, 1/len(game.legal_moves())) for m in game.legal_moves()])
            
            total_visits = sum(child.visit_count for child in root.children.values())
            policy_vector = np.zeros(25) 

            if total_visits > 0:
                for move, child in root.children.items():
                    policy_vector[move[0] * 5 + move[1]] = child.visit_count / total_visits
            else:
                # If no visits, assign equal probability to legal moves
                legal_moves = game.legal_moves()
                if legal_moves:
                    for move in legal_moves:
                        policy_vector[move[0] * 5 + move[1]] = 1 / len(legal_moves)

            # Add Dirichlet noise for exploration
            noise = np.random.dirichlet([alpha] * len(game.legal_moves()))
            for i, move in enumerate(game.legal_moves()):
                policy_vector[move[0] * 5 + move[1]] = (1 - epsilon) * policy_vector[move[0] * 5 + move[1]] + epsilon * noise[i]

            # Store state, policy, and value
            states.append(state)
            policies.append(policy_vector)
            game.make_move(move)

        # Assign outcome (1 if Red wins, -1 if Blue wins)
        outcome = 1 if game.status() == SnortGame.R_WINS else -1
        values = [outcome] * len(states)
        for i in range(len(states)):
            dataset.append((states[i], values[i], policies[i]))

    print("Self-play game generation complete.")
    return dataset

def update_elo(rating_A, rating_B, result, k=32):
    expected_A = 1 / (1 + 10**((rating_B - rating_A) / 400))
    expected_B = 1 - expected_A

    new_rating_A = rating_A + k * (result - expected_A)
    new_rating_B = rating_B + k * (1 - result - expected_B)

    return new_rating_A, new_rating_B


def evaluate_agents(new_agent, old_agent, games=50):
    rating_new, rating_old = 1500, 1500 
    new_wins, old_wins, draws = 0, 0, 0

    for game_num in range(1, games + 1):
        print(f"\nGame {game_num}/{games}")
        game = SnortGame(size=5, debug=False)
        current_player = new_agent if random.random() > 0.5 else old_agent

        while game.status() == SnortGame.ONGOING:
            move = current_player.select_move(game)
            game.make_move(move)
            print(f"\nMove: {move} by {'New Agent' if current_player == new_agent else 'Old Agent'}")
            game.display_board()

            current_player = new_agent if current_player == old_agent else old_agent

        if game.status() == SnortGame.R_WINS:
            new_wins += 1
            result = 1  
            print("\nNew Agent Wins!")
        elif game.status() == SnortGame.B_WINS:
            old_wins += 1
            result = 0  
            print("\nOld Agent Wins!")
        else:
            draws += 1
            result = 0.5  # Draw
            print("\nIt's a Draw!")

        rating_new, rating_old = update_elo(rating_new, rating_old, result)

        print("\nFinal Board State:")
        game.display_board()

    print("\n========== Tournament Results ==========")
    print(f"New Agent Wins: {new_wins}, Old Agent Wins: {old_wins}, Draws: {draws}")
    print(f"Final ELO Ratings - New Agent: {rating_new}, Old Agent: {rating_old}")


if __name__ == "__main__":
    print("Generating self-play games...")
    dataset = generate_self_play_games(num_games=50)

    print("Training the neural network...")
    train_neural_network(dataset, epochs=10)

    print("Loading the trained model...")
    game_network = CNNGameNetwork(board_size=5)
    game_network.load_model()

    print("Starting a game with the trained PUCT player...")
    game = SnortGame(size=5, debug=True)
    puct_player = PUCTPlayer(game_network, simulations=500)
    game.play(puct_player)
    game_network_v1 = CNNGameNetwork(board_size=5)
    game_network_v1.load_model("game_model.pth") 

    game_network_v2 = CNNGameNetwork(board_size=5)
    game_network_v2.load_model("game_model_v1.pth") 

    agent_v1 = PUCTPlayer(game_network_v1, simulations=50)
    agent_v2 = PUCTPlayer(game_network_v2, simulations=50)

    evaluate_agents(agent_v2, agent_v1, games=50) 

class HumanPlayer:
    def select_move(self, game):
        while True:
            try:
                move = input("Enter your move (row col): ").split()
                move = (int(move[0]), int(move[1]))
                if move in game.legal_moves():
                    return move
                print("Invalid move Try again.")
            except (ValueError, IndexError):
                print("Invalid input  enter two numbers separated by space.")

def play_game():
    size = 5  # Board size
    game = SnortGame(size=size, debug=True)

    print("Choose Player 1 (Red):")
    print("1: Human")
    print("2: MCTS AI")
    print("3: PUCT AI")
    choice1 = input("Enter 1, 2, or 3: ")
    
    if choice1 == "1":
        player1 = HumanPlayer()
    elif choice1 == "2":
        player1 = MCTSPlayer(simulations=100)
    elif choice1 == "3":
        game_network = CNNGameNetwork(size)
        game_network.load_model("game_model_v1.pth")
        player1 = PUCTPlayer(game_network, simulations=100)
    else:
        print("Invalid choice! Defaulting to Human.")
        player1 = HumanPlayer()

    print("Choose Player 2 (Blue):")
    print("1: Human")
    print("2: MCTS AI")
    print("3: PUCT AI")
    choice2 = input("Enter 1, 2, or 3: ")

    if choice2 == "1":
        player2 = HumanPlayer()
    elif choice2 == "2":
        player2 = MCTSPlayer(simulations=100)
    elif choice2 == "3":
        game_network = CNNGameNetwork(size)
        game_network.load_model("game_model_v1.pth")
        player2 = PUCTPlayer(game_network, simulations=100)
    else:
        print("Invalid choice! Defaulting to Human.")
        player2 = HumanPlayer()

    game.play(player1, player2)

play_game()

import torch
import random

def main():
    model_path = "game_model_v1.pth"
    game_network = CNNGameNetwork(board_size=5)

    try:
        game_network.load_model(model_path)  
        print(f"Loaded existing model: {model_path}")
    except:
        print(f" No existing model found. Starting from scratch.")

    total_iterations = 5   
    games_per_iteration = 10 
    simulations_per_move = 10 

    for iteration in range(1, total_iterations + 1):
        print(f"\n Training Iteration {iteration}/{total_iterations} ")
        
        dataset = generate_puct_self_play_games(game_network, num_games=games_per_iteration, simulations=simulations_per_move)
        
        train_neural_network(dataset, epochs=5)
        
        model_filename = f"game_model_iter{iteration}.pth"
        game_network.save_model(model_filename)
        print(f"Saved model: {model_filename}")
        
        game_network.save_model(model_path)
        if iteration > 1:
            print("\nEvaluating AI Strength ")
            old_model = CNNGameNetwork(board_size=5)
            old_model.load_model(f"game_model_iter{iteration - 1}.pth")
            
            new_agent = PUCTPlayer(game_network, simulations=simulations_per_move)
            old_agent = PUCTPlayer(old_model, simulations=simulations_per_move)
            
            evaluate_agents(new_agent, old_agent, games=50)

    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
