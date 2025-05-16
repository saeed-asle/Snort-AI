# Snort AI: Self-Play Reinforcement Learning with PUCT and CNN

This project implements a self-play reinforcement learning framework for the combinatorial game **Snort**, using a **Convolutional Neural Network (CNN)** with **PUCT (Predictor + Upper Confidence bounds for Trees)**-based Monte Carlo Tree Search (MCTS).

The project allows for:

* Self-play data generation using MCTS/PUCT
* Training a neural network to evaluate board states
* Iterative self-improvement
* Human vs AI, AI vs AI gameplay
* Agent strength evaluation using ELO ratings

---

## Functions Structure

```
.
├── snort_game.py          # Core game logic for Snort
├── players.py             # HumanPlayer, MCTSPlayer, PUCTPlayer classes
├── network.py             # CNNGameNetwork for board evaluation
├── train.py               # Self-play and training loop
├── evaluate.py            # Agent comparison using ELO
├── play.py                # Human or AI gameplay
├── utils.py               # ELO update, dataset generation, etc.
├── game_model.pth         # (Generated) Trained model file
├── game_model_v1.pth      # (Generated) Later version of model
└── main.py                # Unified entry point (optional)
```

---

## Requirements

Install the required Python packages:

```bash
pip install torch numpy
```

---

## How to Use

Since the code is modular, you can **run different parts** by commenting/uncommenting blocks.

### Train the Agent

Runs self-play games, trains a CNN on the generated data, and saves models over several iterations.

```python
# Uncomment and run this block:

if __name__ == "__main__":
    main()  # Found at the bottom of the file
```

This will:

* Run 5 training iterations
* In each, generate 10 self-play games
* Train the CNN on collected data
* Evaluate current vs previous models using ELO

---

### Evaluate Agents (ELO Comparison)

Compare a new trained agent against an old one using a tournament of 50 games.

```python
evaluate_agents(agent_v2, agent_v1, games=50)
```

Each agent plays both Red and Blue alternately, and ELO is updated per game.

---

### Play the Game

Human vs AI, AI vs AI, or Human vs Human.

```python
play_game()
```

Choose player types:

* `1`: Human
* `2`: MCTS AI
* `3`: PUCT AI (requires trained model)

The board and move history are printed each turn. Input your move as:

```
Enter your move (row col): 2 3
```

---

## Training Details

* **Network**: Convolutional neural network (CNN) to evaluate board states
* **Training Data**: Generated through self-play using PUCT
* **Simulation**: Configurable number of MCTS simulations per move (default: 10–500)
* **Loss Function**: Supervised learning with MSE for value and cross-entropy for policy
* **Model Format**: Saved as `.pth` (PyTorch)

---

## ELO Rating System

A simple Elo rating system is used to track performance between model versions.

```python
def update_elo(r1, r2, result, k=32):
    ...
```

Where `result = 1` for win, `0` for loss, and `0.5` for draw.

---

## Example Outputs

```
Training Iteration 1/5
Generating 10 self-play games...
Training the neural network...
Saved model: game_model_iter1.pth
...
Evaluating AI Strength
New Agent Wins: 32, Old Agent Wins: 18, Draws: 0
Final ELO Ratings - New Agent: 1554.2, Old Agent: 1445.8
```

---

## Components Explained

### `SnortGame`

* Core game engine.
* Manages board state, turns, legal moves, and end conditions.
* Functions: `make_move`, `unmake_move`, `legal_moves`, `encode`, `decode`, `play()`

### `MCTSPlayer`

* Pure MCTS agent with simulated playouts.
* Uses visit count to choose moves.

### `PUCTPlayer`

* MCTS guided by a neural network.
* Uses AlphaZero-style PUCT with Dirichlet noise for exploration.

### `CNNGameNetwork`

* PyTorch CNN.
* Inputs: 4-channel encoded board (R, B, X, Player).
* Outputs: `value` ∈ \[-1,1], `policy` ∈ \[0,1]^25.

### `generate_self_play_games()` / `generate_puct_self_play_games()`

* Simulate games with MCTS/PUCT.
* Record `(state, policy, value)` at each move.

### `train_neural_network()`

* Trains the CNN from self-play data.
* Loss: MSE + CrossEntropy
* Optimizer: Adam

### `evaluate_agents()`

* Plays games between two agents.
* Calculates ELO updates.

---

## About Snort

Snort is a two-player combinatorial game played on a graph. In this simplified version:

* Players alternate placing their colored pieces (Red/Blue) on a 5x5 grid
* A move is legal if no adjacent opponent piece is present
* The game ends when no legal moves remain

---

##  Notes

* The current setup uses manual commenting to toggle between training, evaluation, and play. You can replace this with a menu for cleaner workflow.
* Model files (`.pth`) will be generated automatically during training.

---


---

## Acknowledgments

AlphaZero-style reinforcement learning for board games.
