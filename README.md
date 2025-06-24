# uni-ai-studio
The repository contains the code for Reinforcement Learning for a Game.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)

## Overview
In the root folder `app` are two major packages: `game` and `reinforcement_learning`.
1. `game` - the implementation of the environment of a custom GreedyMap game and its GUI.
2. `reinforcement_learning` - the Q-learning and random agents; the code for training, analyzing, testing, and evaluating the results. 

### Game
The game's environment is an L-shaped 8x8 map with two 3x3 arms.
1. Holes - randomly selected cells that end the game when landed on.
2. Coins - collectable items randomly distributed on the map. A player get's an extra reward for collecting them. Every coin can be collected only once in each episode.

#### Rewards:
- `+2` for reaching the goal (win).
- `+0.5` for collecting a coin.
- `-1.0` for getting into a hole (game over).
- `-0.05` for a regular step.

### Reinforcement Learning
The Q-learning algorithm was selected, where a state is represented as a tuple with two elements: `current_cell_idx` and `sorted_tuple_of_uncollected_coin_idxs`. The RL steps:
1. The training of agents with different sets of the hyperparameters.
2. Calculating metrics like convergence rate, moving average reward, and moving average success rate.
3. Testing the trained policies, by using mean success rate and mean episode length.
4. All the metrics are compared with a base line using the random agent. 

## Installation
The required Python version is `3.12.7`

1. Clone the repo
```sh
git clone https://github.com/baby-platom/uni-ai-studio.git
```

2. Create virtual environment and install the dependencies
```sh
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Optionally: 
- Use [uv](https://docs.astral.sh/uv/) for dependencies management
- Use [ruff](https://docs.astral.sh/ruff/) as a linter and code formatter

## Usage
Both `game` and `reinforcement_learning` packages contain the `main.py` files, which are the entrypoints. To use any of the packages, just run the files

### Play the Game
```sh
python -m app.game.main
```

### Reinforcement Learning
```sh
python -m app.reinforcement_learning.main
```
