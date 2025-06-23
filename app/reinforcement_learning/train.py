import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.game.env import LShapedGridWorldEnv
from app.reinforcement_learning.agents import QLearningAgent, RandomAgent


def run_episode(
    env: LShapedGridWorldEnv,
    agent: QLearningAgent | RandomAgent,
) -> tuple[float, int, int]:
    state = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)

        total_reward += reward
        steps += 1
        state = next_state

        if done:
            break

    success = 1 if total_reward > 0 else 0
    return total_reward, steps, success


def run_training(
    seed: int,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Train a single Q-learning agent for a fixed number of episodes."""
    env = LShapedGridWorldEnv(seed=seed)
    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
    )

    rewards = []
    lengths = []
    successes = []

    for _ in range(episodes):
        reward, steps, success = run_episode(env, agent)
        rewards.append(reward)
        lengths.append(steps)
        successes.append(success)

    return {
        "rewards": np.array(rewards),
        "lengths": np.array(lengths),
        "successes": np.array(successes, dtype=float),
    }


def run_training_random(
    seed: int,
    episodes: int,
) -> dict[str, np.ndarray]:
    """Run a purely random agent."""
    env = LShapedGridWorldEnv(seed=seed)
    agent = RandomAgent(seed=seed)
    rewards, lengths, successes = [], [], []

    for _ in range(episodes):
        reward, steps, success = run_episode(env, agent)
        rewards.append(reward)
        lengths.append(steps)
        successes.append(success)

    return {
        "rewards": np.array(rewards),
        "lengths": np.array(lengths),
        "successes": np.array(successes, dtype=float),
    }


def evaluate_parameter_grid(
    alphas: list[float],
    gammas: list[float],
    epsilons: list[float],
    runs: int,
    episodes: int,
    window_fraction: float = 0.1,
    success_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Evaluate combinations of hyperparameters and a random baseline."""
    results = []
    window = max(1, int(episodes * window_fraction))
    param_grid = list(itertools.product(alphas, gammas, epsilons))

    for alpha, gamma, epsilon in param_grid:
        reward_matrix = []
        success_matrix = []

        for run_idx in range(runs):
            seed = run_idx
            data = run_training(seed, episodes, alpha, gamma, epsilon)
            reward_matrix.append(data["rewards"])
            success_matrix.append(data["successes"])

        reward_matrix = np.vstack(reward_matrix)
        success_matrix = np.vstack(success_matrix)
        mean_rewards = reward_matrix.mean(axis=0)
        mean_success = success_matrix.mean(axis=0)

        mov_avg_rewards = (
            pd.Series(mean_rewards).rolling(window, min_periods=1).mean().values
        )
        mov_avg_success = (
            pd.Series(mean_success).rolling(window, min_periods=1).mean().values
        )

        converged_at = next(
            (i + 1 for i, v in enumerate(mov_avg_success) if v >= success_threshold),
            None,
        )

        results.append(
            {
                "label": f"α={alpha}, γ={gamma}, ε={epsilon}",
                "mean_rewards": mean_rewards,
                "mean_success": mean_success,
                "mov_avg_rewards": mov_avg_rewards,
                "mov_avg_success": mov_avg_success,
                "convergence_episode": converged_at,
            }
        )

    results.append(
        _evaluate_random_baseline(
            episodes,
            runs,
            window,
            success_threshold,
        )
    )

    return results


def _evaluate_random_baseline(
    episodes: int,
    runs: int,
    window: int,
    success_threshold: float,
) -> dict[str, Any]:
    reward_matrix = []
    success_matrix = []
    for run_idx in range(runs):
        data = run_training_random(run_idx, episodes)
        reward_matrix.append(data["rewards"])
        success_matrix.append(data["successes"])

    reward_matrix = np.vstack(reward_matrix)
    success_matrix = np.vstack(success_matrix)
    mean_rewards = reward_matrix.mean(axis=0)
    mean_success = success_matrix.mean(axis=0)
    mov_avg_rewards = (
        pd.Series(mean_rewards).rolling(window, min_periods=1).mean().values
    )
    mov_avg_success = (
        pd.Series(mean_success).rolling(window, min_periods=1).mean().values
    )
    converged_at = next(
        (i + 1 for i, v in enumerate(mov_avg_success) if v >= success_threshold),
        None,
    )

    return {
        "label": "Random baseline",
        "mean_rewards": mean_rewards,
        "mean_success": mean_success,
        "mov_avg_rewards": mov_avg_rewards,
        "mov_avg_success": mov_avg_success,
        "convergence_episode": converged_at,
    }


def plot_learning_curves(results: list[dict[str, Any]], episodes: int) -> None:
    plt.figure(figsize=(10, 6))

    for res in results:
        x = np.arange(1, episodes + 1)
        y = res["mov_avg_rewards"]
        plt.plot(x, y, label=res["label"])

    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward")

    plt.title("Learning Curves (Moving Avg Reward)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_success_rate_curves(results: list[dict[str, Any]], episodes: int) -> None:
    plt.figure(figsize=(10, 6))

    for res in results:
        x = np.arange(1, episodes + 1)
        y = res["mov_avg_success"]
        plt.plot(x, y, label=res["label"])

    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Success Rate")

    plt.title("Success Rate Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
