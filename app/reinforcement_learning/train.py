import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.game.env import FrozenLLakeEnv
from app.reinforcement_learning.agent import QLearningAgent


def run_episode(env: FrozenLLakeEnv, agent: QLearningAgent) -> tuple[float, int, int]:
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
    env = FrozenLLakeEnv(seed=seed)
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


def evaluate_parameter_grid(
    alphas: list[float],
    gammas: list[float],
    epsilons: list[float],
    runs: int,
    episodes: int,
) -> list[dict[str, Any]]:
    """Evaluate combinations of hyperparameters."""
    results = []

    param_grid = list(itertools.product(alphas, gammas, epsilons))

    for alpha, gamma, epsilon in param_grid:
        reward_matrix = []
        success_matrix = []

        for run_idx in range(runs):
            seed = run_idx
            data = run_training(
                seed,
                episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
            )

            reward_matrix.append(data["rewards"])
            success_matrix.append(data["successes"])

        reward_matrix = np.vstack(reward_matrix)
        success_matrix = np.vstack(success_matrix)
        mean_rewards = reward_matrix.mean(axis=0)
        mean_success = success_matrix.mean(axis=0)

        window = min(100, episodes // 10)
        mov_avg_rewards = (
            pd.Series(mean_rewards).rolling(window, min_periods=1).mean().values
        )

        threshold = 0.8
        converged_at = next(
            (i + 1 for i, v in enumerate(mov_avg_rewards) if v >= threshold), None
        )

        results.append(
            {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "mean_rewards": mean_rewards,
                "mean_success": mean_success,
                "mov_avg_rewards": mov_avg_rewards,
                "convergence_episode": converged_at,
            }
        )

    return results


def plot_learning_curves(results: list[dict[str, Any]], episodes: int) -> None:
    """Plot the smoothed reward curves for each hyperparameter setting."""
    plt.figure(figsize=(10, 6))

    for res in results:
        label = f"α={res['alpha']}, γ={res['gamma']}, ε={res['epsilon']}"
        plt.plot(range(1, episodes + 1), res["mov_avg_rewards"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward")

    plt.title("Learning Curves (Moving Avg Reward)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    episodes = 1000
    runs = 10

    alphas = [0.1, 0.5]
    gammas = [0.9, 0.99]
    epsilons = [0.05, 0.1, 0.2]

    results = evaluate_parameter_grid(
        alphas,
        gammas,
        epsilons,
        runs,
        episodes,
    )

    for res in results:
        print(
            f"α={res['alpha']}, "
            f"γ={res['gamma']}, "
            f"ε={res['epsilon']}: "
            f"converged at episode {res['convergence_episode']}"
        )

    plot_learning_curves(results, episodes)


if __name__ == "__main__":
    main()
