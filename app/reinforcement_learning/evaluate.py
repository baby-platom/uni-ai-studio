import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from app.game.env import LShapedGridWorldEnv
from app.reinforcement_learning.agent import QLearningAgent
from app.reinforcement_learning.train import run_episode


def _evaluate_trained_agent(
    env: LShapedGridWorldEnv,
    agent: QLearningAgent,
    episodes: int = 100,
) -> dict[str, float]:
    """Evaluate a trained Q-learning agent by disabling exploration and updates."""
    original_epsilon = agent.epsilon
    original_update = agent.update
    agent.epsilon = 0.0
    agent.update = lambda *_, **__: None

    successes = []
    lengths = []
    for _ in range(episodes):
        _, steps, success = run_episode(env, agent)
        successes.append(success)
        lengths.append(steps)

    agent.epsilon = original_epsilon
    agent.update = original_update

    return {
        "mean_success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def _train_agent(
    seed: int,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> QLearningAgent:
    env = LShapedGridWorldEnv(seed=seed)
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)

    for _ in range(episodes):
        run_episode(env, agent)
    return agent


def train_and_evaluate(
    alphas: list[float],
    gammas: list[float],
    epsilons: list[float],
    train_episodes: int,
    eval_episodes: int = 100,
    runs: int = 10,
) -> list[dict[str, Any]]:
    """Train and evaluate agents over a hyperparameter grid."""
    results: list[dict[str, Any]] = []

    for alpha, gamma, epsilon in itertools.product(alphas, gammas, epsilons):
        success_rates: list[float] = []
        episode_lengths: list[float] = []

        for run_idx in range(runs):
            seed = run_idx
            agent = _train_agent(seed, train_episodes, alpha, gamma, epsilon)
            env = LShapedGridWorldEnv(seed=seed)

            metrics = _evaluate_trained_agent(env, agent, episodes=eval_episodes)
            success_rates.append(metrics["mean_success_rate"])
            episode_lengths.append(metrics["mean_episode_length"])

        results.append(
            {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "mean_success_rate": float(np.mean(success_rates)),
                "mean_episode_length": float(np.mean(episode_lengths)),
            }
        )
    return results


def plot_evaluation(eval_results: list[dict[str, Any]]) -> None:
    labels = [f"α={r['alpha']}, γ={r['gamma']}, ε={r['epsilon']}" for r in eval_results]
    success_rates = [r["mean_success_rate"] for r in eval_results]
    episode_lengths = [r["mean_episode_length"] for r in eval_results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    _ = ax1.bar(
        x - width / 2, success_rates, width, label="Mean Success Rate", color="C0"
    )
    ax1.set_ylabel("Mean Success Rate")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    _ = ax2.bar(
        x + width / 2, episode_lengths, width, label="Mean Episode Length", color="C1"
    )
    ax2.set_ylabel("Mean Episode Length")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.grid(False)
    ax2.grid(False)

    plt.title("Evaluation of Trained Policies")
    fig.tight_layout()
    plt.show()
