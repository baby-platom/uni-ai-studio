from app.reinforcement_learning.evaluate import plot_evaluation, train_and_evaluate
from app.reinforcement_learning.train import (
    evaluate_parameter_grid,
    plot_learning_curves,
    plot_success_rate_curves,
)


def main() -> None:
    episodes = 5000
    runs = 2

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
        print(f"{res['label']}: converged at episode {res['convergence_episode']}")

    plot_learning_curves(results, episodes)
    plot_success_rate_curves(results, episodes)

    eval_results = train_and_evaluate(
        alphas,
        gammas,
        epsilons,
        train_episodes=episodes,
        runs=runs,
    )
    plot_evaluation(eval_results)


if __name__ == "__main__":
    main()
