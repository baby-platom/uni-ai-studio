from app.reinforcement_learning.test import plot_test_results, train_and_test
from app.reinforcement_learning.train import (
    evaluate_parameter_grid,
    plot_learning_curves,
    plot_success_rate_curves,
)


def main() -> None:
    episodes = 2000
    runs = 10

    alphas = [0.1, 0.5]
    gammas = [0.9, 0.99]
    epsilons = [0.05, 0.1, 0.2]

    window_fraction = 0.1
    window_size = max(1, int(episodes * window_fraction))

    results = evaluate_parameter_grid(
        alphas,
        gammas,
        epsilons,
        runs,
        episodes,
        window_size,
    )

    for res in results:
        print(f"{res['label']}: converged at episode {res['convergence_episode']}")

    plot_learning_curves(results, episodes, window_size)
    plot_success_rate_curves(results, episodes, window_size)

    eval_results = train_and_test(
        alphas,
        gammas,
        epsilons,
        train_episodes=episodes,
        runs=runs,
    )
    plot_test_results(eval_results)


if __name__ == "__main__":
    main()
