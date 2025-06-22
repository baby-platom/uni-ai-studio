import numpy as np

from app.game.vos import Action


class QLearningAgent:
    ACTION_SPACE = tuple(Action)

    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon: float,
        seed: int | None = None,
    ) -> None:
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.rng = np.random.RandomState(seed)

        self.q_table: dict[int, np.ndarray] = {}

    def _get_q_values(self, state: int) -> np.ndarray:
        """Return Q-values for the given state, initializing to zero if unseen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.ACTION_SPACE), dtype=float)
        return self.q_table[state]

    def choose_action(self, state: int) -> Action:
        if self.rng.rand() < self.epsilon:
            return self.rng.choice(self.ACTION_SPACE)  # Explore

        q_vals: np.ndarray = self._get_q_values(state)
        best_idx: int = int(np.argmax(q_vals))
        return self.ACTION_SPACE[best_idx]  # Exploit

    # ruff: noqa: FBT001
    def update(
        self,
        state: int,
        action: Action,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Perform the Q-learning update for a single transition."""
        q_vals: np.ndarray = self._get_q_values(state)
        current: float = q_vals[action.value]

        target = reward
        if not done:
            next_q_vals: np.ndarray = self._get_q_values(next_state)
            target = reward + self.gamma * np.max(next_q_vals)

        q_vals[action.value] = current + self.alpha * (target - current)
