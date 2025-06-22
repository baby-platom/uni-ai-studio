import random

import numpy as np

from app.game.vos import TileMergingAction


class QLearningAgent:
    def __init__(
        self,
        action_space: list[TileMergingAction],
        alpha: float,
        gamma: float,
        epsilon: float,
    ) -> None:
        self.action_space: list[TileMergingAction] = action_space
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon

        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    def _get_q_values(self, state: tuple[int, ...]) -> np.ndarray:
        """Return Q-values for the given state, initializing to zero if unseen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space), dtype=float)
        return self.q_table[state]

    def choose_action(self, state: tuple[int, ...]) -> TileMergingAction:
        if random.random() < self.epsilon:
            return random.choice(self.action_space)  # Explore

        q_vals: np.ndarray = self._get_q_values(state)
        best_idx: int = int(np.argmax(q_vals))
        return self.action_space[best_idx]  # Exploit

    # ruff: noqa: FBT001
    def update(
        self,
        state: tuple[int, ...],
        action: TileMergingAction,
        reward: float,
        next_state: tuple[int, ...],
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
