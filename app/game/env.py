from collections import deque

import numpy as np

from app.game.vos import Action


class FrozenLLakeEnv:
    """Custom L-shaped Frozen Lake environment with random holes.

    Rewards:
      - Step: -0.1
      - Goal: +1
      - Hole: -1
    """

    GOAL_REWARD = 1.0
    HOLE_REWARD = -1.0
    STEP_REWARD = -0.1

    DELTAS: tuple[tuple[int, int]] = ((-1, 0), (0, 1), (1, 0), (0, -1))

    height: int
    width: int
    arm_width: int
    arm_height: int
    hole_prob: float

    random_state: np.random.RandomState
    mask: np.ndarray
    pos2state: dict[tuple[int, int], int]
    state2pos: dict[int, tuple[int, int]]

    holes: set[tuple[int, int]]
    start_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    current_pos: tuple[int, int]

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        arm_width: int = 4,
        arm_height: int = 4,
        hole_prob: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.arm_width = arm_width
        self.arm_height = arm_height
        self.hole_prob = hole_prob

        self.random_state = np.random.RandomState(seed)

        self._build_mask()
        self._init_mappings()
        self._generate_holes()

        self.reset()

    def _build_mask(self) -> None:
        """Construct the boolean mask for the L-shaped map."""
        m = np.zeros((self.height, self.width), dtype=bool)
        m[:, : self.arm_width] = True
        m[self.height - self.arm_height :, :] = True
        self.mask = m

    def _init_mappings(self) -> None:
        self.start_pos = (0, 0)
        self.goal_pos = (self.height - 1, self.width - 1)

        coords = [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self.mask[row, col]
        ]
        self.pos2state = {pos: idx for idx, pos in enumerate(coords)}
        self.state2pos = {idx: pos for pos, idx in self.pos2state.items()}

    def _is_reachable(self, holes: set[tuple[int, int]]) -> bool:
        """Check if goal is reachable from start."""
        visited: set[tuple[int, int]] = set()
        queue = deque([self.start_pos])
        visited.add(self.start_pos)

        while queue:
            row, col = queue.popleft()
            if (row, col) == self.goal_pos:
                return True

            for dr, dc in self.DELTAS:
                nxt_row, nxt_col = row + dr, col + dc
                nxt = (nxt_row, nxt_col)

                if (
                    0 <= nxt_row < self.height
                    and 0 <= nxt_col < self.width
                    and self.mask[nxt_row, nxt_col]
                    and nxt not in holes
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    queue.append(nxt)

        return False

    def _generate_holes(self) -> None:
        """Generate random holes until a solvable map is produced."""
        max_attempts = 1000

        for _ in range(max_attempts):
            holes: set[tuple[int, int]] = set()

            for pos in self.pos2state:
                if pos in (self.start_pos, self.goal_pos):
                    continue
                if self.random_state.rand() < self.hole_prob:
                    holes.add(pos)

            if self._is_reachable(holes):
                self.holes = holes
                return

        raise RuntimeError("Unable to generate a solvable map")

    def reset(self) -> int:
        self.current_pos = self.start_pos
        return self.get_state()

    def step(self, action: Action) -> tuple[int, float, bool]:
        """Apply the given action.

        Returns:
            int: The new state.
            float: The reward for the action.
            bool: `True` if the game is over.
        """
        dr, dc = self.DELTAS[action.value]
        row, col = self.current_pos
        new_pos = (row + dr, col + dc)

        if new_pos in self.pos2state:
            self.current_pos = new_pos

        if self.current_pos == self.goal_pos:
            return self.get_state(), self.GOAL_REWARD, True
        if self.current_pos in self.holes:
            return self.get_state(), self.HOLE_REWARD, True

        return self.get_state(), self.STEP_REWARD, False

    def get_state(self) -> int:
        return self.pos2state[self.current_pos]

    def is_done(self) -> bool:
        return self.current_pos == self.goal_pos or self.current_pos in self.holes
