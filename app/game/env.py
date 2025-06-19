import random

import numpy as np

from app.game.vos import TileMergingAction


class TileMergingEnv:
    """
    Environment for a 2048-style tile merging game on an N x N grid.

    The special feature of the game is a random cell filled with an obstacle.
    The obstacle is a cell that doesn't move and cannot be merged with other tiles.

    The board uses:
      - -1 for obstacle cells
      -  0 for empty cells
      - >=2 powers of two for tile values
    """

    random_tile_2_prob = 0.9
    random_tile_4_prob = 1 - random_tile_2_prob

    def __init__(
        self,
        size: int = 4,
        obstacle_enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.obstacle_enabled = obstacle_enabled

        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the board, place an obstacle, add two starting tiles."""
        self.board = np.zeros((self.size, self.size), dtype=int)

        if self.obstacle_enabled:
            empties = [(i, j) for i in range(self.size) for j in range(self.size)]
            self.obstacle_pos = random.choice(empties)
            self.board[self.obstacle_pos] = -1

        self._add_random_tile()
        self._add_random_tile()
        self.score = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def get_available_actions(self) -> list[TileMergingAction]:
        valid_actions: list[TileMergingAction] = []

        for action in TileMergingAction:
            new_board, _ = self._move_board(self.board.copy(), action)
            if not np.array_equal(new_board, self.board):
                valid_actions.append(action)

        return valid_actions

    def step(self, action: TileMergingAction) -> tuple[np.ndarray, int, bool]:
        """
        Apply the given action.

        Returns:
            np.ndarray: The new board state.
            int: The reward for the action.
            bool: `True` if no moves left.
        """
        old_board = self.board.copy()
        self.board, reward = self._move_board(self.board, action)

        # Only add a new tile if the board changed
        if not np.array_equal(old_board, self.board):
            self._add_random_tile()

        self.score += reward
        done = not self.get_available_actions()
        return self.get_state(), reward, done

    def _add_random_tile(self) -> None:
        """Add a 2 or 4 tile to a random empty cell."""
        empty_cells = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.board[i, j] == 0
        ]
        if not empty_cells:
            return

        x, y = random.choice(empty_cells)
        tile_value = 2 if random.random() < self.random_tile_2_prob else 4
        self.board[x, y] = tile_value

    def _move_board(
        self,
        board: np.ndarray,
        action: TileMergingAction,
    ) -> tuple[np.ndarray, int]:
        """Return a new board and reward for the given action, without adding a tile."""
        total_score = 0
        N = self.size

        match action:
            case TileMergingAction.LEFT | TileMergingAction.RIGHT:
                for i in range(N):
                    row = list(board[i, :])
                    if action == TileMergingAction.RIGHT:
                        row = row[::-1]

                    moved, score = self.__move_line_with_obstacle(row)
                    total_score += score
                    if action == TileMergingAction.RIGHT:
                        moved = moved[::-1]
                    board[i, :] = moved
            case TileMergingAction.UP | TileMergingAction.DOWN:
                for j in range(N):
                    col = list(board[:, j])
                    if action == TileMergingAction.DOWN:
                        col = col[::-1]

                    moved, score = self.__move_line_with_obstacle(col)
                    total_score += score
                    if action == TileMergingAction.DOWN:
                        moved = moved[::-1]
                    board[:, j] = moved

        return board, total_score

    def __move_line_with_obstacle(self, line: list[int]) -> tuple[list[int], int]:
        """Process one row/column list with obstacles (-1) splitting into segments."""
        total = 0

        segments: list[list[int]] = []
        idx_segs: list[list[int]] = []
        curr_seg: list[int] = []
        curr_idx: list[int] = []

        for idx, v in enumerate(line):
            if v == -1:
                if curr_seg:
                    segments.append(curr_seg)
                    idx_segs.append(curr_idx)
                    curr_seg = []
                    curr_idx = []
            else:
                curr_seg.append(v)
                curr_idx.append(idx)
        if curr_seg:
            segments.append(curr_seg)
            idx_segs.append(curr_idx)

        # Merge each segment
        updated = {}
        for seg, ids in zip(segments, idx_segs):
            merged, score = self.__merge_segment(seg)
            total += score
            for pos, val in zip(ids, merged):
                updated[pos] = val

        # Rebuild line
        new_line = []
        for i, v in enumerate(line):
            if v == -1:
                new_line.append(-1)
            else:
                new_line.append(updated.get(i, 0))
        return new_line, total

    def __merge_segment(self, line: list[int]) -> tuple[list[int], int]:
        """Merge one segment (no obstacles) by sliding and combining equal tiles."""
        non_zero = [v for v in line if v > 0]
        merged = []
        score = 0

        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                new_val = non_zero[i] * 2
                merged.append(new_val)
                score += new_val
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        merged += [0] * (len(line) - len(merged))
        return merged, score
