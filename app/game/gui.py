import sys
from collections.abc import Mapping
from types import MappingProxyType

import pygame

from app.game.env import LShapedGridWorldEnv
from app.game.vos import Action


class LShapedGridWorldGUI:
    """Pygame GUI for the L-shaped grid world environment."""

    CELL_SIZE: int = 40
    MARGIN: int = 2
    FONT_SIZE: int = 24
    FPS: int = 10
    WINDOW_TITLE: str = "L-Shaped Grid World"

    QUIT_KEY: int = pygame.K_q
    RESTART_KEY: int = pygame.K_r

    # Colors
    BG_COLOR: tuple[int, int, int] = (30, 30, 30)
    GRID_COLOR: tuple[int, int, int] = (200, 200, 200)
    AGENT_COLOR: tuple[int, int, int] = (0, 0, 255)
    HOLE_COLOR: tuple[int, int, int] = (50, 50, 50)
    GOAL_COLOR: tuple[int, int, int] = (0, 200, 0)
    COIN_COLOR: tuple[int, int, int] = (255, 215, 0)
    TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)

    KEY_ACTION_MAP: Mapping[int, Action] = MappingProxyType(
        {
            pygame.K_UP: Action.UP,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_DOWN: Action.DOWN,
            pygame.K_LEFT: Action.LEFT,
        }
    )

    def __init__(self, env: LShapedGridWorldEnv) -> None:
        self.env: LShapedGridWorldEnv = env

        self.score: float = 0.0
        self.game_over: bool = False
        self.win: bool = False

        self._init_pygame()
        self._init_display()
        self._init_font()

    def _init_pygame(self) -> None:
        pygame.init()
        self.clock = pygame.time.Clock()

    def _init_display(self) -> None:
        grid_w = self.env.width * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
        grid_h = self.env.height * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
        info_h = self.FONT_SIZE + 2 * self.MARGIN

        self.screen = pygame.display.set_mode((grid_w, grid_h + info_h))
        pygame.display.set_caption(self.WINDOW_TITLE)

    def _init_font(self) -> None:
        self.font = pygame.font.Font(None, self.FONT_SIZE)

    def run(self) -> None:
        self._reset_game()

        while True:
            self._handle_events()
            self._draw()
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def _reset_game(self) -> None:
        self.env.reset()
        self.score = 0.0
        self.game_over = False
        self.win = False

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == self.QUIT_KEY:
                    pygame.quit()
                    sys.exit()
                if event.key == self.RESTART_KEY:
                    self._reset_game()

                if not self.game_over and event.key in self.KEY_ACTION_MAP:
                    action = self.KEY_ACTION_MAP[event.key]

                    _, reward, done = self.env.step(action)
                    self.score += reward
                    if done:
                        self.game_over = True
                        self.win = reward == self.env.GOAL_REWARD

    def _draw(self) -> None:
        self.screen.fill(self.BG_COLOR)
        self._draw_grid()
        self._draw_info()

        if self.game_over:
            self._draw_end_screen()

    def _draw_grid(self) -> None:
        for row in range(self.env.height):
            for col in range(self.env.width):
                if not self.env.mask[row, col]:
                    continue

                x = self.MARGIN + col * (self.CELL_SIZE + self.MARGIN)
                y = self.MARGIN + row * (self.CELL_SIZE + self.MARGIN)
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

                pos = (row, col)
                if pos == self.env.goal_pos:
                    color = self.GOAL_COLOR
                elif pos in self.env.holes:
                    color = self.HOLE_COLOR
                else:
                    color = self.GRID_COLOR

                pygame.draw.rect(self.screen, color, rect)

                if pos in self.env.coins:
                    center = (
                        x + self.CELL_SIZE // 2,
                        y + self.CELL_SIZE // 2,
                    )
                    radius = self.CELL_SIZE // 4
                    pygame.draw.circle(self.screen, self.COIN_COLOR, center, radius)

        agent_row, agent_col = self.env.current_pos
        x = self.MARGIN + agent_col * (self.CELL_SIZE + self.MARGIN)
        y = self.MARGIN + agent_row * (self.CELL_SIZE + self.MARGIN)

        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.AGENT_COLOR, rect)

    # ruff: noqa: FBT003
    def _draw_info(self) -> None:
        info_y = self.env.height * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
        text_lines = [f"Score: {self.score:.2f}", "Press R to restart, Q to quit"]

        for idx, line in enumerate(text_lines):
            text_surf = self.font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(
                text_surf, (self.MARGIN, info_y + idx * (self.FONT_SIZE + self.MARGIN))
            )

    def _draw_end_screen(self) -> None:
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        status = "You Win!" if self.win else "Game Over!"
        text_lines = [
            status,
            f"Final Score: {self.score:.2f}",
            "Press R to restart, Q to quit",
        ]

        total_height = (
            len(text_lines) * self.FONT_SIZE + (len(text_lines) - 1) * self.MARGIN
        )
        start_y = (self.screen.get_height() - total_height) / 2

        for idx, line in enumerate(text_lines):
            text_surf = self.font.render(line, True, self.TEXT_COLOR)
            text_rect = text_surf.get_rect(
                center=(
                    self.screen.get_width() / 2,
                    start_y + idx * (self.FONT_SIZE + self.MARGIN),
                )
            )
            self.screen.blit(text_surf, text_rect)
