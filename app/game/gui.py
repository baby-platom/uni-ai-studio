import sys

import pygame

from app.game.env import TileMergingEnv
from app.game.vos import TileMergingAction


# ruff: noqa: FBT003
class TileMergingGUI:
    """GUI for a 2048-style tile merging game with obstacles."""

    GAME_NAME = "2048 with Obstacles"
    default_tile_color = (60, 58, 50)

    def __init__(
        self,
        size: int = 4,
        cell_size: int = 100,
        margin: int = 5,
        fps: int = 60,
    ) -> None:
        """Initialize pygame, window, environment, fonts, and colors.

        Args:
            size: The size of the grid.
            cell_size: The size of each cell in pixels.
            margin: The margin between cells in pixels.
            fps: The number of frames per second.
        """

        self.size = size
        self.cell_size = cell_size
        self.margin = margin
        self.fps = fps

        self.env: TileMergingEnv

        self.__init_pygame()
        self.__init_env()
        self.__init_fonts()
        self.__init_colors()

        self.state = self.env.reset()

    def __init_pygame(self) -> None:
        """Initialize pygame display and clock."""
        pygame.init()

        self.grid_px = self.size * self.cell_size + (self.size + 1) * self.margin
        self.score_bar_px = 50
        self.screen_size = (
            self.grid_px,
            self.grid_px + self.score_bar_px,
        )

        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption(self.GAME_NAME)
        self.clock = pygame.time.Clock()

    def __init_env(self) -> None:
        self.env = TileMergingEnv(size=self.size, obstacle_enabled=True)

    def __init_fonts(self) -> None:
        """Load fonts for rendering text."""
        self.tile_font = pygame.font.SysFont(None, self.cell_size // 3)
        self.score_font = pygame.font.SysFont(None, 36)
        self.info_font = pygame.font.SysFont(None, 24)

    def __init_colors(self) -> None:
        """Define colors for tiles and obstacles."""
        self.color_map: dict[int, tuple[int, int, int]] = {
            0: (205, 193, 180),
            -1: (119, 110, 101),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }

    def _draw(self) -> None:
        """Draw grid, tiles, score, and controls info."""
        self.screen.fill((187, 173, 160))

        for row in range(self.size):
            for col in range(self.size):
                value = int(self.state[row, col])

                rect = pygame.Rect(
                    self.margin + col * (self.cell_size + self.margin),
                    self.margin + row * (self.cell_size + self.margin),
                    self.cell_size,
                    self.cell_size,
                )
                color = self.color_map.get(value, self.default_tile_color)
                pygame.draw.rect(self.screen, color, rect, border_radius=8)

                if value > 0:
                    text = self.tile_font.render(str(value), True, (0, 0, 0))
                    self.screen.blit(text, text.get_rect(center=rect.center))

        self.__draw_score()
        self.__draw_controls_info()

        pygame.display.flip()

    def __draw_score(self) -> None:
        score_surf = self.score_font.render(f"Score: {self.env.score}", True, (0, 0, 0))
        self.screen.blit(
            score_surf,
            (10, self.grid_px + (self.score_bar_px - score_surf.get_height()) // 2),
        )

    def __draw_controls_info(self) -> None:
        info_text = "Arrows: Move  |  R: Restart  |  Q: Quit"
        info_surf = self.info_font.render(info_text, True, (0, 0, 0))
        info_pos = (
            self.screen_size[0] - info_surf.get_width() - 10,
            self.grid_px + (self.score_bar_px - info_surf.get_height()) // 2,
        )
        self.screen.blit(info_surf, info_pos)

    def _handle_key(self, key: int) -> bool:
        """Step through environment."""
        mapping = {
            pygame.K_UP: TileMergingAction.UP,
            pygame.K_DOWN: TileMergingAction.DOWN,
            pygame.K_LEFT: TileMergingAction.LEFT,
            pygame.K_RIGHT: TileMergingAction.RIGHT,
        }

        if key in mapping:
            self.state, _, done = self.env.step(mapping[key])
            return done
        return False

    def _display_game_over(self) -> None:
        overlay = pygame.Surface(self.screen_size)
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (0, 0))

        go_font = pygame.font.SysFont(None, 72)
        go_surf = go_font.render("Game Over", True, (255, 0, 0))
        self.screen.blit(
            go_surf,
            go_surf.get_rect(
                center=(self.screen_size[0] // 2, self.screen_size[1] // 2 - 30)
            ),
        )

        prompt_font = pygame.font.SysFont(None, 36)
        prompt_surf = prompt_font.render(
            "Press R to Restart or Q to Quit", True, (0, 0, 0)
        )
        self.screen.blit(
            prompt_surf,
            prompt_surf.get_rect(
                center=(self.screen_size[0] // 2, self.screen_size[1] // 2 + 30)
            ),
        )

        pygame.display.flip()

    def run(self) -> None:
        running = True
        game_over = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.state = self.env.reset()
                        game_over = False

                    elif not game_over:
                        game_over = self._handle_key(event.key)

            if game_over:
                self._display_game_over()
            else:
                self._draw()

            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()
