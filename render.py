# render.py
from __future__ import annotations
from typing import Tuple
import pygame

from env import SnakeState

Color = Tuple[int, int, int]


class SnakeRenderer:
    """
    Pygame renderer for SnakeState.
    This class only draws. Game logic stays in env.py.
    """

    def __init__(self, cell_size: int = 24, margin: int = 2):
        self.cell_size = cell_size
        self.margin = margin

        # You can tweak these colors later
        self.bg: Color = (25, 25, 25)
        self.grid: Color = (35, 35, 35)
        self.snake: Color = (80, 220, 120)
        self.head: Color = (40, 180, 90)
        self.food: Color = (220, 80, 80)
        self.text: Color = (235, 235, 235)

        self.font = None

    def init_display(self, width: int, height: int) -> pygame.Surface:
        pygame.init()
        w_px = width * self.cell_size
        h_px = height * self.cell_size + 40  # space for score text
        screen = pygame.display.set_mode((w_px, h_px))
        pygame.display.set_caption("Snake RL - Play")
        self.font = pygame.font.SysFont("consolas", 22)
        return screen

    def _cell_rect(self, x: int, y: int) -> pygame.Rect:
        cs = self.cell_size
        m = self.margin
        return pygame.Rect(x * cs + m, y * cs + m, cs - 2 * m, cs - 2 * m)

    def draw(self, screen: pygame.Surface, state: SnakeState) -> None:
        screen.fill(self.bg)

        # grid lines (optional)
        for x in range(state.width):
            pygame.draw.line(
                screen, self.grid,
                (x * self.cell_size, 0),
                (x * self.cell_size, state.height * self.cell_size),
                1
            )
        for y in range(state.height):
            pygame.draw.line(
                screen, self.grid,
                (0, y * self.cell_size),
                (state.width * self.cell_size, y * self.cell_size),
                1
            )

        # food
        fx, fy = state.food
        pygame.draw.rect(screen, self.food, self._cell_rect(fx, fy))

        # snake body
        for i, (sx, sy) in enumerate(state.snake):
            color = self.head if i == 0 else self.snake
            pygame.draw.rect(screen, color, self._cell_rect(sx, sy))

        # score text area
        if self.font is not None:
            text = self.font.render(f"Score: {state.score}  Length: {len(state.snake)}", True, self.text)
            screen.blit(text, (10, state.height * self.cell_size + 8))

        pygame.display.flip()
