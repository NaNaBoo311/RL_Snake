# features.py
from __future__ import annotations

from typing import Tuple, List
import numpy as np

from env import SnakeState, SnakeGame, Coord


def _collision(next_cell: Coord, snake: List[Coord], width: int, height: int) -> bool:
    """True if next_cell would collide with wall or snake body."""
    x, y = next_cell
    if x < 0 or x >= width or y < 0 or y >= height:
        return True
    # Same collision rule as env: allow moving into tail because tail moves away (if not eating).
    # For *danger* estimation, we assume a normal move (tail moves), so check against snake[:-1].
    if next_cell in snake[:-1]:
        return True
    return False


def _turn(direction: Coord, action: int) -> Coord:
    """
    Same turning logic as env.py:
      0 = straight, 1 = left, 2 = right
    """
    if action not in (SnakeGame.STRAIGHT, SnakeGame.TURN_LEFT, SnakeGame.TURN_RIGHT):
        raise ValueError("action must be 0(straight),1(left),2(right)")

    if action == SnakeGame.STRAIGHT:
        return direction

    order = [SnakeGame.UP, SnakeGame.RIGHT, SnakeGame.DOWN, SnakeGame.LEFT]  # clockwise
    idx = order.index(direction)
    if action == SnakeGame.TURN_LEFT:
        return order[(idx - 1) % 4]
    else:
        return order[(idx + 1) % 4]


def state_to_features(state: SnakeState) -> np.ndarray:
    """
    Convert SnakeState -> 11-dim feature vector:
    [danger_straight, danger_left, danger_right,
     dir_up, dir_down, dir_left, dir_right,
     food_left, food_right, food_up, food_down]
    """
    head_x, head_y = state.snake[0]
    dx, dy = state.direction

    # Directions after relative actions
    dir_straight = _turn(state.direction, SnakeGame.STRAIGHT)
    dir_left = _turn(state.direction, SnakeGame.TURN_LEFT)
    dir_right = _turn(state.direction, SnakeGame.TURN_RIGHT)

    # Next cells for each possible relative move
    straight_cell = (head_x + dir_straight[0], head_y + dir_straight[1])
    left_cell = (head_x + dir_left[0], head_y + dir_left[1])
    right_cell = (head_x + dir_right[0], head_y + dir_right[1])

    danger_straight = 1 if _collision(straight_cell, state.snake, state.width, state.height) else 0
    danger_left = 1 if _collision(left_cell, state.snake, state.width, state.height) else 0
    danger_right = 1 if _collision(right_cell, state.snake, state.width, state.height) else 0

    # Current direction one-hot
    dir_up = 1 if state.direction == SnakeGame.UP else 0
    dir_down = 1 if state.direction == SnakeGame.DOWN else 0
    dir_left_oh = 1 if state.direction == SnakeGame.LEFT else 0
    dir_right_oh = 1 if state.direction == SnakeGame.RIGHT else 0

    # Food relative position
    food_x, food_y = state.food
    food_left = 1 if food_x < head_x else 0
    food_right = 1 if food_x > head_x else 0
    food_up = 1 if food_y < head_y else 0
    food_down = 1 if food_y > head_y else 0

    features = np.array(
        [
            danger_straight,
            danger_left,
            danger_right,
            dir_up,
            dir_down,
            dir_left_oh,
            dir_right_oh,
            food_left,
            food_right,
            food_up,
            food_down,
        ],
        dtype=np.float32,
    )
    return features


# Quick test
if __name__ == "__main__":
    from env import SnakeGame

    env = SnakeGame(width=10, height=10, seed=0)
    s = env.reset()
    f = state_to_features(s)
    print("features shape:", f.shape)
    print("features:", f.tolist())
