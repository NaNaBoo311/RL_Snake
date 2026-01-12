from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Coord = Tuple[int, int] #(x,y)

@dataclass
class SnakeState:
    width: int
    height: int
    snake: List[Coord]
    direction: Coord
    food: Coord
    score: int
    steps_since_food: int


class SnakeGame:
    """
    Pure Snake game logic (no pygame).
    Grid coordinates: (x, y) where:
      - x increases to the right
      - y increases downward
    """
    
    #Directions in (dx,dy)
    UP: Coord = (0, -1)
    DOWN: Coord = (0, 1)
    LEFT: Coord = (-1, 0)
    RIGHT: Coord = (1, 0)

    #Actions
    STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        seed: Optional[int] = None,
        step_penalty: float = -0.01,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        max_steps_without_food_factor: int = 100,
    ):
        #Board size
        self.width = width
        self.height = height

        #Rewards and penalties
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.max_steps_without_food_factor = max_steps_without_food_factor

        #Random number generator
        self._rng = random.Random(seed)
        self.state: Optional[SnakeState] = None

    def seed(self, seed: int) -> None:
        """Set RNG seed for reproducibility."""
        self._rng.seed(seed)

    def reset(self) -> SnakeState:
        """Start a new episode and return the initial state."""
        cx, cy = self.width // 2, self.height // 2

        # Start moving RIGHT with length 3
        direction = self.RIGHT
        snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]

        food = self._spawn_food(snake)

        self.state = SnakeState(
            width=self.width,
            height=self.height,
            snake=snake,
            direction=direction,
            food=food,
            score=0,
            steps_since_food=0,
        )
        return self.state

    def step(self, action: int) -> Tuple[SnakeState, float, bool, Dict]:
        """
        Apply action and advance the game by 1 tick.

        Returns:
          next_state, reward, done, info
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        # 1) Update direction based on relative action
        new_dir = self._turn(self.state.direction, action)

        head_x, head_y = self.state.snake[0]
        dx, dy = new_dir
        new_head = (head_x + dx, head_y + dy)

        # 2) Compute reward baseline
        reward = float(self.step_penalty)
        done = False
        info: Dict = {}

        # 3) Check collision with walls
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            reward = float(self.death_penalty)
            done = True
            info["reason"] = "wall"
            # update direction for consistency
            self.state.direction = new_dir
            return self.state, reward, done, info

        # 4) Check collision with self
        # Note: moving into the tail is allowed ONLY if the tail is going to move away.
        # We'll handle this by checking collision against the body excluding the last cell,
        # because if we are not eating food, the tail will be removed.
        body = self.state.snake[:-1]
        if new_head in body:
            reward = float(self.death_penalty)
            done = True
            info["reason"] = "self"
            self.state.direction = new_dir
            return self.state, reward, done, info

        # 5) Move snake
        ate_food = (new_head == self.state.food)
        new_snake = [new_head] + self.state.snake

        if ate_food:
            reward = float(self.food_reward)
            self.state.score += 1
            self.state.steps_since_food = 0
            # keep tail (grow), spawn new food
            new_food = self._spawn_food(new_snake)
            self.state.food = new_food
        else:
            # remove tail
            new_snake.pop()
            self.state.steps_since_food += 1

        self.state.snake = new_snake
        self.state.direction = new_dir

        # 6) Anti-stall termination
        max_no_food = self.max_steps_without_food_factor * max(1, len(self.state.snake))
        if self.state.steps_since_food >= max_no_food:
            reward = float(self.death_penalty)
            done = True
            info["reason"] = "stall"

        return self.state, reward, done, info

    # -------------------------
    # Helpers
    # -------------------------

    def _spawn_food(self, snake: List[Coord]) -> Coord:
        """Spawn food in a random empty cell."""
        occupied = set(snake)
        empties = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        if not empties:
            # no empty cell (won the game)
            # put food anywhere; episode should end in training logic if you want
            return (0, 0)
        return self._rng.choice(empties)

    def _turn(self, direction: Coord, action: int) -> Coord:
        """Convert relative action (straight/left/right) into a new direction."""
        if action not in (self.STRAIGHT, self.TURN_LEFT, self.TURN_RIGHT):
            raise ValueError(f"Invalid action {action}. Must be 0(straight),1(left),2(right).")

        if action == self.STRAIGHT:
            return direction

        # Direction order clockwise: UP -> RIGHT -> DOWN -> LEFT
        order = [self.UP, self.RIGHT, self.DOWN, self.LEFT]
        idx = order.index(direction)

        if action == self.TURN_LEFT:
            return order[(idx - 1) % 4]
        else:  # TURN_RIGHT
            return order[(idx + 1) % 4]

# Quick manual test (only runs when executing env.py directly)
if __name__ == "__main__":
    env = SnakeGame(width=20, height=20, seed=0)
    s = env.reset()
    print("Initial head:", s.snake[0], "food:", s.food, "dir:", s.direction)

    total_reward = 0.0
    for t in range(50):
        action = env._rng.choice([0, 1, 2])  # random policy
        s, r, done, info = env.step(action)
        total_reward += r
        if done:
            print(f"Done at step {t}, reason={info.get('reason')}, score={s.score}, total_reward={total_reward:.2f}")
            break
    else:
        print(f"Survived 50 steps, score={s.score}, total_reward={total_reward:.2f}")