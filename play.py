# play.py
from __future__ import annotations

import os
import time
import pygame

from env import SnakeGame
from features import state_to_features
from dqn_agent import DQNAgent, DQNConfig
from render import SnakeRenderer


def main():
    # -----------------------------
    # Settings
    # -----------------------------
    MODEL_PATH = os.path.join("models", "best.pt")  # change to last.pt if you want
    GRID_W, GRID_H = 20, 20
    SEED = 0

    FPS = 15           # increase to speed up, decrease to watch slower
    EPISODES_TO_PLAY = 999999  # effectively infinite; close window to stop

    # -----------------------------
    # Env + Agent
    # -----------------------------
    env = SnakeGame(width=GRID_W, height=GRID_H, seed=SEED)

    cfg = DQNConfig()  # values don't matter much for playing, but required for init
    agent = DQNAgent(config=cfg, seed=SEED)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            f"Make sure you trained and saved it (models/best.pt)."
        )
    agent.load(MODEL_PATH)

    # -----------------------------
    # Pygame init
    # -----------------------------
    renderer = SnakeRenderer(cell_size=24, margin=2)
    screen = renderer.init_display(GRID_W, GRID_H)
    clock = pygame.time.Clock()

    # -----------------------------
    # Play loop
    # -----------------------------
    ep = 0
    state = env.reset()
    obs = state_to_features(state)
    done = False

    running = True
    while running and ep < EPISODES_TO_PLAY:
        # Handle window events (important, otherwise OS says "not responding")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Optional: press R to reset episode manually
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    done = True

        # Agent chooses action greedily (epsilon = 0)
        action = agent.select_action(obs, epsilon=0.0)

        # Step environment
        state, reward, done, info = env.step(action)
        obs = state_to_features(state)

        # Render
        renderer.draw(screen, state)

        # Episode end -> restart after short pause
        if done:
            ep += 1
            print(f"Episode {ep} ended. Score={state.score}, reason={info.get('reason')}")
            time.sleep(0.5)
            state = env.reset()
            obs = state_to_features(state)
            done = False

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
