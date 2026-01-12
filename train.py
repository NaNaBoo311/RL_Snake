# train.py
from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

from env import SnakeGame
from features import state_to_features
from dqn_agent import DQNAgent, DQNConfig


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly decay epsilon from start -> end over decay_steps."""
    if step >= decay_steps:
        return end
    frac = step / float(decay_steps)
    return start + frac * (end - start)


def main():
    # -----------------------------
    # Config you can tweak
    # -----------------------------
    SEED = 0
    GRID_W, GRID_H = 20, 20

    NUM_EPISODES = 3000          # start small to sanity-check; increase later
    MAX_STEPS_PER_EP = 10_000   # env has its own stall termination anyway

    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY_STEPS = 20_000   # decay over environment steps, not episodes

    PRINT_EVERY = 10
    SAVE_DIR = "models"
    BEST_PATH = os.path.join(SAVE_DIR, "best.pt")
    LAST_PATH = os.path.join(SAVE_DIR, "last.pt")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # -----------------------------
    # Env + Agent
    # -----------------------------
    env = SnakeGame(width=GRID_W, height=GRID_H, seed=SEED)

    cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_capacity=100_000,
        start_learning_after=1_000,
        target_update_every=1_000,
        max_grad_norm=10.0,
    )
    agent = DQNAgent(config=cfg, seed=SEED)

    # Stats
    scores_window = deque(maxlen=100)
    best_avg100 = -1e9
    global_step = 0
    t0 = time.time()

    # -----------------------------
    # Training loop
    # -----------------------------
    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        obs = state_to_features(state)

        ep_score = 0
        ep_steps = 0
        losses = []

        done = False
        while not done and ep_steps < MAX_STEPS_PER_EP:
            epsilon = linear_epsilon(global_step, EPS_START, EPS_END, EPS_DECAY_STEPS)

            action = agent.select_action(obs, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            next_obs = state_to_features(next_state)

            agent.store(obs, action, reward, next_obs, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            ep_steps += 1
            global_step += 1
            ep_score = next_state.score  # score = foods eaten so far

        scores_window.append(ep_score)

        avg100 = float(np.mean(scores_window))
        avg_loss = float(np.mean(losses)) if losses else float("nan")

        # Save best model by avg100
        if len(scores_window) == scores_window.maxlen and avg100 > best_avg100:
            best_avg100 = avg100
            agent.save(BEST_PATH)

        # Always save last
        agent.save(LAST_PATH)

        if ep % PRINT_EVERY == 0 or ep == 1:
            elapsed = time.time() - t0
            epsilon_now = linear_epsilon(global_step, EPS_START, EPS_END, EPS_DECAY_STEPS)
            print(
                f"Ep {ep:4d} | score {ep_score:3d} | avg100 {avg100:6.2f} | "
                f"eps {epsilon_now:5.2f} | steps {global_step:7d} | loss {avg_loss:7.4f} | "
                f"time {elapsed:6.1f}s"
            )

    print("\nTraining complete.")
    print(f"Saved last model to: {LAST_PATH}")
    if os.path.exists(BEST_PATH):
        print(f"Saved best model to: {BEST_PATH} (best avg100={best_avg100:.2f})")
    else:
        print("Best model not saved yet (need >=100 episodes to compute avg100).")


if __name__ == "__main__":
    main()
