# dqn_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Simple MLP for DQN:
      input: 11-dim feature vector
      output: 3 Q-values (straight/left/right)
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-size FIFO replay buffer.
    Stores transitions: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int = 100_000, seed: Optional[int] = None):
        self.buffer: Deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # store as float32 arrays to keep memory small/consistent
        self.buffer.append(
            (
                state.astype(np.float32),
                int(action),
                float(reward),
                next_state.astype(np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),                      # (B, 11)
            np.array(actions, dtype=np.int64),     # (B,)
            np.array(rewards, dtype=np.float32),   # (B,)
            np.stack(next_states),                 # (B, 11)
            np.array(dones, dtype=np.float32),     # (B,) 1.0 if done else 0.0
        )


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    start_learning_after: int = 1_000  # minimum transitions before training
    target_update_every: int = 1_000   # gradient steps
    max_grad_norm: float = 10.0        # gradient clipping for stability


class DQNAgent:
    def __init__(
        self,
        config: DQNConfig = DQNConfig(),
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Seeding (reproducibility)
        self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else 0)
        torch.manual_seed(seed if seed is not None else 0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed if seed is not None else 0)

        self.q = QNetwork().to(self.device)
        self.q_target = QNetwork().to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.replay = ReplayBuffer(capacity=self.cfg.replay_capacity, seed=seed)

        self.train_steps = 0  # number of gradient updates

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy action selection:
          with prob epsilon -> random action
          else -> argmax Q(s)
        """
        if self.rng.random() < epsilon:
            return self.rng.choice([0, 1, 2])

        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)  # (1,11)
        with torch.no_grad():
            q_values = self.q(state_t)  # (1,3)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.replay.push(s, a, r, s2, done)

    def can_train(self) -> bool:
        return len(self.replay) >= max(self.cfg.start_learning_after, self.cfg.batch_size)

    def train_step(self) -> Optional[float]:
        """
        Perform ONE gradient step from a minibatch.
        Returns the scalar loss, or None if not enough data yet.
        """
        if not self.can_train():
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size)

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)              # (B,11)
        actions_t = torch.from_numpy(actions).to(self.device)            # (B,)
        rewards_t = torch.from_numpy(rewards).to(self.device)            # (B,)
        next_states_t = torch.from_numpy(next_states).to(self.device)    # (B,11)
        dones_t = torch.from_numpy(dones).to(self.device)                # (B,)

        # Q(s,a)
        q_sa = self.q(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Target: r + gamma * max_a' Q_target(s',a') * (1-done)
        with torch.no_grad():
            q_next = self.q_target(next_states_t).max(dim=1).values  # (B,)
            target = rewards_t + self.cfg.gamma * q_next * (1.0 - dones_t)

        loss = nn.MSELoss()(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        # Clip gradients for stability
        if self.cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)

        self.optim.step()

        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        payload = {
            "q_state_dict": self.q.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        if map_location is None:
            map_location = str(self.device)
        payload = torch.load(path, map_location=map_location)

        self.q.load_state_dict(payload["q_state_dict"])
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()


# Quick self-test
if __name__ == "__main__":
    agent = DQNAgent(seed=0)
    dummy_s = np.zeros(11, dtype=np.float32)
    dummy_s2 = np.ones(11, dtype=np.float32)

    # Fill replay with random transitions
    for _ in range(1200):
        a = np.random.randint(0, 3)
        r = float(np.random.randn())
        done = bool(np.random.rand() < 0.1)
        agent.store(dummy_s, a, r, dummy_s2, done)

    # Train a few steps
    for i in range(5):
        loss = agent.train_step()
        print("loss:", loss)
