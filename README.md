# Snake Reinforcement Learning (DQN with Feature Vectors)

This project implements a Reinforcement Learning (RL) agent that learns to play the classic Snake game using Deep Q-Learning (DQN).  
The agent is trained using a compact feature-vector state representation instead of raw pixels, allowing fast and stable learning on CPU.

The project is fully modular:

- Game logic is separated from rendering
- Training is headless (no graphics)
- Pygame is used only for visualization after training

---

## 1. Project Overview

### Goal

Train an autonomous agent that can:

- Avoid collisions with walls and itself
- Navigate toward food
- Grow and survive as long as possible

### Approach

- Algorithm: Deep Q-Network (DQN)
- State representation: 11-dimensional feature vector
- Action space: Relative actions (straight / left / right)
- Framework: PyTorch
- Rendering: Pygame (for visualization only)

---

## 2. Project Structure

```
snake_rl/
│
├── env.py          # Pure Snake game logic (no rendering)
├── features.py     # Converts game state → 11-dim feature vector
├── dqn_agent.py    # DQN, replay buffer, target network
├── train.py        # Training loop (headless)
├── render.py       # Pygame renderer
├── play.py         # Watch the trained agent play
├── requirements.txt
│
└── models/
    ├── best.pt     # Best-performing model (avg100)
    └── last.pt     # Last trained model
```

---

## 3. Environment Design (env.py)

- Grid-based Snake game
- No rendering (logic only)
- Deterministic and lightweight, ideal for RL training

### Key design choices

- Relative actions instead of absolute directions:

  - 0 = go straight
  - 1 = turn left
  - 2 = turn right  
    This prevents illegal reverse moves and simplifies learning.

- Reward function:

  - +10 for eating food
  - -10 for dying (wall or self collision)
  - -0.01 per step to discourage stalling

- Anti-stall termination:
  - Episode ends if the snake survives too long without eating food

---

## 4. State Representation (features.py)

Instead of using the full grid or pixels, the agent observes an 11-dimensional feature vector:

```
[
  danger_straight,
  danger_left,
  danger_right,
  dir_up,
  dir_down,
  dir_left,
  dir_right,
  food_left,
  food_right,
  food_up,
  food_down
]
```

### Intuition

The agent only needs to know:

- Whether it will collide if it moves next
- Its current direction
- Where the food is relative to its head

This representation:

- Reduces state complexity
- Enables fast CPU training
- Generalizes across grid sizes

---

## 5. Learning Algorithm (dqn_agent.py)

### Deep Q-Learning (DQN)

A neural network approximates the Q-function:

Q(s, a) ≈ expected future reward

Network architecture:

```
11 → 128 → 128 → 3
```

### Key components

- Replay Buffer: stores past transitions to break correlation
- Target Network: stabilizes learning by slowing target updates
- Epsilon-greedy exploration: balances exploration and exploitation

---

## 6. Training Loop (train.py)

Training is performed without rendering for speed.

### Training process

1. Reset environment
2. Extract feature vector
3. Select action using epsilon-greedy policy
4. Step environment
5. Store transition in replay buffer
6. Perform DQN update
7. Log metrics and save models

### Metrics

- score: food eaten in current episode
- avg100: average score over the last 100 episodes (learning stability indicator)

### Model saving

- last.pt: most recent model
- best.pt: model with highest avg100

---

## 7. Visualization (render.py + play.py)

After training, the agent can be observed playing the game using pygame.

- render.py: handles drawing only
- play.py: loads trained model and runs greedy policy (epsilon = 0)

Controls:

- Close window to exit
- Press R to reset the episode

---

## 8. Setup Instructions

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the agent:

```bash
python train.py
```

Watch the trained agent:

```bash
python play.py
```

---

## 9. Performance Notes

- Training is fast and CPU-friendly
- ~300 episodes (~1 minute) already produce a strong policy
- Longer training improves consistency
- GPU is not required due to small network size

---

## 10. Possible Extensions

- Double DQN for improved stability
- Distance-based reward shaping
- Larger grid sizes
- Pixel-based observations (CNN)
- Human vs agent mode
- Gameplay recording to video

---

## Dem

https://github.com/user-attachments/assets/e6bebccc-432c-4e1b-9539-c943f039e5e3

o
