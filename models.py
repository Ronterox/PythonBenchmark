import torch
import random

from torch import nn
from collections import deque
from copy import deepcopy

from snake import Direction, Snake, State
from global_types import Memory


class QModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 0.001):
        super(QModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def set_env(self, env: Snake) -> 'QModel':
        self.block_size = env.block_size
        self.width, self.height = env.width, env.height
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.l1(x))
        out = self.l2(out)
        return out

    def transform_state(self, state: State) -> torch.Tensor:
        hx, hy, fx, fy = state.headx, state.heady, state.fruitx, state.fruity
        bs, w, h = self.block_size, self.width, self.height
        tails, dir = deepcopy(state.tails), state.direction

        dangerRight = (hx + bs) >= (w - bs)
        dangerLeft = (hx - bs) <= 0
        dangerDown = (hy + bs) >= (h - bs)
        dangerUp = (hy - bs) <= 0

        hr = hx + bs, hy
        hl = hx - bs, hy
        hd = hx, hy + bs
        hu = hx, hy - bs

        tailbefore = tails[0]
        tailbefore.x, tailbefore.y = hx, hy
        for tail in tails[1:]:
            tail.x, tail.y, tailbefore = tailbefore.x, tailbefore.y, tail.copy()

            pos = tail.x, tail.y
            if pos == hr:
                dangerRight = True
            elif pos == hl:
                dangerLeft = True
            elif pos == hd:
                dangerDown = True
            elif pos == hu:
                dangerUp = True

        dangerStraight = False
        dangerRight = False
        dangerLeft = False

        dir_r = dir == Direction.RIGHT.value
        dir_l = dir == Direction.LEFT.value
        dir_d = dir == Direction.DOWN.value
        dir_u = dir == Direction.UP.value

        if dir_r:
            dangerStraight = dangerRight
            dangerRight = dangerDown
            dangerLeft = dangerUp
        elif dir_l:
            dangerStraight = dangerLeft
            dangerRight = dangerUp
            dangerLeft = dangerDown
        elif dir_d:
            dangerStraight = dangerDown
            dangerRight = dangerLeft
            dangerLeft = dangerRight
        elif dir_u:
            dangerStraight = dangerUp
            dangerRight = dangerRight
            dangerLeft = dangerLeft

        return torch.tensor([
            dangerStraight,
            dangerRight,
            dangerLeft,

            dir_r,  # direction right
            dir_l,  # direction left
            dir_u,  # direction up
            dir_d,  # direction down

            fx > hx,  # food right
            fx < hx,  # food left
            fy > hy,  # food down
            fy < hy,  # food up
        ], dtype=torch.float32)

    def transform_states(self, states: list[State]) -> torch.Tensor:
        return torch.stack([self.transform_state(state) for state in states])

    def learn(self, memory: deque[Memory] | list[Memory], batch_size: int, gamma: float):
        if len(memory) < batch_size:
            batch = memory
        else:
            batch = random.sample(memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = self.transform_states(states)
        actions_tensor = torch.tensor([action.value for action in actions])
        rewards_tensor = torch.tensor(rewards, dtype=torch.int32)
        next_states_tensor = self.transform_states(next_states)
        dones_tensor = torch.tensor(dones, dtype=torch.int32)

        next_q_values = self.forward(next_states_tensor)
        values, _ = torch.max(next_q_values, dim=1)

        q_results = rewards_tensor + gamma * values * (1 - dones_tensor)

        predictions = self.forward(states_tensor)
        action_indexes = torch.argmax(actions_tensor, 1, keepdim=True)

        targets = predictions.clone()
        targets.scatter_(1, action_indexes, q_results.unsqueeze(1))

        self.optimizer.zero_grad()
        loss = self.loss(predictions, targets)
        loss.backward()

        self.optimizer.step()

    def save(self, path: str):
        print(f"Saving model to {path}...")
        torch.save(self.state_dict(), path)
        print("Model saved")

    def load(self, path: str):
        print(f"Loading model from {path}...")
        self.load_state_dict(torch.load(path))
        print("Model loaded")
