import torch
import random

from torch import nn
from collections import deque

from snake import Snake, State
from global_types import Memory


class QModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 0.01):
        super(QModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def set_env(self, env: Snake) -> 'QModel':
        self.block_size = env.block_size
        self.width, self.width = env.width, env.width
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.l1(x))
        out = self.l2(out)
        return out

    def transform_state(self, state: State) -> torch.Tensor:
        hx, hy, fx, fy = state.headx, state.heady, state.fruitx, state.fruity
        bs, w, h = self.block_size, self.width, self.width
        tails = state.tails

        return torch.tensor([
            fx > hx,  # food right
            fx < hx,  # food left
            fy > hy,  # food down
            fy < hy,  # food up

            hx >= w - bs or (hx + bs, hy) in tails,  # wall right or tail right
            hx <= 0 or (hx - bs, hy) in tails,  # wall left or tail left
            hy >= h - bs or (hx, hy + bs) in tails,  # wall down or tail down
            hy <= 0 or (hx, hy - bs) in tails  # wall up or tail up
        ], dtype=torch.float32)

    def transform_states(self, states: list[State]) -> torch.Tensor:
        return torch.stack([self.transform_state(state) for state in states])

    def learn(self, memory: deque[Memory], batch_size: int, gamma: float):
        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        next_states_tensor = self.transform_states(next_states)
        q_values = self.forward(next_states_tensor)

        values, _ = torch.max(q_values, dim=1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.int32)

        q_results = rewards_tensor + gamma * values * (1 - dones_tensor)

        states_tensor = self.transform_states(states)
        predictions = self.forward(states_tensor)

        actions_tensor = torch.tensor([action.value for action in actions])
        action_indexes = torch.argmax(actions_tensor, 1, keepdim=True)

        targets = predictions.clone()
        targets.scatter_(1, action_indexes, q_results.unsqueeze(1))

        self.optimizer.zero_grad()
        loss = self.loss(predictions, targets)
        loss.backward()

        self.optimizer.step()
