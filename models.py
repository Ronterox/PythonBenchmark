import torch
import random

from torch import nn
from collections import deque

from snake import State
from global_types import Memory


class QModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 0.01):
        super(QModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.l1(x))
        out = self.l2(out)
        return out

    def transform_state(self, state: State) -> torch.Tensor:
        hx, hy, fx, fy = state.headx, state.heady, state.fruitx, state.fruity
        return torch.tensor([
            fx > hx,  # food right
            fx < hx,  # food left
            fy > hy,  # food down
            fy < hy,  # food up
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
        action_indexes = torch.argmax(actions_tensor, 1)

        targets = predictions.clone()
        targets[:, action_indexes] = q_results

        self.optimizer.zero_grad()
        loss = self.loss(predictions, targets)
        loss.backward()

        self.optimizer.step()
