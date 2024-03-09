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
        return torch.tensor([state.headx, state.heady,
                             state.fruitx, state.fruity],
                            dtype=torch.float32)

    def transform_states(self, states: list[State]) -> torch.Tensor:
        return torch.stack([self.transform_state(state) for state in states])

    def learn(self, memory: deque[Memory], batch_size: int = 32, gamma: float = 0.9):
        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        next_states_tensor = self.transform_states(next_states)

        q_values = self.forward(next_states_tensor)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.int32)

        q_targets = rewards_tensor + gamma * \
            torch.max(q_values, dim=1).values * (1 - dones_tensor)

        actions_tensor = torch.stack(
            [torch.tensor(action.value) for action in actions])
        targets = q_targets.unsqueeze(0).transpose(0, 1) * actions_tensor

        states_tensor = self.transform_states(states)
        predictions = self.forward(states_tensor)

        self.optimizer.zero_grad()
        loss = self.loss(predictions, targets)
        loss.backward()

        self.optimizer.step()
