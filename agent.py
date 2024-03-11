import random
import torch

from collections import deque
from abc import ABC, abstractmethod

from models import QModel
from snake import DIRECTIONS, KEYS, Snake, State
from global_types import Action, Memory


class Agent(ABC):
    def __init__(self, env: Snake, act_every: int = 1, enabled: bool = True):
        self.env = env
        self.enabled = enabled
        self.actions = [action for action in Action]
        self.act_every = act_every

    @abstractmethod
    def get_action(self, state: State) -> Action:
        raise NotImplementedError

    def key_to_action(self, key: int | None, state: State) -> Action:
        if key is None:
            return Action.DO_NOTHING

        direction = state.direction
        index = KEYS.index(key)
        next_direction = DIRECTIONS[index]

        if next_direction == direction:
            return Action.DO_NOTHING

        if next_direction == DIRECTIONS[(DIRECTIONS.index(direction) + 1) % 4]:
            return Action.TURN_RIGHT

        return Action.TURN_LEFT

    def get_action_key(self, state: State) -> int | None:
        action = self.get_action(state)
        direction = state.direction

        if action == Action.TURN_RIGHT:
            index = DIRECTIONS.index(direction)
            return KEYS[(index + 1) % len(KEYS)]

        if action == Action.TURN_LEFT:
            index = DIRECTIONS.index(direction)
            return KEYS[index - 1]

        return None


class RandomAgent(Agent):
    def get_action(self, state: State) -> Action:
        return random.choice(self.actions)


class ModelAgent(Agent):
    def __init__(self, env: Snake, model: QModel,  act_every: int = 1, enabled: bool = True):
        super().__init__(env, act_every, enabled)
        self.model = model
        self.memory: deque[Memory] = deque(maxlen=100_000)
        self.epsilon = 1.

    def get_action(self, state: State) -> Action:
        self.state = state

        if random.randint(0, 200) < self.epsilon:
            self.action = random.choice(self.actions)
            return self.action

        stateTensor = self.model.transform_state(state)
        stateAction: torch.Tensor = self.model(stateTensor)
        action_index = torch.argmax(stateAction).item().__int__()

        self.action = self.actions[action_index]
        return self.action
