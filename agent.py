import random

from enum import Enum
from abc import ABC, abstractmethod
from snake import DIRECTIONS, KEYS, Snake


class Action(Enum):
    DO_NOTHING = [1, 0, 0]
    TURN_RIGHT = [0, 1, 0]
    TURN_LEFT = [0, 0, 1]


class Agent(ABC):
    def __init__(self, env: Snake, act_every: int = 1, enabled: bool = True):
        self.env = env
        self.enabled = enabled
        self.actions = [action for action in Action]
        self.act_every = act_every

    @abstractmethod
    def get_action(self, reward, state, is_done) -> Action:
        raise NotImplementedError

    def get_action_key(self, reward, state, is_done) -> int | None:
        if not self.enabled:
            return None

        action = self.get_action(reward, state, is_done)
        direction = self.env.direction

        if action == Action.TURN_RIGHT:
            index = DIRECTIONS.index(direction)
            return KEYS[(index + 1) % len(KEYS)]

        if action == Action.TURN_LEFT:
            index = DIRECTIONS.index(direction)
            return KEYS[index - 1]

        return None


class RandomAgent(Agent):
    def get_action(self, reward, state, is_done) -> Action:
        return random.choice(self.actions)


class ModelAgent(Agent):
    def get_action(self, reward, state, is_done) -> Action:
        # Either random depending on epsilon greedy or from the model
        pass
