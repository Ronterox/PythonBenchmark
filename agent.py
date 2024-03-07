import random

from enum import Enum
from abc import ABC, abstractmethod
from snake import DIRECTIONS, KEYS, Snake


class Action(Enum):
    DO_NOTHING = [1, 0, 0]
    TURN_RIGHT = [0, 1, 0]
    TURN_LEFT = [0, 0, 1]


class Agent(ABC):
    def __init__(self, env: Snake):
        self.env = env
        self.actions = [action for action in Action]

    @abstractmethod
    def get_action_key(self, state) -> int | None:
        raise NotImplementedError


class RandomAgent(Agent):
    def get_random_action(self, actions: list[Action]) -> int | None:
        action = random.choice(actions)
        direction = self.env.direction

        if action == Action.TURN_RIGHT:
            index = DIRECTIONS.index(direction)
            return KEYS[(index + 1) % len(KEYS)]

        if action == Action.TURN_LEFT:
            index = DIRECTIONS.index(direction)
            return KEYS[index - 1]

        return None

    def get_action_key(self, state) -> int | None:
        return self.get_random_action(self.actions)
