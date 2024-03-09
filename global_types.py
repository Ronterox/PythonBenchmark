from collections import namedtuple
from enum import Enum


class Action(Enum):
    DO_NOTHING = [1, 0, 0]
    TURN_RIGHT = [0, 1, 0]
    TURN_LEFT = [0, 0, 1]


State = namedtuple(
    'State', ['headx', 'heady', 'fruitx', 'fruity', 'direction', 'tails'])

Memory = namedtuple(
    'Memory', ['state', 'action', 'reward',  'next_state', 'done'])
