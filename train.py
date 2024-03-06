import random

from enum import Enum
from snake import Snake, DIRECTIONS, KEYS


class Actions(Enum):
    DO_NOTHING = [1, 0, 0]
    TURN_RIGHT = [0, 1, 0]
    TURN_LEFT = [0, 0, 1]


snake = Snake(fps=20, resolution=2)
actions = [action for action in Actions]


def get_random_action() -> int | None:
    action = random.choice(actions)
    direction = snake.direction

    if action == Actions.TURN_RIGHT:
        index = DIRECTIONS.index(direction)
        return KEYS[(index + 1) % len(KEYS)]

    if action == Actions.TURN_LEFT:
        index = DIRECTIONS.index(direction)
        return KEYS[index - 1]

    return None


scores = []
for _ in range(10_000):
    snake.reset()

    i = 0
    while snake.run:
        key = None
        if i % 5 == 0:
            key = get_random_action()

        snake.check_events(key)
        snake.update()
        snake.clock.tick(snake.fps)
        i += 1

    scores.append(snake.score)

print(f'Average score: {sum(scores) / len(scores)}')
print(f'Total Score: {sum(scores)}')
print(f'Max score: {max(scores)}')

snake.finish()
