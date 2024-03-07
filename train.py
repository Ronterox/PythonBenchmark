import random
import signal

from enum import Enum
from snake import Snake, DIRECTIONS, KEYS
from plotting import Plot


class Actions(Enum):
    DO_NOTHING = [1, 0, 0]
    TURN_RIGHT = [0, 1, 0]
    TURN_LEFT = [0, 0, 1]


NUM_GAMES = 1000
snake = Snake(fps=-1, resolution=2)
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


def print_results():
    games, total = len(scores), sum(scores)
    print(f'\nAverage score: {total / games} on {games} games')
    print(f'Total Score: {total}')
    print(f'Max score: {max(scores)}\n')


def signal_handler(__, _):
    snake.finish()
    print_results()
    exit()


# Gracefully exit the program, but still save the results
signal.signal(signal.SIGINT, signal_handler)

plot = Plot()
scores = []
for i in range(NUM_GAMES):
    snake.reset()

    j = 0
    while snake.run:
        key = None
        if i % 5 == 0:
            key = get_random_action()

        snake.check_events(key)
        snake.update()
        snake.clock.tick(snake.fps)
        j += 1

    scores.append(snake.score)
    print(f'Game {i + 1}/{NUM_GAMES}: {snake.score}, {j} steps')

    if i % 10 == 0:
        plot.clean() \
            .title(f'Game {i + 1}/{NUM_GAMES}') \
            .labels('Games', 'Scores') \
            .plot(scores)\
            .text(i, snake.score, f'{snake.score}')\
            .pause(0.1)


print_results()
snake.finish()
