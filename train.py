import signal

from snake import Snake
from plotting import Plot
from agent import RandomAgent

NUM_GAMES = 1000
FPS_LIMIT = -1
RESOLUTION = 2


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

snake = Snake(FPS_LIMIT, RESOLUTION)
agent = RandomAgent(snake)

plot = Plot()
scores = []
for i in range(NUM_GAMES):
    snake.reset()

    j = 0
    while snake.run:
        key = None
        if i % 5 == 0:
            key = agent.get_action_key(None)

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

snake.finish()
print_results()
