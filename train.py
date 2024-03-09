import signal

from snake import Snake
from plotting import Plot
from agent import ModelAgent
from models import QModel
from global_types import Action, Memory, State

NUM_GAMES = 1000
FPS_LIMIT = -1
RESOLUTION = 2

ENABLE_AGENT = True
AGENT_ACT_EVERY = 2
AGENT_TYPE = ModelAgent


def print_results():
    games, total, maximum = len(rewards), sum(rewards), max(rewards)
    print(f'\nAverage reward: {total / games} on {games} games')
    print(f'Total Rewards: {total}')
    print(f'Max rewards: {maximum}, on run {rewards.index(maximum)}\n')


def signal_handler(__, _):
    snake.finish()
    print_results()
    exit()


# Gracefully exit the program, but still save the results
signal.signal(signal.SIGINT, signal_handler)

snake = Snake(FPS_LIMIT, RESOLUTION)
model = QModel(input_size=len(State._fields) - 1,
               hidden_size=8,
               output_size=Action.__len__())
agent = AGENT_TYPE(snake, model, AGENT_ACT_EVERY, ENABLE_AGENT)

plot = Plot()
rewards = []
for i in range(NUM_GAMES):
    reward, state, is_done = snake.reset()

    j = 0
    total_reward = 0
    while snake.run and j < len(snake.tails) * 50:
        key = None
        if i % agent.act_every == 0:
            key = agent.get_action_key(state)

        snake.check_events(key)
        reward, state, is_done = snake.update()
        snake.clock.tick(snake.fps)

        if agent.model is not None:
            agent.memory.append(
                Memory(agent.state, agent.action, reward, state, is_done))
            agent.model.learn(agent.memory)

        total_reward += reward
        j += 1

    rewards.append(total_reward)
    print(f'Game {i + 1}/{NUM_GAMES}: {snake.score}, {j} steps')

    if i % 10 == 0:
        avg_reward = sum(rewards) / len(rewards)
        print(f'Avg reward: {avg_reward}')
        plot.clean() \
            .title(f'Game {i + 1}/{NUM_GAMES}') \
            .labels('Games', 'Scores') \
            .plot(rewards)\
            .plot(avg_reward) \
            .text(i, snake.score, f'{snake.score}')\
            .text(i, int(avg_reward), f'{avg_reward}')\
            .pause(0.1)


snake.finish()
print_results()
