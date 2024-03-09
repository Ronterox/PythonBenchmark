import signal

from snake import Snake
from plotting import Plot
from agent import ModelAgent
from models import QModel
from global_types import Memory

NUM_GAMES = 10000
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
model = QModel(input_size=8, hidden_size=16, output_size=3).set_env(snake)
agent = AGENT_TYPE(snake, model, AGENT_ACT_EVERY, ENABLE_AGENT)

plot = Plot()
rewards = []
mean_rewards = []
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
            if j % 5 == 0:
                agent.model.learn(agent.memory, batch_size=64, gamma=0.95)

        total_reward += reward
        j += 1

    rewards.append(total_reward)
    mean_rewards.append(sum(rewards) / len(rewards))
    print(f'Game {i + 1}/{NUM_GAMES}: {snake.score}, {j} steps')

    if i % 10 == 0:
        avg_reward = mean_rewards[-100:]
        last_reward = rewards[-100:]
        print(f'Avg reward: {sum(avg_reward) / len(avg_reward)}')
        plot.clean() \
            .title(f'Game {i + 1}/{NUM_GAMES}') \
            .labels('Games', 'Scores') \
            .plot(last_reward) \
            .plot(avg_reward) \
            .text(i, last_reward[-1], f'{last_reward[-1]}') \
            .text(i, avg_reward[-1], f'{avg_reward[-1]}') \
            .pause(0.1)


snake.finish()
print_results()
