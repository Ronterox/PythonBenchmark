import signal

from snake import Snake
from plotting import Plot
from agent import ModelAgent
from models import QModel
from global_types import Memory

NUM_GAMES = 10000
# -1 for no limit
FPS_LIMIT = -1
RESOLUTION = 2

# Agent has priority over the player
ENABLE_AGENT = True
AGENT_ACT_EVERY = 1
AGENT_TYPE = ModelAgent
LOAD_MODEL_PATH = ""

# 0 for no limit
REWARDS_PLOT_LIMIT = 0


def print_results():
    games, total, maximum = len(rewards), sum(rewards), max(rewards)
    print(f'\nAverage reward: {total / games} on {games} games')
    print(f'Total Rewards: {total}')
    print(f'Max rewards: {maximum}, on run {rewards.index(maximum)}\n')


def signal_handler(__, _):
    snake.finish()
    print_results()
    if IS_TRAINING:
        agent.model.save("model.pth")
    exit()


# Gracefully exit the program, but still save the results
signal.signal(signal.SIGINT, signal_handler)

snake = Snake(FPS_LIMIT, RESOLUTION)
model = QModel(input_size=8, hidden_size=256, output_size=3).set_env(snake)
agent = AGENT_TYPE(snake, model, AGENT_ACT_EVERY, ENABLE_AGENT)

IS_TRAINING = agent.model is not None and ENABLE_AGENT

if LOAD_MODEL_PATH:
    agent.model.load(LOAD_MODEL_PATH)


plot = Plot()
rewards = []
mean_rewards = []
for i in range(NUM_GAMES):
    reward, state, is_done = snake.reset()

    j = 0
    total_reward = 0
    while snake.run and j < len(snake.tails) * 50:
        key = None
        if IS_TRAINING and i % agent.act_every == 0:
            key = agent.get_action_key(state)

        snake.check_events(key)
        reward, state, is_done = snake.update()
        snake.clock.tick(snake.fps)

        if IS_TRAINING:
            agent.memory.append(
                Memory(agent.state, agent.action, reward, state, is_done))
            if (j + 1) % 10 == 0:
                agent.model.learn(agent.memory, batch_size=100, gamma=0.9)

        total_reward += reward
        j += 1

    output = f'Game {i + 1}/{NUM_GAMES} - Steps: {j} - Reward: {total_reward}'
    if IS_TRAINING:
        agent.epsilon = min(0.95, i / NUM_GAMES * 25)
        agent.model.learn(agent.memory, batch_size=1000, gamma=0.9)
        output += f' - Epsilon: {agent.epsilon:.2f}'

    rewards.append(total_reward)
    mean_rewards.append(sum(rewards) / len(rewards))
    print(output)

    if i % 10 == 0:
        avg_reward = mean_rewards[-REWARDS_PLOT_LIMIT:]
        last_reward = rewards[-REWARDS_PLOT_LIMIT:]
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
