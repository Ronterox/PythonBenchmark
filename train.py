import argparse
import signal
import time
import os

from snake import Snake
from plotting import Plot
from agent import ModelAgent
from models import QModel
from global_types import Memory

parser = argparse.ArgumentParser(description='Train a snake model')
parser.add_argument('--model', type=str, help='Load a model', default="")
parser.add_argument('--games', type=int,
                    help='Number of games to play', default=1000)
parser.add_argument('--fps', type=int, help='FPS limit', default=-1)
parser.add_argument('--headless', action=argparse.BooleanOptionalAction,
                    help='Run the game without a window', default=False)
parser.add_argument('--plot', action=argparse.BooleanOptionalAction,
                    help='Plot the rewards', default=True)
parser.add_argument('--resolution', type=int,
                    help='Resolution factor', default=1)
args = parser.parse_args()

NUM_GAMES = args.games
# -1 for no limit
FPS_LIMIT = args.fps
RESOLUTION = args.resolution

# Agent has priority over the player
ENABLE_AGENT = True
HEADLESS = args.headless
AGENT_TYPE = ModelAgent
LOAD_MODEL_PATH = args.model

BATCH_SIZE = 256
PATIENCE = 0.5

# 0 for no limit
PLOT_REWARDS = args.plot
REWARDS_PLOT_LIMIT = 0


def close():
    snake.finish()

    if IS_TRAINING and not LOAD_MODEL_PATH:
        agent.model.save("model.pth")

    games, total, maximum = len(rewards), sum(rewards), max(rewards)
    print(f'\nAverage reward: {total / games} on {games} games')
    print(f'Total Rewards: {total}')
    print(f'Max rewards: {maximum}, on run {rewards.index(maximum)}\n')
    print(f'Time: {time.time() - time_start:.2f}s')


# Gracefully exit the program, but still save the results
signal.signal(signal.SIGINT, lambda _sig, _: close() or exit())

if HEADLESS:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

snake = Snake(FPS_LIMIT, RESOLUTION)
model = QModel(input_size=11, hidden_size=256, output_size=3).set_env(snake)
agent = AGENT_TYPE(snake, model, ENABLE_AGENT)

IS_TRAINING = agent.model is not None and ENABLE_AGENT

if LOAD_MODEL_PATH:
    agent.model.load(LOAD_MODEL_PATH)
    agent.epsilon = 0.05

plot = Plot()
rewards = mean_rewards = []
times = times_avg = []
time_start = time.time()
for i in range(NUM_GAMES):
    start = time.time()
    reward, state, is_done = snake.reset()

    total_reward = 0
    if not LOAD_MODEL_PATH:
        agent.epsilon = max(0.1, 0.4 - 0.4 * i / NUM_GAMES)

    while snake.run:
        key = None
        if IS_TRAINING:
            key = agent.get_action_key(state)

        snake.check_events(key)
        agent.model.transform_state(state)
        reward, state, is_done = snake.update()

        if IS_TRAINING:
            memory = Memory(agent.state, agent.action, reward, state, is_done)
            agent.model.learn([memory], batch_size=1, gamma=PATIENCE)
            agent.memory.append(memory)

        total_reward += reward
        snake.clock.tick(snake.fps)

    output = f'Game {i + 1}/{NUM_GAMES} - Steps: {snake.steps} - Reward: {total_reward}'
    if IS_TRAINING:
        agent.model.learn(agent.memory, batch_size=BATCH_SIZE, gamma=PATIENCE)
        output += f' - Epsilon: {agent.epsilon:.3f}'

    rewards.append(total_reward)
    mean_rewards.append(sum(rewards) / len(rewards))

    times.append(time.time() - start)
    times = times[-10:]
    time_avg = sum(times) / len(times)
    times_avg.append(time_avg)

    eta = sum(times_avg) / len(times_avg) * (NUM_GAMES - i)
    print(f"ETA: {eta:.2f}s ({int(eta // 60)}m {int(eta % 60)}s)")
    print(output)

    if i % 10 == 0 and PLOT_REWARDS:
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
close()
