# Snake Game Training Program

This program trains a snake model using reinforcement learning. It provides options for setting up the training environment and monitoring the training progress through reward plotting.

## Usage

To use this program, follow the instructions below:

### Prerequisites

- Python 3.x
- Required libraries: `argparse`, `signal`, `time`, `os`
- Custom modules: `snake`, `plotting`, `agent`, `models`, `global_types`

### Running the Program

Run the following command in your terminal:

```bash
python train_snake.py [--model MODEL] [--games GAMES] [--fps FPS] [--headless] [--plot] [--resolution RESOLUTION]
```

- `--model MODEL`: Specify the path to a pre-trained model to continue training from a checkpoint.
- `--games GAMES`: Number of games to play for training. Default is 1000.
- `--fps FPS`: Set the FPS (frames per second) limit. Default is -1 (no limit).
- `--headless`: Run the game without a window (useful for server environments).
- `--plot`: Enable plotting of rewards during training. Default is True.
- `--resolution RESOLUTION`: Set the resolution factor. Default is 1.

## Files

- `train_snake.py`: Main Python script for training the snake model.
- `snake.py`: Module containing the Snake game implementation.
- `plotting.py`: Module for plotting rewards during training.
- `agent.py`: Module defining the agent for the snake game.
- `models.py`: Module containing the definition of the Q-learning model.
- `global_types.py`: Module containing global types used in the program.

## Exiting the Program

You can gracefully exit the program by pressing `Ctrl + C`. The program will save the results before exiting.

## Output

The program outputs the following information during training:

- Average reward per game
- Total rewards accumulated
- Maximum reward obtained and the corresponding game
- Training time in seconds

## Notes

- The program uses Q-learning for training the snake model.
- The agent has priority over the player if both are enabled.
- If training is interrupted, the program saves the trained model.
