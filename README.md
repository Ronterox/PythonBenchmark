# Snake Game with Random Agent

> Random readme created by ChatGPT lol (Is all wrong)

## Introduction
This Python script implements a simple Snake game using the `snake` module along with a random agent for controlling the snake's movements. The game is run for a specified number of games, and the results are plotted using the `plotting` module.

## Requirements
- Python 3.x
- Dependencies: `signal`, `snake`, `plotting`, `agent`

## Usage
1. Ensure you have the required dependencies installed.
   ```bash
   pip install signal snake plotting agent
   ```

2. Run the script.
   ```bash
   python snake_game.py
   ```

3. The game will run for the specified number of games, and the results will be printed and plotted.

## Configuration
- `NUM_GAMES`: Number of games to play.
- `FPS_LIMIT`: Frames per second limit. Set to -1 for unlimited.
- `RESOLUTION`: Grid resolution for the game.

## Results
The script prints and plots the results after each game, displaying the average score, total score, and maximum score achieved during the specified number of games.

## Signal Handling
The script gracefully handles the interrupt signal (Ctrl+C) by finishing the current game, printing the results, and then exiting.

## Components
- `snake.py`: Module containing the Snake class.
- `plotting.py`: Module for creating and updating plots.
- `agent.py`: Module containing the RandomAgent class.

## Acknowledgments
- The Snake game logic is implemented in the `Snake` class.
- The agent controlling the snake's movements is a simple `RandomAgent`.
- Results are visualized using the `Plot` class.

Feel free to explore and modify the script to experiment with different agents, game settings, or visualization methods. Enjoy the Snake game!
