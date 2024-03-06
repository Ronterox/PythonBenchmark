from snake import Snake

snake = Snake(fps=20, resolution=2)

while snake.run:
    snake.check_events()
    snake.update()
    snake.clock.tick(snake.fps)

snake.finish()
