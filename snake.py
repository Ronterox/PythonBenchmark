import pygame

pygame.init()
run = True

RESOLUTION = 2
WIDTH, HEIGHT = 320, 240

font = pygame.font.SysFont('Arial', 30)
screen = pygame.display.set_mode((WIDTH * RESOLUTION, HEIGHT * RESOLUTION))

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False

pygame.quit()
