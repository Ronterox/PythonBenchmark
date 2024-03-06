import pygame
from random import randint

pygame.init()
run = True

RESOLUTION = 2
FPS = 60

WIDTH, HEIGHT = 320 * RESOLUTION, 240 * RESOLUTION
RES_FACTOR = WIDTH // 320

BLOCK_SIZE = 20 * RES_FACTOR
INITIAL_SPEED = 10
INITIAL_DIRECTION = [0, -1]


def get_random_coords():
    randx, randy = randint(
        0, WIDTH - BLOCK_SIZE), randint(0, HEIGHT - BLOCK_SIZE)
    randcolor = (randint(0, 128), randint(0, 255), randint(0, 255))
    return randx, randy, randcolor


font = pygame.font.SysFont('Arial', 20 * RES_FACTOR)
screen = pygame.display.set_mode((WIDTH, HEIGHT))

x, y = WIDTH // 2, HEIGHT // 2
head, rect_color = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE), (255, 0, 0)
direction, spd = INITIAL_DIRECTION, INITIAL_SPEED

fruitx, fruity, fruitcolor = get_random_coords()
score = 0

tails = []
clock = pygame.time.Clock()
while run:
    dt = clock.tick(FPS) * 0.001
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False
            elif event.key == pygame.K_UP:
                direction = [0, -1]
            elif event.key == pygame.K_DOWN:
                direction = [0, 1]
            elif event.key == pygame.K_LEFT:
                direction = [-1, 0]
            elif event.key == pygame.K_RIGHT:
                direction = [1, 0]

    fruit = pygame.Rect(fruitx, fruity, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, fruitcolor, fruit)

    pygame.draw.rect(screen, rect_color, head)
    movex = round(spd * direction[0] * BLOCK_SIZE * dt)
    movey = round(spd * direction[1] * BLOCK_SIZE * dt)
    head.move_ip(movex, movey)

    x, y = head.x, head.y
    if x < 0 or x + BLOCK_SIZE > WIDTH:
        head.x -= movex
    if y < 0 or y + BLOCK_SIZE > HEIGHT:
        head.y -= movey

    if head.colliderect(fruit):
        fruitx, fruity, fruitcolor = get_random_coords()
        spd += 1
        # spd = 0
        score += 1

        tail = pygame.Rect(x - direction[0] * BLOCK_SIZE,
                           y - direction[1] * BLOCK_SIZE,
                           BLOCK_SIZE, BLOCK_SIZE)
        tails.append(tail)

    text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text, (10, 10))
    text = font.render(f"x: {x}, y: {y}", True, (255, 255, 255))
    screen.blit(text, (10, 50))

    pygame.display.update()

pygame.quit()
