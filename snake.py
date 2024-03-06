import pygame
from random import randint

pygame.init()
run = True

RESOLUTION = 2
WIDTH, HEIGHT = 320 * RESOLUTION, 240 * RESOLUTION
RES_FACTOR = WIDTH // 320

BLOCK_SIZE = 20 * RES_FACTOR


def get_random_coords():
    randx, randy = randint(
        0, WIDTH - BLOCK_SIZE), randint(0, HEIGHT - BLOCK_SIZE)
    randcolor = (randint(0, 255), randint(0, 255), randint(0, 255))
    return randx, randy, randcolor


font = pygame.font.SysFont('Arial', 20 * RES_FACTOR)
screen = pygame.display.set_mode((WIDTH, HEIGHT))

x, y = WIDTH // 2, HEIGHT // 2
rect, rect_color = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE), (255, 0, 0)
direction, spd = [0, 0], 2

fruitx, fruity, fruitcolor = get_random_coords()

score = 0
clock = pygame.time.Clock()
while run:
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

    pygame.draw.rect(screen, rect_color, rect)
    rect.move_ip(spd * direction[0], spd * direction[1])
    x, y = rect.x, rect.y

    if x < 0 or x + BLOCK_SIZE > WIDTH:
        rect.x -= spd * direction[0]
    if y < 0 or y + BLOCK_SIZE > HEIGHT:
        rect.y -= spd * direction[1]

    if rect.colliderect(fruit):
        fruitx, fruity, fruitcolor = get_random_coords()
        spd += 2
        score += 1

    text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.update()
    clock.tick(60)

pygame.quit()
