import pygame
from random import randint

pygame.init()
run = True

RESOLUTION = 2
FPS = 20

WIDTH, HEIGHT = 320 * RESOLUTION, 240 * RESOLUTION
RES_FACTOR = WIDTH // 320

BLOCK_SIZE = 10 * RES_FACTOR
INITIAL_SPEED = 1
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
direction = INITIAL_DIRECTION

fruitx, fruity, fruitcolor = get_random_coords()
score = 0

tails = [pygame.Rect(x, y + BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
         pygame.Rect(x, y + 2 * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)]
clock = pygame.time.Clock()
while run:
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

    screen.fill((0, 0, 0))

    fruit = pygame.Rect(fruitx, fruity, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, fruitcolor, fruit)

    if head.colliderect(fruit):
        fruitx, fruity, fruitcolor = get_random_coords()
        score += 1

        last_tail = tails[-1]
        tail = pygame.Rect(last_tail.x + BLOCK_SIZE,
                           last_tail.y + BLOCK_SIZE,
                           BLOCK_SIZE, BLOCK_SIZE)
        tails.append(tail)

    tailbefore = head.copy()
    head.x += direction[0] * BLOCK_SIZE
    head.y += direction[1] * BLOCK_SIZE
    pygame.draw.rect(screen, rect_color, head)
    for tail in tails:
        tmp = tail.copy()
        tail.x, tail.y = tailbefore.x, tailbefore.y
        pygame.draw.rect(screen, rect_color, tail)
        tailbefore = tmp

    x, y = head.x, head.y
    inside_width = x >= 0 and x + BLOCK_SIZE <= WIDTH
    inside_height = y >= 0 and y + BLOCK_SIZE <= HEIGHT
    run = inside_width and inside_height and head not in tails

    text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    clock.tick(FPS)
    pygame.display.update()

pygame.quit()
