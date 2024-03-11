import pygame
import random

from enum import Enum
from random import randint
from copy import deepcopy

from global_types import State


class Direction(Enum):
    UP = [0, -1]
    RIGHT = [1, 0]
    DOWN = [0, 1]
    LEFT = [-1, 0]


DIRECTIONS = [direction.value for direction in Direction]
KEYS = [pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT]


class Snake:
    def __init__(self, fps, resolution):
        if not pygame.get_init():
            pygame.init()

        self.run = True
        self.fps = fps
        self.resolution = resolution
        self.width, self.height = 320 * resolution, 240 * resolution
        self.res_factor = self.width // 320
        self.block_size = 10 * self.res_factor

        self.font = pygame.font.SysFont('Arial', 20 * self.res_factor)
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.head = pygame.Rect(self.width // 2, self.height // 2,
                                self.block_size, self.block_size)
        self.rect_color = (255, 0, 0)
        self.direction = random.choice(DIRECTIONS)

        dx, dy = self.direction
        blockx, blocky = -dx * self.block_size, -dy * self.block_size

        self.tails = [pygame.Rect(self.head.x + blockx, self.head.y + blocky,
                                  self.block_size, self.block_size),
                      pygame.Rect(self.head.x + blockx * 2, self.head.y + blocky * 2,
                                  self.block_size, self.block_size),]

        fruitx, fruity, self.fruitcolor = self.get_random_fruit()
        self.fruit = pygame.Rect(
            fruitx, fruity, self.block_size, self.block_size)
        self.score = 0

        self.clock = pygame.time.Clock()

    def reset(self) -> tuple[int, State, bool]:
        self.run = True
        self.head.x, self.head.y = self.width // 2, self.height // 2
        self.direction = random.choice(DIRECTIONS)
        self.tails = self.tails[:2]
        self.score = 0

        fruitx, fruity, self.fruitcolor = self.get_random_fruit()
        self.fruit.x, self.fruit.y = fruitx, fruity
        state = State(self.head.x, self.head.y,
                      self.fruit.x, self.fruit.y,
                      self.direction.copy(), deepcopy(self.tails))
        return 0, state, False

    def get_random_fruit(self) -> tuple[int, int, tuple[int, int, int]]:
        bs = self.block_size
        limitx, limity = self.width - bs, self.height - bs

        randx, randy = randint(0, limitx), randint(0, limity)
        randcolor = (randint(0, 255), randint(0, 255), randint(0, 255))

        return randx, randy, randcolor

    def check_events(self, keydown: int | None = None):
        if keydown is not None:
            pygame.event.post(pygame.event.Event(
                pygame.KEYDOWN, {'key': keydown}))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.run = False
                    continue
                for direction, key in zip(DIRECTIONS, KEYS):
                    if event.key == key:
                        self.direction = direction

    def update(self) -> tuple[int, State, bool]:
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, self.fruitcolor, self.fruit)

        reward = 0
        if self.head.colliderect(self.fruit):
            self.fruit.x, self.fruit.y, self.fruitcolor = self.get_random_fruit()
            self.score += 1
            reward = 10

            last_tail = self.tails[-1]
            tail = pygame.Rect(last_tail.x + self.block_size,
                               last_tail.y + self.block_size,
                               self.block_size, self.block_size)
            self.tails.append(tail)

        tailbefore = self.head.copy()
        self.head.x += self.direction[0] * self.block_size
        self.head.y += self.direction[1] * self.block_size
        pygame.draw.rect(self.screen, self.rect_color, self.head)
        for tail in self.tails:
            tail.x, tail.y, tailbefore = tailbefore.x, tailbefore.y, tail.copy()
            pygame.draw.rect(self.screen, self.rect_color, tail)

        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        x, y = self.head.x, self.head.y
        inside_width = x >= 0 and x + self.block_size <= self.width
        inside_height = y >= 0 and y + self.block_size <= self.height
        not_hit = self.head not in self.tails

        self.run = self.run and inside_width and inside_height and not_hit
        pygame.display.update()

        # Gym Attributes
        is_done = not self.run

        if is_done:
            reward = -10

        state = State(self.head.x, self.head.y,
                      self.fruit.x, self.fruit.y,
                      self.direction.copy(), deepcopy(self.tails))

        return reward, state, is_done

    def finish(self):
        print(f"Your score was: {self.score}")
        pygame.quit()
