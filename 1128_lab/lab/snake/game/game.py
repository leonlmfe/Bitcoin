import pygame
import numpy as np
from pygame.locals import QUIT, KEYDOWN
from .environment import Environment
from .constants import Kolor
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

class Game:
    def __init__(self):
        self.pixel_size = 40
        self.environment = Environment()
        self.navigationbar_height = 30
        self.width = self.environment.shape[0] * self.pixel_size
        self.height = self.navigationbar_height + self.environment.shape[1] * self.pixel_size

    def loop_with_model(self, model, fps = 10, play_turns = None):
        pygame.init()
        self.fps = fps
        self.screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        self.surface = pygame.Surface(self.screen.get_size())
        self.font = pygame.font.SysFont("Arial", int(self.navigationbar_height / 1.3))
        self.navigationbar_rect = pygame.Rect((0, 0), (self.width, self.navigationbar_height))
        self.stats = "( min / mean / max) = ( {}/ {:.2f}/ {})".format(*model.stats())

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                if event.type == KEYDOWN:
                    model.user_input(event)

            pygame.time.Clock().tick(self.fps)

            new_direction = model.move(*self.environment.board_info())
            self.environment.update_snake_direction(new_direction)

            if not self.environment.update():
                model.reset()
                model.log_score(self.environment.score)
                self.stats = "( min / mean / max) = ( {}/ {:.2f}/ {})".format(*model.stats())
                self.reset_environment()

                if play_turns is not None:
                    play_turns -= 1
                    if play_turns < 1:
                        pygame.quit()
                        return

            self.draw_screen()
            pygame.display.flip()

    def draw_screen(self):
        self.draw_navigation_bar()
        self.draw_board()
        self.screen.blit(self.surface, (0, 0))

    def draw_navigation_bar(self):
        pygame.draw.rect(self.surface, Kolor.gray, self.navigationbar_rect)

        score_text = self.font.render(str(self.environment.score), 1, Kolor.green)
        score_text_rect = score_text.get_rect()
        score_text_rect.center = (self.navigationbar_height // 2, self.navigationbar_height // 2)
        self.surface.blit(score_text, score_text_rect)

        stats_text = self.font.render(self.stats, 1, Kolor.black)
        stats_text_rect = stats_text.get_rect()
        stats_text_rect.center = (self.width // 2, self.navigationbar_height // 2)
        self.surface.blit(stats_text, stats_text_rect)

    def draw_board(self):
        for x in range(self.environment.shape[0]):
            for y in range(self.environment.shape[1]):
                rect = pygame.Rect((x * self.pixel_size, y * self.pixel_size + self.navigationbar_height), (self.pixel_size, self.pixel_size))
                if self.environment.board[x, y] == Environment.WALL:
                    pygame.draw.rect(self.surface, Kolor.black, rect)
                elif self.environment.board[x, y] == Environment.FOOD:
                    pygame.draw.rect(self.surface, Kolor.red, rect)
                elif self.environment.board[x, y] == Environment.EMPTY:
                    pygame.draw.rect(self.surface, Kolor.white, rect)
                elif self.environment.board[x, y] == self.environment.snake_length:
                    pygame.draw.rect(self.surface, Kolor.blue, rect)
                else:
                    pygame.draw.rect(self.surface, Kolor.green, rect)

    def reset_environment(self):
        self.environment = Environment()

