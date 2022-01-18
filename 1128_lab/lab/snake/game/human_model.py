from pygame.locals import K_UP, K_RIGHT, K_DOWN, K_LEFT
from .model import Model
from .constants import Direction

class HumanModel(Model):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.action = None

    def user_input(self, event): 
        if event.key == K_UP:
            self.action = Direction.UP
        elif event.key == K_RIGHT:
            self.action = Direction.RIGHT
        elif event.key == K_DOWN:
            self.action = Direction.DOWN
        elif event.key == K_LEFT:
            self.action = Direction.LEFT

    def move(self, board, snake_head, food):
        t, self.action = self.action, None
        return t
