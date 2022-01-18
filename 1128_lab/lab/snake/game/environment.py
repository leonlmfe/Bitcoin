import numpy as np
from .constants import Direction

class Environment:
    EMPTY   = 0
    WALL    = -1
    FOOD    = -2

    def __init__(self):
        self.shape = (15, 17)
        self.board = np.zeros(self.shape, dtype=int)
        
        # Setup the wall
        self.board[0] = self.board[-1] = Environment.WALL
        self.board[:, 0] = self.board[:, -1] = Environment.WALL

        # Setup the snake
        self.board[7, 9] = 2
        self.board[7, 10] = 1
        self.snake_head = (7, 9)
        self.snake_length = 2
        self.snake_direction = Direction.UP

        # Setup the food
        self.food = (7, 4)
        self.board[7, 4] = Environment.FOOD

        self.score = 0

    def update(self):
        dx, dy = Direction.MOVE[self.snake_direction]
        x, y = self.snake_head
        x, y = x + dx, y + dy

        if self.board[x, y] == Environment.WALL or self.board[x, y] > 1:
            return False
        elif self.board[x, y] == Environment.FOOD:
            self.snake_head = (x, y)
            self.snake_length += 1
            self.board[x, y] = self.snake_length
            self.score += 1
            self._place_food()
        else:
            self.snake_head = (x, y)
            self.board = np.where(self.board > 0, self.board - 1, self.board)
            self.board[x, y] = self.snake_length

        return True

    def board_info(self):
        return (np.array(self.board), self.snake_head, self.food)

    def update_snake_direction(self, direction):
        if direction == Direction.UP or direction == Direction.RIGHT or direction == Direction.DOWN or direction == Direction.LEFT:
            if direction + self.snake_direction != Direction.MASK:
                self.snake_direction = direction

    def _place_food(self):
        x, y = np.nonzero(self.board == Environment.EMPTY)
        ind = np.random.randint(x.size)
        self.food = (x[ind], y[ind])
        self.board[x[ind], y[ind]] = Environment.FOOD

