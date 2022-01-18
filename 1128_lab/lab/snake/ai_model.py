from game import Model, Direction

class AIModel(Model):
    def __init__(self, logger = None):
        super().__init__(logger)

    def move(self, board, snake_head, food):
        pass

    def reset(self):
        pass
