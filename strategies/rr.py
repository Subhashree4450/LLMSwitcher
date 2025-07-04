from core.config import *

class RoundRobinSelector:
    def __init__(self):
        # Only include generative models
        self.queue = [PHI, LLAMA, GEMMA]
        self.index = 0

    def get_next_model(self):
        model_id = self.queue[self.index]
        self.index = (self.index + 1) % len(self.queue)
        return model_id
