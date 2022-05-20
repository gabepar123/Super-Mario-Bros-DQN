from collections import deque
import numpy as np
import random

QUEUE_SIZE = 1_000_000
MAX_BATCH_SIZE = 128



class replay_buffer:
    def __init__(self):
        self.buffer = deque(maxlen=QUEUE_SIZE)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def random_sample(self):
        batch_size = min(MAX_BATCH_SIZE, len(self.buffer))
        return random.sample(self.buffer, batch_size)
