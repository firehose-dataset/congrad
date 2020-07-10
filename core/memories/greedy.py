import copy
import numpy as np
from collections import deque

from core.memories.base import BaseBuffer

class GreedyBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._wordpieces = []
        self._wordends = []
        self._users = []
        self._next_idx = 0

        self.buffer_size = buffer_size
        self.buffer_filled = 0

    def add_data(self, wordpiece, wordend, user):
        data2rm = None
        if self._next_idx >= self.buffer_filled:
            self._wordpieces.append(wordpiece)
            self._wordends.append(wordend)
            self._users.append(user)

            self.buffer_filled += 1
        else:
            data2rm = (
                self._wordpieces[self._next_idx],
                self._wordends[self._next_idx],
                self._users[self._next_idx]
            )
            self._wordpieces[self._next_idx] = wordpiece
            self._wordends[self._next_idx] = wordend
            self._users[self._next_idx] = user

        # add data to replay buffer
        self._next_idx = (self._next_idx + 1) % self.buffer_size

        return data2rm

    def sample_data(self):
        sample_idx = np.random.randint(0, self.buffer_filled)
        wordpiece = self._wordpieces[sample_idx]
        wordend = self._wordends[sample_idx]
        user = self._users[sample_idx]

        return wordpiece, wordend, user, 1.0, sample_idx

    def __len__(self):
        return len(self._users)


if __name__ == '__main__':

    with launch_ipdb_on_exception():
        print('Test buffer size increase...')
        buffer = GreedyBuffer(10)
        for _ in range(15):
            buffer.add_data(_, _)
        print('Before adjustment to the buffer size.')
        print(buffer._data)

        buffer.adjust_size(12)
        print('After adjustment to the buffer size.')
        print(buffer._data)

        for _ in range(1):
            buffer.add_data(_, _)

        print('After new data inserction to the buffer.')
        print(buffer._data)

        print('Test buffer size reduction...')
        buffer = GreedyBuffer(10)
        for _ in range(15):
            buffer.add_data(_, _)
        print('Before adjustment to the buffer size.')
        print(buffer._data)

        buffer.adjust_size(7)
        print('After adjustment to the buffer size.')
        print(buffer._data)

        for _ in range(1):
            buffer.add_data(_, _)

        print('After new data inserction to the buffer.')
        print(buffer._data)
