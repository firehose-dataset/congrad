import numpy as np

from core.memories.base import BaseBuffer

class ReservoirBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._wordpieces = [None] * buffer_size
        self._wordends = [None] * buffer_size
        self._users = [None] * buffer_size

        self._size = 0
        self.buffer_size = buffer_size
        # important for reservoir sampling
        self.example_seen_so_far = 0

    def add_data(self, wordpiece, wordend, user):
        """ This function implements a reservoir update that ensures each data points has the equal probability to be kept in the memory buffer.
        """
        data2rm = None
        if self.example_seen_so_far < self.buffer_size:
            idx = self.example_seen_so_far
            # Fill the buffer when it is not full
            self._wordpieces[idx] = wordpiece
            self._wordends[idx] = wordend
            self._users[idx] = user

            self._size = min(self._size + 1, self.buffer_size)
        else:
            idx = np.random.randint(0, self.example_seen_so_far)
            if idx < self.buffer_size:
                data2rm = (
                    self._wordpieces[idx], 
                    self._wordends[idx],
                    self._users[idx] 
                )
                self._wordpieces[idx] = wordpiece
                self._wordends[idx] = wordend
                self._users[idx] = user
            else:
                data2rm = (
                    wordpiece,
                    wordend,
                    user
                )         

        self.example_seen_so_far += 1

        return data2rm

    def sample_data(self):
        sample_idx = np.random.randint(0, self._size)
        wordpiece = self._wordpieces[sample_idx]
        wordend = self._wordends[sample_idx]
        user = self._users[sample_idx]

        return wordpiece, wordend, user, 1.0, sample_idx

    def __len__(self):
        return self._size
