import torch
import numpy as np
from collections import OrderedDict

from core.memories.base import BaseBuffer
from core.memories.greedy import GreedyBuffer
from core.memories.reservoir import ReservoirBuffer


class StratifiedBuffer(BaseBuffer):
    """Unbounded user replay buffer"""
    def __init__(
        self,
        buffer_size,
        *args,
        buffer_class='GreedyBuffer',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # lazy initialization
        self.user_buffers = OrderedDict()
        self.user_num = 0
        self.inv_user_dict = {}
        self.buffer_class = eval(buffer_class)
        self.buffer_args = args
        self.buffer_kwargs = kwargs
        self.per_user_buffer_size = buffer_size

    def add_data(self, wordpiece, wordend, user):
        if self.user_buffers.get(user, None) is None:
            self.inv_user_dict[self.user_num] = user
            self.user_buffers[user] = self.buffer_class(self.per_user_buffer_size,
                                                       *self.buffer_args, **self.buffer_kwargs)

            self.user_num += 1
        # Add data to user specific buffer
        return self.user_buffers[user].add_data(wordpiece, wordend, user)

    def sample_data(self):
        user_idx = np.random.randint(0, self.user_num)
        sampled_user = self.inv_user_dict[user_idx]
        return self.sample_user_data(sampled_user)

    def sample_user_data(self, user):
        if self.user_buffers.get(user, None) is not None:
            return self.user_buffers[user].sample_data()
        else:
            raise ValueError('Unencountered User.')

    def __len__(self):
        return sum([ len(buf) for buf in self.user_buffers.values() ])

if __name__ == '__main__':
    import time
    from tqdm import tqdm

    with launch_ipdb_on_exception():
        print('Test buffer size increase...')
        buffer = StratifiedBuffer(5, 50)

        start_tm = time.time()
        for user in tqdm(range(917359)):
            for data in range(5):
                buffer.add_data(list(range(5)), list(range(5)), user)
        print('Add data: {} sec.'.format(time.time() - start_tm))

        device = torch.device('cuda')
        sample_tm = time.time()
        for iter_id in tqdm(range(100)):
            buffer.sample_batch(256, device)
        print('Sample data: {} sec.'.format(time.time() - sample_tm))
