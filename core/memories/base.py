import torch
import numpy as np

from core.dataset.utils import encap_batch

class BaseBuffer(object):
    def __init__(self, max_seqlen):
        self.max_seqlen = max_seqlen

    def add_data(self, wordpiece, wordend, user):
        raise NotImplementedError('[Fatal Error] add_data(self, wordpiece, wordend, user)'
                                 +' needs to be implemeneted by subtype.')

    def sample_data(self, sampled_idx):
        raise NotImplementedError('[Fatal Error] sample_data(self, wordpiece, wordend, user)'
                                 +' needs to be implemeneted by subtype.')

    def add_batch(self, wordpieces, wordends, users):
        batch2rm = [[], [], []]
        for i, (wordpiece, wordend, user) in enumerate(zip(wordpieces, wordends, users)):
            data2rm = self.add_data(wordpiece, wordend, user)
            if data2rm is not None:
                batch2rm[0].append(data2rm[0])
                batch2rm[1].append(data2rm[1])
                batch2rm[2].append(data2rm[2])

        return batch2rm

    def sample_batch(self, batch_size, device, bptt=None):
        if len(self) == 0: return None

        batch = [ [], [], [] ]
        data_weights = torch.FloatTensor(1, batch_size)
        data_ids = torch.LongTensor(1, batch_size)
        for i in range(batch_size):
            wp, wd, user, data_weight, data_id = self.sample_data()
            batch[0].append(wp)
            batch[1].append(wd)
            batch[2].append(user)
            data_weights[:, i] = data_weight
            data_ids[:, i] = data_id

        bptt = max([ len(_) for _ in batch[0] ]) if bptt is None else bptt
        bptt = min(bptt, self.max_seqlen)
        encoded_batch = encap_batch(batch, bptt, device)
        return encoded_batch + (data_weights.to(device), data_ids)

    def __str__(self):
        return "{}()".format(self.__class__.__name__)
