import numpy as np
import torch

class LMOnlineIterator(object):
    def __init__(
            self,
            data,
            users,
            device='cpu',
            ext_len=None,
            shuffle=False,
            vocab=None,
            subword_augment=False,
            user_dict=None,
            break_ratio=1.0,
        ):
        """
            data  -- list[LongTensor] -- there is no order among the LongTensors
            users -- list[LongTensor] -- there is no order among the LongTensors
        """
        assert vocab is not None
        self.sents = data
        self.users = users

        num_data = int(len(self.sents) * break_ratio)
        if num_data < len(self.sents):
          orig_sent_num = len(self.sents)
          self.sents = self.sents[:num_data]
          self.users = self.users[:num_data]
          print('[Warning] Only considering {} data (out of {} data).'.format(len(self.sents), orig_sent_num))

        self.user_dict = user_dict

        self.device = device
        self.shuffle = shuffle

        self.vocab = vocab
        self.subword_augment = subword_augment

    def __iter__(self):
        sent_stream = self.get_sent_stream()

        for sent, user in sent_stream:
            wordpieces, wordends = self.vocab.encode_as_ids(sent, sample=self.subword_augment)

            data   = torch.LongTensor(wordpieces).view(1, -1)
            target = torch.LongTensor(wordpieces).view(1, -1)

            target = target.fill_(-1)
            target[:-1] = data[1:]

            token_len = data.size(-1)
            word_len  = sum(wordends)

            yield data, target, user, token_len, word_len

    def get_sent_stream(self):
        # index iterator
        num_sents = len(self.sents)
        epoch_indices = np.random.permutation(num_sents) if self.shuffle \
            else np.array(range(num_sents))

        # sentence iterator
        for idx in epoch_indices:
            yield self.sents[idx], self.users[idx]

    def get_data(self, idx):
        return self.sents[idx], self.users[idx]
