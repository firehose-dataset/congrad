import numpy as np
import torch

from core.iterators import LMOnlineIterator
from core.dataset.utils import encap_batch

class LMOnePassIterator(object):
    def __init__(
        self,
        data,
        user,
        bsz,
        maxlen,
        *args,
        **kwargs,
    ):
        self._wrapped_iter = LMOnlineIterator(
                                data,
                                user,
                                *args, **kwargs)
        self.bsz = bsz
        self.maxlen = maxlen
        print("[WARNING] {} truncates text with maximum length of {} tokens (Whilst Max Twitter Size = 280 Character).".format(
                self.__class__.__name__,
                self.maxlen,
            ))

    def stream_iterator(self, sent_stream):
        wordpieces  = [None] * self.bsz
        wordends = [None] * self.bsz
        users = [None] * self.bsz

        batch_size = self.bsz
        while True:
            valid_batch = True
            for i in range(batch_size):
                try:
                    while True:
                        sent, users[i] = next(sent_stream)
                        wordpieces[i], wordends[i] = self.vocab.encode_as_ids(sent, sample=self.subword_augment)

                        # discard tweets that is longer than max length
                        if len(wordpieces[i]) <= self.maxlen: break
                except StopIteration:
                    valid_batch = False

                    # last batch has number of instances smaller than i
                    batch_size = i
                    wordpieces = wordpieces[:batch_size]
                    wordends = wordends[:batch_size]
                    users = users[:batch_size]
                    break

            yield wordpieces, wordends, users

            if not valid_batch: return

    def get_batch(self, batch_id):
        num_batch = len(self)
        assert batch_id < num_batch, '[Fatal error]'
        start_id = batch_id * self.bsz
        end_id   = (batch_id + 1) * self.bsz

        wordpieces  = [None] * self.bsz
        wordends = [None] * self.bsz
        users = [None] * self.bsz
        for i, data_id in enumerate(range(start_id, end_id)):
            sent, users[i] = self.get_data(data_id)
            wordpieces[i], wordends[i] = self.vocab.encode_as_ids(sent, sample=self.subword_augment)

        bptt = min(max([len(wp) for wp in wordpieces]), self.maxlen)

        return encap_batch((wordpieces, wordends, users), bptt, self.device)

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch

    @property
    def wrapped_iter(self):
        return self._wrapped_iter

    def __getattr__(self, attr):
        if attr == '_wrapped_iter':
            raise AttributeError()
        return getattr(self._wrapped_iter, attr)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_iter)

    def __len__(self):
        return len(self.sents) // self.bsz
