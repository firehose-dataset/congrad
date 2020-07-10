import numpy as np
import torch

from core.dataset.utils import encap_batch
from core.iterators import LMOnlineIterator

class LMBatchIterator(object):
    def __init__(self, data, user, bsz, maxlen, *args, break_ratio=1.0, **kwargs):
        self._wrapped_iter = LMOnlineIterator(
                                data,
                                user,
                                *args,
                                break_ratio=break_ratio,
                                **kwargs)

        self.break_ratio = break_ratio
        self.bsz = bsz
        # In Twitter domain, all text is under 280 characters
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
                        wordpieces[i], wordends[i] = \
                            self.vocab.encode_as_ids(sent, sample=self.subword_augment)

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

    @property
    def wrapped_iter(self):
        return self._wrapped_iter

    def __getattr__(self, attr):
        if attr == '_wrapped_iter':
            raise AttributeError()
        return getattr(self._wrapped_iter, attr)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_iter)

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            bptt = min(max([len(wp) for wp in batch[0]]), self.maxlen)
            yield encap_batch(batch, bptt, self.device)

    def __len__(self):
        return len(self.sents) // self.bsz
