import os, sys
import glob
import time

import numpy as np
import torch
from collections import defaultdict

from core.iterators import LMOnlineIterator, LMBatchIterator, LMOnePassIterator
from core.dataset.vocabulary import Vocab

#constants
DATA_POSTFIX = defaultdict(lambda x: '')
DATA_POSTFIX.update({'sm': '.small', 'tiny': '.tiny', 'freq': '.freq'})
VALID_DATASET = {
    'Firehose10M',
    'Firehose100M',
    'TwitterPT2013',
}
class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.cased = kwargs.get('cased', False)
        self.vocab = Vocab(*args, prepend_bos=True, **kwargs)
        self.vocab_file = kwargs['vocab_file']

        assert self.dataset in VALID_DATASET
        print('Loading dataset {}...'.format(self.dataset))
        data_tm = time.time()
        infile_template = os.path.join(path,  "{}."+"{}.pt".format(
                                       'cased' if self.cased else 'uncased'))
        self._create_datastruct(self.vocab.load_file, infile_template)
        print('Done. ({:.2f} sec)'.format(time.time() - data_tm))

    def _create_datastruct(self, load_fn, infile_template):
        self.train_data, self.train_users = load_fn(infile_template.format("train"))
        self.val_data, self.val_users     = load_fn(infile_template.format("val"))
        self.test_data, self.test_users   = load_fn(infile_template.format("test"))

        self.user_dict  = { user: uid for uid, user \
                            in enumerate(set(self.train_users + self.val_users + self.test_users)) }
        self.user_idict = { uid: user for user, uid in self.user_dict.items() }

        self.train_users = [ self.user_dict[_] for _ in self.train_users ]
        self.val_users   = [ self.user_dict[_] for _ in self.val_users   ]
        self.test_users  = [ self.user_dict[_] for _ in self.test_users  ]

    def get_iterator(self, split, *args, iterator_name='LMBatchIterator', **kwargs):
        data  = getattr(self, '{}_data'.format(split))
        users = getattr(self, '{}_users'.format(split))

        iterator_class = eval(iterator_name)
        data_iter = iterator_class(
                        data,
                        users,
                        *args,
                        vocab=self.vocab,
                        **kwargs,
                    )

        return data_iter

    @property
    def num_users(self):
        return len(self.user_dict)


def get_lm_corpus(datadir, dataset, vocab_file, cased, **kwargs):
    assert os.path.exists(vocab_file), 'Please compute the vocabulary first.'

    kwargs['cased'] = cased
    kwargs['vocab_file'] = vocab_file
    corpus = Corpus(datadir, dataset, **kwargs)

    return corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='data/Twitter10M_dataset',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='Twitter10M',
                        choices=['Twitter10M', 'Twitter100M'],
                        help='dataset name')
    parser.add_argument('--vocab_file', type=str, default='vocab/Twitter10M_cased_32000.model',
                        help='location of the vocabulary of the corpus')
    parser.add_argument('--cased', default=True, action='store_false',
                        help='whether to use cased or uncased corpus.')

    args = parser.parse_args()

    vocab_file = osp.abspath(args.vocab_file)
    print(vocab_file)
    corpus = get_lm_corpus(args.datadir, args.dataset, vocab_file, args.cased)
    print('Vocab size : {}'.format(len(corpus.vocab)))
    iterator = corpus.get_iterator('train', 512, 280, shuffle=True, iterator_name='LMOnePassIterator')

    from tqdm import tqdm
    with launch_ipdb_on_exception():
        tweet_lens = []
        for i_, (wordpieces, wordends, users) in tqdm(enumerate(iterator)):
            tweet_lens.extend([ len(_) for _ in wordpieces])

        print('Length Avg: {}'.format(np.mean(tweet_lens)))
        print('Length Std: {}'.format(np.std(tweet_lens)))
        import ipdb; ipdb.set_trace()


