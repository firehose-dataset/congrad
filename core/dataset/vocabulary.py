import os
from nltk.tokenize import TweetTokenizer
from collections import Counter, OrderedDict

import torch

import sentencepiece as spm
from tqdm import tqdm

TOKENIZER = TweetTokenizer()
SPIECE_UNDERLINE = '▁'

def _tokenize(text):
    return TOKENIZER.tokenize(text)

def _is_start_piece(piece):
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  if (piece.startswith("▁") or piece.startswith("<")
      or piece in special_pieces):
    return True
  else:
    return False

# Adapted from [XL-Net](https://github.com/zihangdai/xlnet/blob/master/prepro_utils.py#L68)
def encode_pieces(sp_model, text, sample=False):
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    new_wordends = []
    word_cnt = 0
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

        if _is_start_piece(new_pieces[-1]):
            new_wordends.append(1)
        else:
            new_wordends.append(0)

    return new_pieces, new_wordends

class Vocab(object):
    def __init__(self, vocab_file, cased=False, small=False, prepend_bos=False):
        self.cased = cased
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        if prepend_bos:
          self.sp_model.SetEncodeExtraOptions("bos")

    def encode_as_symbols(self, sent, sample=False):
        if not self.cased: sent = sent.lower()
        return encode_pieces(self.sp_model, sent, sample)

    def encode_as_ids(self, sent, sample=False):
        sentpiece, wordend = self.encode_as_symbols(sent, sample)
        return self.get_indices(sentpiece), wordend

    def encode_sents_as_ids(self, sents, verbose=False):
        sentpieces = self.encode_sents_as_symbols(sents, verbose)
        encoded = []
        encoded_wordends = []
        for sentpiece, wordend in sentpieces:
            encoded.append(self.get_indices(sentpiece))
            encoded_wordends.append(wordend)

        return encoded, encoded_wordends

    def encode_sents_as_symbols(self, sents, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for sent in sents:
            encoded.append(self.encode_as_symbols(sent))

        return encoded

    def load_file(self, filepath, verbose=True):
        input_data = torch.load(filepath)
        iterator = tqdm(input_data, ncols=64) if verbose else iter(input_data)

        data = []
        users = []
        for data_item in iterator:
            data.append(data_item[-1])
            users.append(data_item[0])

        return data, users

    def encode_file(self, filepath, verbose=True):
        input_data = torch.load(filepath)
        iterator = tqdm(input_data, ncols=64) if verbose else iter(input_data)

        data = []
        users = []
        for data_item in iterator:
            sent = data_item[-1]
            data.append(self.encode_as_ids(sent))
            users.append(data_item[0])

        return data, users

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.sp_model.IdToPiece(idx)

    def get_idx(self, sym):
        return self.sp_model.PieceToId(sym)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, ids):
        return torch.LongTensor(ids)

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            pieces = [self.get_sym(idx) for idx in indices]
        else:
            pieces = [self.get_sym(idx) for idx in indices if idx not in exclude]

        return self.sp_model.DecodePieces(pieces)

    def decode_pieces(self, *args):
        return self.sp_model.DecodePieces(*args)

    def __len__(self):
        return len(self.sp_model)

if __name__ == '__main__':
    tweet_data = [
    'If i get my server running again, i think ill make a php script for this,perhaps twitter everything people put on my site :)',
    'i forgot I signed up \n editing video in the Nicholls State University TV lab. I have on tape, footage of myself interviewing Ozzy Osbourne\'s guitarist Zakk Wylde.',
    'Fixed a bug with the At-Large states (AK,DE,MT,ND,SD,VT,WY) for people.reps.getRepFromDistrict.php. Will now accept &district=At-Large',
    'Happy we could help: <url>',
    '<mention> DO YER JAWB!',
    'My thumb is starting to feel sore... Can I sue Microsoft and Bungie? lol\n trying to concentrate on police reports but The Office is on tonight and it\'s all I can think about. yessssssssssss!',
    'Wooow, this is amaziiiing!!',
    'I was born in 2000, and this is falsé.',
    'I was born in 92000, and this is falsé.',
    ]
    tweet_data = [ " ".join(TOKENIZER.tokenize(_)) for _ in tweet_data ]
    basic_tokenizer = Vocab('vocab/cikm2010_uncased_16000.model')

    for id, tweet in enumerate(tweet_data):
        print('======================== Tweet #{} ========================'.format(id))
        print('[Original]    {}'.format(tweet.lower()))
        tokens = basic_tokenizer.sp_model.EncodeAsPieces(tweet.lower())
        print('[Standard]    {}'.format(" ".join(tokens)))
        tokens, wordend = basic_tokenizer.encode_as_symbols(tweet, sample=True)
        print('[Tokenized text 1] {}'.format(" ".join(tokens)))
        print(wordend)
        tokens, wordend = basic_tokenizer.encode_as_symbols(tweet, sample=True)
        print('[Tokenized text 2] {}'.format(" ".join(tokens)))
        print(wordend)
        tokens, wordend = basic_tokenizer.encode_as_symbols(tweet, sample=True)
        print('[Tokenized text 3] {}'.format(" ".join(tokens)))
        print(wordend)
