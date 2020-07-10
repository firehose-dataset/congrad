# Code adapted from https://github.com/lopuhin/transformer-xl/blob/fb11489ca4c6000573d27d5eaca3a641057c0a6a/pytorch/inference.py#L99
import sys
import math
import functools
import operator
from nltk.tokenize import TweetTokenizer
from typing import List, Tuple

from dataclasses import dataclass, field
from typing import Any
from queue import PriorityQueue

import numpy as np
import torch

import unicodedata
import six
import string
from IPython import embed

from core.constants import constants
from core.dataset.vocabulary import Vocab
from core.models import (
    MemTransformerLM
)

TOKENIZER = TweetTokenizer()
BOS_token = '<s>'
EOS_token = '<eot>'

SUBSTITUTES = [ ("``", '"'), ("''", '"'), ('\x7f', "") ]
def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs
  for src_str, dst_str in SUBSTITUTES:
      outputs = outputs.replace(src_str, dst_str)

  if six.PY2 and isinstance(outputs, str):
    outputs = outputs.decode('utf-8')

  if not keep_accents:
    outputs = unicodedata.normalize('NFKD', outputs)
    outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs

class BeamSearchNode(object):
  def __init__(self, previousNode, tokens, logProb, length,
               alpha = 0.0, beta = 0.0, gamma = 0.0):
    '''
    :param previousNode:
    :param wordId:
    :param logProb:
    :param length:
    '''
    self.prevNode = previousNode
    self.tokens = tokens
    self.logp = logProb
    self.leng = length
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

  def score(self):
    reward = 0
    lp = float(self.beta + self.leng)**self.alpha / float(self.beta + 1)
    return - self.logp / lp + self.gamma * reward

  def __lt__(self, other):
    return self.score() < other.score()

  def __str__(self):
    return "({}, {})".format(self.score(), " ".join(self.tokens))

class ModelWrapper:
    def __init__(
            self,
            model: MemTransformerLM,
            vocab: Vocab,
            cased: bool = False,
        ):
        self.model = model
        self.vocab = vocab
        self.device = next(model.parameters()).device
        self.lower = not cased


    def tokenize(
            self,
            text: str,
        ) -> List[str]:
        tokens = TOKENIZER.tokenize(
                        preprocess_text(text,
                            lower=self.lower,
                            remove_space=True,
                            keep_accents=False,
                        )
                    )
        processed_tokens = []
        for token in tokens:
            if token.lower().startswith('http'):
                processed_tokens.append('<url>')
            elif len(token) > 2 and token.startswith('@'):
                processed_tokens.append('<mention>')
            else:
                processed_tokens.append(
                    token.lower() if self.lower else token
                )
        return_tokens, _ = self.vocab.encode_as_symbols(" ".join(processed_tokens))

        return return_tokens

    def get_logprobs(
            self,
            tokens: List[str],
            user: int = 0,
            softmax_temp: float = 1.0,
        ) -> torch.Tensor:
        """ Return log probabilities for next tokens.
        Shape of returned tensor is len(tokens) x len(self.vocab),
        where the first element contains log probabilities for tokens
        after the first, and last element log probabilities for tokens
        after the last one.
        """
        self.model.eval() # turn model into eval mode
        assert len(tokens) > 0, 'tokens must be non-empty string array'
        xs = self.vocab.convert_to_tensor(self.vocab.get_indices(tokens))
        log_probs = []
        with torch.no_grad():
            mems = tuple()

            batch_dim = 1  # batch size dimension is 1
            user = torch.LongTensor([user]).to(self.device)
            user = user.unsqueeze(batch_dim)

            xs = xs.to(self.device)
            xs = xs.unsqueeze(batch_dim)

            log_probs = self.model.get_logprob(xs, user, *mems, temperature=softmax_temp)
            log_probs = log_probs.squeeze(batch_dim).data.cpu()

        return log_probs

    def sample_text(
            self,
            text: str,
            top_k: int = 100,
            max_text_len: int = 100,
            user: int = 0,
            sampling_strategy: str = 'argmax',
            softmax_temp: float = 1.0,
            beam_width: int = 10, # optional: only for BEAM search
            beam_alpha: float = 0.0,
            beam_beta: float = 0.0,
        ):
        """ An iterator yielding pieces of generated text, resulting text
        can be obtained by joining all of them with an empty string.
        """
        # TODO for longer texts we want to use memory and don't feed all tokens
        tokens = self.tokenize(text)

        if sampling_strategy == 'beamsearch':
          generated = self.beam_decode(
                            tokens,
                            top_k,
                            max_text_len,
                            user,
                            1,
                            softmax_temp,
                            beam_width,
                            beam_alpha,
                            beam_beta,
                        )
        elif sampling_strategy in {'argmax', 'multinomial'}:
          generated = self.greedy_decode(
                            tokens,
                            top_k,
                            max_text_len,
                            user,
                            sampling_strategy
                        )

        return generated

    def beam_decode(
          self,
          tokens: List[str],
          top_k: int = 100,
          max_text_len: int = 100,
          user: int = 0,
          nbest: int = 1,
          softmax_temp: float = 1.0,
          beam_width: int = 10,
          beam_alpha: float = 0.0,
          beam_beta: float = 0.0,
      ):
        endnodes = []

        node = BeamSearchNode(None, tokens, 0, len(tokens), beam_alpha, beam_beta)
        nodes = PriorityQueue()

        nodes.put( node ) # negative ll, node
        qsize = 1

        while True: # decoding loop
          if qsize > 2000:
            print('Give up')
            break # give up

          # fetch the best node
          _node = nodes.get()

          # stop condition
          if _node.tokens[-1] == EOS_token:
            assert _node.prevNode != None
            endnodes.append( _node )

            if len(endnodes) >= nbest:
              break
            else:
              continue

          # forward model
          log_probs = self.get_logprobs(_node.tokens, user, softmax_temp)

          # beam search start
          log_prob, indexes = torch.topk(log_probs[-1, :], beam_width)
          nextnodes = []
          for new_k in range(beam_width):
            decoded_t = indexes[new_k].item()
            logp = log_prob[new_k].item()

            node = BeamSearchNode(
                      _node,
                      _node.tokens + [self.vocab.get_sym(decoded_t)],
                      _node.logp + logp,
                      _node.leng + 1,
                      beam_alpha,
                      beam_beta,
                    )
            nextnodes.append( node )

          # enque nodes
          for i in range(len(nextnodes)):
            nodes.put(nextnodes[i])
          # increase qsize
          qsize += len(nextnodes) - 1

        # print([ str(_) for _ in nodes.queue[:5] ])
        # if nothing found
        if len(endnodes) == 0:
          print('Nothing found...use backoff prediction')
          endnodes.append(nodes.get())

        return endnodes[0].tokens

    def greedy_next(
            self,
            tokens: List[str],
            top_k: int = 100,
            user: int = 0,
            sampling_strategy: str = 'argmax',
             softmax_temp: float = 1.0,
        ) -> str:
        log_probs = self.get_logprobs(tokens, user, softmax_temp)
        log_probs = log_probs[-1, :]
        if sampling_strategy == 'argmax':
            sampled_idx = torch.argsort(log_probs)[-1].item()
        elif sampling_strategy == 'multinomial':
            top_indices = torch.argsort(log_probs)[-top_k:]
            top_probs = log_probs[top_indices].double().exp()
            sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
        else:
            raise ValueError('Unsupported Sampling Method')

        return self.vocab.get_sym(sampled_idx), log_probs[sampled_idx]

    def greedy_decode(
        self,
        tokens: List[str],
        top_k: int = 100,
        max_text_len: int = 100,
        user: int = 0,
        sampling_strategy: str = 'argmax',
        softmax_temp: float = 1.0,
      ):
        complete = False
        while len(tokens) < max_text_len and not complete:
          next_token, logprob = self.greedy_next(tokens, top_k=top_k, user=user, sampling_strategy=sampling_strategy, softmax_temp=softmax_temp)
          tokens.append(next_token)
          if EOS_token in next_token: complete = True

        return tokens
