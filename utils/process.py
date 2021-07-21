import sys
import os
import gzip
import torch
import pickle
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from env import env

src_tokenizer = AutoTokenizer.from_pretrained(env.BERT_TOKENIZER)
# src_tokenizer = KoBERTTokenizer.from_pretrained(env.BERT_TOKENIZER, sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

def load_annotation(fpath):
    if env.LANGUAGE == 'ge':
        with gzip.open(fpath, 'rb') as f:
            _file = pickle.load(f)
        return _file
    elif env.LANGUAGE == 'ko':
        with open(fpath, 'r') as f:
            lines = f.readlines()
            _file = []
            for line in lines:
                line = line.replace("\n","").strip()
                _file.append(line)
        return _file

def tokenize_BERT(txt_input):
    # return src_tokenizer.batch_encode_plus(txt_input, return_tensors = 'pt', padding=True).to(env.DEVICE)
    return src_tokenizer(txt_input, return_tensors='pt', padding = True).to(env.DEVICE)

def tokenize_Gl(txt_input):
    return txt_input.strip().split()

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([env.BOS_IDX]),torch.tensor(token_ids),torch.tensor([env.EOS_IDX])))

def yield_tokens(gloss):
    for i in gloss:
        yield i.strip().split()

def build_vocab():

    anno = []
    if env.LANGUAGE == 'ge':
        for i in env.data_set:
            anno = anno + [j['gloss'] for j in load_annotation(env.dir + i)]
    elif env.LANGUAGE == 'ko':
        for i in env.data_set:
            anno = anno + load_annotation(env.dir.format(i) + env.dir_type[1])

    vocab_g = build_vocab_from_iterator(yield_tokens(anno), specials=env.special_symbols, special_first=True)
    vocab_g.set_default_index(env.UNK_IDX)
    print(vocab_g['<pad>'])
    print(vocab_g['학교'])
    return vocab_g

vocab_g = build_vocab()
print(len(vocab_g))

def make_tensor(txt_input):
    result = []
    for t in txt_input:
        result.append(vocab_g[t])
    return torch.tensor(result)


SRC_TRANSFORM = [tokenize_BERT]
TGT_TRANSFORM = [tokenize_Gl, make_tensor, tensor_transform]

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            # print(txt_input)
            txt_input = transform(txt_input)
        return txt_input
    return func


def generate_square_subseqeunt_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=env.DEVICE))).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subseqeunt_mask(tgt_seq_len)

    tgt_padding_mask = (tgt == env.PAD_IDX).transpose(0,1)
    return tgt_mask, tgt_padding_mask


def collate_fn(batch):
    src_batch, tgt_batch = [],[]
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(sequential_transforms(tokenize_Gl, make_tensor, tensor_transform)(tgt_sample))
    src_batch = sequential_transforms(tokenize_BERT)(src_batch)
    tgt_batch = pad_sequence(tgt_batch, padding_value=env.PAD_IDX)
    return src_batch, tgt_batch