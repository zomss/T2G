import gzip, pickle
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab


dir = "/media/sdd1/slp_data/phoenix14t.pose."
data_set = ['dev','test','train']

class Vocab_g(vocab):
    def __init__(self):
        super().__init__()
        self.special_symbols = ['<unk>','<pad>','<bos>','<eos>']


def load_annotation(fpath):
    with gzip.open(fpath, 'rb') as f:
        _file = pickle.load(f)
    return _file

def yield_tokens(gloss):
    for i in gloss:
        yield i.lower().strip().split()

def build_vocab():
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0,1,2,3
    special_symbols = ['<unk>','<pad>','<bos>','<eos>']

    anno = []
    for i in data_set:
        anno = anno + [j['gloss'] for j in load_annotation(dir + i)]

    vocab_g = build_vocab_from_iterator(yield_tokens(anno),min_freq=1, specials=special_symbols, special_first=True)
    vocab_g.set_default_index(UNK_IDX)
    return vocab_g

if __name__ == "__main__":
    vocab = build_vocab()
    print(vocab['<unk>'])