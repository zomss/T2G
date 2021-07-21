import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoModel
# from transformers import AutoTokenizer, BertForPretraining
from transformers import BertModel
import math

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size,
                 dropout = 0.1,
                 maxlen = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class BERT_T2G(nn.Module):
    def __init__(self,tgt_vocab_size, BERT, DEVICE, EMB_SZ):
        super(BERT_T2G, self).__init__()
        # self.bert = AutoModel.from_pretrained(BERT).to(DEVICE)
        self.bert = BertModel.from_pretrained(BERT).to(DEVICE)
        decoder_layer = TransformerDecoderLayer(EMB_SZ, 8)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 12)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, EMB_SZ)
        self.positional_encoding = PositionalEncoding(EMB_SZ)
        self.generator = nn.Linear(EMB_SZ, tgt_vocab_size)

    def forward(self, src, trg, tgt_mask,  tgt_padding_mask):
        # print(src.input_ids.shape)
        # print(src.token_type_ids.shape)
        # print(src.attention_mask.shape)
        # src_emb = self.bert(input_ids = torch.tensor(src.input_ids), attention_mask = torch.tensor(src.attention_mask), token_type_ids= torch.tensor(src.token_type_ids))
        src_emb = self.bert(**src)
        # print(src_emb)
        # print(src_emb.last_hidden_state.shape)
        tgt_emb = self.tgt_tok_emb(trg)
        tgt_emb = self.positional_encoding(tgt_emb)
        src_emb_in = torch.transpose(src_emb.last_hidden_state, 0, 1)
        outs = self.transformer_decoder(tgt_emb, src_emb_in, tgt_key_padding_mask = tgt_padding_mask, tgt_mask = tgt_mask)
        return self.generator(outs)

    def encode(self, src):
        # print(src)
        # return self.bert(input_ids = torch.tensor(src.input_ids), attention_mask = torch.tensor(src.attention_mask), token_type_ids= torch.tensor(src.token_type_ids))
        return self.bert(**src)

    def decode(self, memory, tgt, tgt_mask):
        return self.transformer_decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask = tgt_mask)