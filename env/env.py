import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
# dir = "./data/phoenix14t.pose."
# data_set = ['dev','test','train']
data_set = ['val','test','train']
dir = "./data_KO/dataset_whoru.no_duplicates.{}."
dir_type = ['ko','ksl','raw.ko']
# BERT = "dbmdz/bert-base-german-uncased"
BERT = "klue/bert-base"
BERT_TOKENIZER = "klue/bert-base"
# BERT_TOKENIZER = "dbmdz/bert-base-german-uncased"
# BERT = 'skt/kobert-base-v1'
# BERT_TOKENIZER = 'skt/kobert-base-v1'
EMB_SZ = 768
# LANGUAGE = 'ge'
LANGUAGE = 'ko'
weight = [(1.0,0,0,0),(0.5,0.5,0,0),(0.33,0.33,0.33,0),(0.25,0.25,0.25,0.25)]