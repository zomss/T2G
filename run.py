import torch
from torch.utils.data import DataLoader
from utils import process
from env import env
from timeit import default_timer as timer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from model.BERT_T2G import BERT_T2G
from utils import process
from env import env
from tqdm import tqdm

def greedy_decode(model, src, max_len):
    memory = model.encode(src)
    memory_tr = torch.transpose(memory.last_hidden_state, 0, 1)
    ys = torch.ones(1, 1).fill_(env.BOS_IDX).type(torch.long).to(env.DEVICE)
    prev_word_1 = -1
    prev_word_2 = -1
    for i in range(max_len - 1):
        tgt_mask = (process.generate_square_subseqeunt_mask(ys.size(0)).type(torch.bool)).to(env.DEVICE)
        out = model.decode(memory_tr, ys, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        next_word_candidate = torch.argsort(prob, dim=1, descending=True)
        # print(next_word_candidate)
        if prev_word_1 != int(next_word_candidate[:, 0]) and prev_word_2 != int(next_word_candidate[:, 0]):
            next_word = next_word_candidate[:, 0]
        elif prev_word_1 != int(next_word_candidate[:, 1]) and prev_word_2 != int(next_word_candidate[:, 1]):
            next_word = next_word_candidate[:, 1]
        else:
            next_word = next_word_candidate[:, 2]
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).to(env.DEVICE).type_as(torch.cuda.IntTensor()).fill_(next_word)], dim=0)
        if next_word == env.EOS_IDX:
            break
        prev_word_1 = prev_word_2
        prev_word_2 = int(next_word)
    return ys

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    if env.LANGUAGE == 'ge':
        train_iter = [(j['text'],j['gloss']) for j in process.load_annotation(env.dir + 'train')]
    elif env.LANGUAGE == 'ko':
        ko = process.load_annotation(env.dir.format('train') + env.dir_type[2])
        ksl = process.load_annotation(env.dir.format('train') + env.dir_type[1])
        train_iter = [(ko[j], ksl[j]) for j in range(len(ko))]
    else:
        exit()
    train_dataloader = DataLoader(train_iter, batch_size = 16, collate_fn=process.collate_fn)
    for src, tgt in tqdm(train_dataloader, desc = 'train_time'):
        tgt = tgt.to(env.DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:,:]
        tgt_mask, tgt_padding_mask = process.create_mask(src['input_ids'], tgt_input)

        logits = model(src, tgt_input, tgt_mask, tgt_padding_mask)

        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
        # break

    return losses / len(train_dataloader)

def evaluate(model):
    model.eval()
    losses = 0
    if env.LANGUAGE == 'ge':
        val_iter = [(j['text'],j['gloss']) for j in process.load_annotation(env.dir + 'val')]
    elif env.LANGUAGE == 'ko':
        ko = process.load_annotation(env.dir.format('val') + env.dir_type[2])
        ksl = process.load_annotation(env.dir.format('val') + env.dir_type[1])
        val_iter = [(ko[j], ksl[j]) for j in range(len(ko))]
    else:
        exit()
    dev_dataloader = DataLoader(val_iter, batch_size = 16, collate_fn=process.collate_fn)
    for src, tgt in tqdm(dev_dataloader, desc = 'val time'):
        tgt = tgt.to(env.DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:,:]
        tgt_mask, tgt_padding_mask = process.create_mask(src['input_ids'], tgt_input)

        logits = model(src, tgt_input, tgt_mask, tgt_padding_mask)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    bleu_score = [0,0,0,0]
    smoothie = SmoothingFunction().method5
    for (src, tgt) in val_iter:
        src_em = process.sequential_transforms(process.tokenize_BERT)([src])
        tgt_tokens = greedy_decode(model, src_em, 20)
        out = " ".join(process.vocab_g.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        # print(out)
        bleu_score = [bleu_score[i] + sentence_bleu([tgt.split(" ")], out.split(" ")[1:-1], env.weight[i], smoothing_function=smoothie) for i in range(0,4)]
        # if sentence_bleu([tgt.split(" ")], out.split(" ")[1:-1], env.weight[3], smoothing_function=smoothie) > 0:
        #     print(out.split(" ")[1:-1])
        #     print(tgt.split(" "))
    bleu_score = [v / len(val_iter) for v in bleu_score]
    return losses / len(dev_dataloader), bleu_score

def test_(model):
    model.eval()
    losses = 0
    if env.LANGUAGE == 'ge':
        val_iter = [(j['text'],j['gloss']) for j in process.load_annotation(env.dir + 'val')]
    elif env.LANGUAGE == 'ko':
        ko = process.load_annotation(env.dir.format('test') + env.dir_type[2])
        ksl = process.load_annotation(env.dir.format('test') + env.dir_type[1])
        val_iter = [(ko[j], ksl[j]) for j in range(len(ko))]
    else:
        exit()
    dev_dataloader = DataLoader(val_iter, batch_size = 16, collate_fn=process.collate_fn)
    for src, tgt in tqdm(dev_dataloader, desc='test_time'):
        tgt = tgt.to(env.DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:,:]
        tgt_mask, tgt_padding_mask = process.create_mask(src['input_ids'], tgt_input)

        logits = model(src, tgt_input, tgt_mask, tgt_padding_mask)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    bleu_score = [0,0,0,0]
    smoothie = SmoothingFunction().method5
    for (src, tgt) in tqdm(val_iter, desc = 'test_time'):
        # print(src)
        src_em = process.sequential_transforms(process.tokenize_BERT)(src)
        tgt_tokens = greedy_decode(model, src_em.to(env.DEVICE), 20)
        out = " ".join(process.vocab_g.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        bleu_score = [bleu_score[i] + sentence_bleu([tgt.split(" ")], out.split(" ")[1:-1], env.weight[i], smoothing_function=smoothie) for i in range(0,4)]
        # break
    bleu_score = [v / len(val_iter) for v in bleu_score]

    return losses / len(dev_dataloader), bleu_score

if __name__ == "__main__":
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=env.PAD_IDX)
    model = BERT_T2G(len(process.vocab_g), env.BERT, env.DEVICE, env.EMB_SZ).to(env.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.98), eps=1e-9)

    for epoch in range(1,20):
        start_time = timer()
        train_loss = train_epoch(model, optimizer)
        end_time = timer()
        print((
            f"Train Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        start_time = timer()
        val_loss, bleu_score = evaluate(model)
        end_time = timer()
        print((
            f"Val Epoch: {epoch}, Val loss: {val_loss:.3f}, bleu-1 : {bleu_score[0]:.3f}, bleu-2 : {bleu_score[1]:.3f}, bleu-3 : {bleu_score[2]:.3f}, bleu-4 : {bleu_score[3]:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))

        start_time = timer()
        test_loss, bleu_score = test_(model)
        end_time = timer()
        print((
            f"test Epoch: {epoch}, test loss: {test_loss:.3f}, bleu-1 : {bleu_score[0]:.3f}, bleu-2 : {bleu_score[1]:.3f}, bleu-3 : {bleu_score[2]:.3f}, bleu-4 : {bleu_score[3]:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
