import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

os.path.dirname(os.path.abspath(__file__))
import rnn_att_model

import numpy as np
import torchtext
from torchtext.data.utils import get_tokenizer

# data = './data/ptb'
model_type = 'LSTM'
emsize = 100
nhid = 128
nlayers = 1
lr = float(20)
clip = 0.25
epochs = 5
batch_size = 5
bptt = 1
dropout = 0.2
tied = False
seed = 1111
log_interval = 10
save = 'model.pt'
att = True
att_width = 3
cuda = True

try:
    att_width < bptt
except KeyError:
    raise ValueError("""attention width should be less than sequence length,
                        att_width < bptt""")

if torch.cuda.is_available():
    if not cuda:
        print("Warning: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(seed)

torch.manual_seed(seed)


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def convert_encode2utf8(file, original_encode, des_encode):
    with open(file, 'rb') as f:
        x = f.read()
        x_decode = x.decode(original_encode, 'ignore')
        x_encode = x_decode.encode(des_encode)
    with open(file, 'wb') as f:
        f.write(x_encode)


def tokenize(path):
    """Tokenizes a text file"""
    assert os.path.exists(path)
    dictionary = Dictionary()
    # Add words to the dictionary
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                dictionary.add_word(word)
    with open(path, 'r', encoding='utf8') as f:
        idss = []
        for line in f:
            words = line.split() + ['<eos>']
            ids = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)
    return ids


convert_encode2utf8('train.txt', 'utf8', 'utf8')
convert_encode2utf8('valid.txt', 'utf8', 'utf8')
convert_encode2utf8('test.txt', 'utf8', 'utf8')
convert_encode2utf8('wordlist.txt', 'utf8', 'utf8')

eval_batch_size = 10
train_txt = tokenize('train.txt')
val_txt = tokenize('valid.txt')
test_txt = tokenize('test.txt')
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)
with open('wordlist.txt', 'rb') as f:
    wordlist = f.readlines()

# ntokens = len(corpus.dictionary)
ntokens = len(wordlist)
model = rnn_att_model.RNNModel(model_type, ntokens, emsize, nhid, nlayers, dropout, tied, att, att_width, cuda)

criterion = nn.CrossEntropyLoss()
if cuda:
    model = model.cuda()


## Training code

def repackage_hidden(h):
    """warps hidden states in new Variables, to detach them from their history"""
    return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
    return data, target


def evaluate(data_source):
    model.eval()
    total_loss = 0
    ntokens = len(wordlist)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, evaluate, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
    return total_loss / len(data_source)


def train():
    model.train()
    total_loss = 0
    data = train_data
    source = data
    ntokens = len(wordlist)
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, data.size(0) - 1, bptt)):
        start_time = time.time()
        data, targets = get_batch(source, i)
        # hidden = repackage_hidden(hidden)
        model.zero_grad()
        ouput, hidden = model(data, hidden)
        loss = criterion(ouput.view(-1, ntokens), targets)
        loss.backward(retain_graph=True)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        for n, p in model.named_parameters():
            p.data.add_(-lr, p.grad.data)
        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | perplexity {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                              elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


lr = lr
best_val_loss = None

try:
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
        lr /= 2.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(save, 'rb') as f:
    model = torch.load(f)
# Run on test data
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test perplexity {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
