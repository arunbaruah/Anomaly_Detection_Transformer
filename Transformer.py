#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math, copy, time
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
WINDOW_SIZE = 10
VOCAB_SIZE = 29

class EncoderDecoder(nn.Module):
#    A standard Encoder-Decoder architecture. Base for this and many other models.
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        #Take in and process masked src and target sequences.
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    
class Generator(nn.Module):
    #Define standard linear + softmax generation step.
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    #Produce N identical layers.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    #Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        #Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    #Construct a layernorm module (See citation for details).
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    #A residual connection followed by a layer norm.
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #Apply residual connection to any sublayer with the same size.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    #Encoder is made up of self-attn and feed forward (defined below)
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    #Generic N layer decoder with masking.
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        emb = self.lut(x) * math.sqrt(self.d_model)
            
        return emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask =                 self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor *             (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
    
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            # l = l.sum()[0] / normalize
            l = l.sum() / normalize
            total += l.data
            # total += l.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        x = out
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, tgt, max_len, start_symbol, pred, g):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    torch.set_printoptions(precision=2)
    
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
       
        predicted = torch.argsort(prob, 1)[0][-g:]
        label = tgt[0][i]   
             
        if label == 0:
            return ys
        _, next_key = torch.max(prob, dim = 1)
        next_key = next_key.data[0]
        
        if pred:
            print("Incoming log:", label) 
            print("Candidate logs:", predicted, "\n")
#             print("Probabilities: ", torch.sort(prob,1)[0][0][-g:])

        if label not in predicted:
            abn = torch.tensor([-1])
            abn = abn.data[0]
            ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(label)], dim=1)
            ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(abn)], dim=1)
            # print(ys[:,1:])
            return ys[:,1:]
        else: 
            ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(label)], dim=1)
            # _, next_key = torch.max(prob, dim = 1)
            # next_key = next_key.data[0]
            # ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_key)], dim=1)
                    
    return ys[:,1:]
        
def data_gen(filename, V, WINDOW_SIZE, batch, nbatches):

    num_sessions = 0
    inputs = []
    outputs = []
    t1 = torch.from_numpy(np.zeros((batch, WINDOW_SIZE+1),dtype=int))
    t2 = torch.from_numpy(np.zeros((batch, WINDOW_SIZE+1),dtype=int))
    
    with open('Dataset/' + filename, 'r') as f:        
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line[0: WINDOW_SIZE])
            outputs.append(line[WINDOW_SIZE:WINDOW_SIZE*2])

            if len(outputs[-1]) != WINDOW_SIZE: 
                for _ in range(WINDOW_SIZE - len(outputs[-1])): 
                    outputs[-1] = outputs[-1] + (0,)
            if len(inputs[-1]) != WINDOW_SIZE: 
                for _ in range(WINDOW_SIZE - len(inputs[-1])): 
                    inputs[-1] = inputs[-1] + (0,)
                    
        for j in range(nbatches):
            for i in range(batch):           
                x = inputs[i]
                y = outputs[i]
                t1[i][1:] = torch.tensor(x, dtype=torch.float).to(device)
                t2[i][1:] = torch.tensor(y, dtype=torch.float).to(device)

                t1[:,0] = 1
                t2[:,0] = 1
            src = Variable(t1, requires_grad=False)
            tgt = Variable(t2, requires_grad=False)

            yield Batch(src, tgt, 0)
            
def train(N=2, d_model=512, d_ff=2048, h=4, dropout=0.1):
    
    log_keys = "HDFS/hdfs_train"
    devices = [0, 1]
    frac = 1

    print(N, d_model, d_ff, h, dropout)

    # data_train = generate('hdfs_train')
    # BOS_WORD = '<s>'
    # EOS_WORD = '</s>'
    # BLANK_WORD = "<blank>"
    # SRC = data.Field(pad_token=BLANK_WORD)
    # TGT = data.Field(init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)
    # MAX_LEN = 10
    # MIN_FREQ = 2
    # SRC.build_vocab(source, min_freq=MIN_FREQ)
    # TGT.build_vocab(target, min_freq=MIN_FREQ)
    # model = make_model(len(SRC.vocab), len(TGT.vocab), N=N)
    # criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=0, smoothing=0.1)
    # BATCH_SIZE = 12000
    # train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=True)
    # valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=False)

    #Build model
    model = make_model(VOCAB_SIZE, VOCAB_SIZE, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
    criterion = LabelSmoothing(size = VOCAB_SIZE, padding_idx=0, smoothing=0.0)

    # global_model.cuda()
    # criterion.cuda()
    
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    model_par = nn.DataParallel(model, device_ids=devices)
    
    for epoch in range(10):
        model.train()
        run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 30), model, SimpleLossCompute(model.generator, criterion, model_opt))
        # print(run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 30), model_par, MultiGPULossCompute(global_model.generator, criterion, devices=devices, opt=model_opt)))
        model.eval()
        print(run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 5), model, SimpleLossCompute(model.generator, criterion, None)))
        # print(run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 5), model_par, MultiGPULossCompute(global_model.generator, criterion, devices=devices, opt=None)))       
    torch.save(model.state_dict(), "Model/centralized_model.pt")
    torch.save(model, "Model/centralized_models.pt")

    return model

def federated_training(rounds = 1, clients = 0, N=2, d_model=512, d_ff=2048, h=4, dropout=0.1):
    log_keys = "HDFS_1/hdfs_train"
    devices = [0, 1]
    frac = 1

    global_model = make_model(VOCAB_SIZE, VOCAB_SIZE, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
    criterion = LabelSmoothing(size = VOCAB_SIZE, padding_idx=0, smoothing=0.0)

    print(N, d_model, d_ff, h, dropout)

    # Set the model to train and send it to device.
    # global_model.to(device)
    # global_model.train()

    # copy weights
    # global_weights = global_model.state_dict()
    global_weights = []

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0       

    for epoch in tqdm(range(rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        m = max(int(frac * clients), 1)
        idxs_users = np.random.choice(range(clients), m, replace=False)

        for i in idxs_users:
            print("Client:", i)

            log_keys = "hdfs_train" + str(i)
            model = make_model(VOCAB_SIZE, VOCAB_SIZE, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)

            model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
                
            for epoch in range(10):
                model.train()
                run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 30), model, SimpleLossCompute(model.generator, criterion, model_opt))
                # print(run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 30), model_par, MultiGPULossCompute(global_model.generator, criterion, devices=devices, opt=model_opt)))
                model.eval()
                print(run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 5), model, SimpleLossCompute(model.generator, criterion, None)))
                # print(run_epoch(data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 5), model_par, MultiGPULossCompute(global_model.generator, criterion, devices=devices, opt=None)))       

            local_weights.append(copy.deepcopy(model.state_dict()))
            torch.save(model.state_dict(), "Model/model" + str(i) + ".pt")

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        torch.save(global_model, "Model/global_models.pt")
    
    return global_model

def test(model):

    FP = 0
    TP = 0
    TN = 0
    FN = 0

    model.eval()

    num = 5000

    input_n, output_n = generate("HDFS/hdfs_test_normal")
    input_a, output_a = generate("HDFS/hdfs_test_abnormal")

    src_mask = Variable(torch.ones(1, 1, WINDOW_SIZE + 1) )

    t1 = torch.from_numpy(np.zeros((1, WINDOW_SIZE + 1),dtype=int))
    t2 = torch.from_numpy(np.zeros((1, WINDOW_SIZE),dtype=int))
    start_time = time.time()
    
    for i in range(num):

        t1[0][1:WINDOW_SIZE+1] = torch.tensor(input_n[i], dtype=torch.float).to(device)
        t2[0][0:WINDOW_SIZE+1] = torch.tensor(output_n[i], dtype=torch.float).to(device)
        
        t1[0][0]=1

        src = Variable(t1, requires_grad=False)
        tgt = Variable(t2, requires_grad=False)

        pred = greedy_decode(model, src, src_mask, tgt, max_len=WINDOW_SIZE+1, start_symbol=1, pred=False, g=9) 
        
        if -1 in pred: FP = FP + 1
        else: TN = TN + 1

        if (i%1000) == 0: print(i)

    t1 = torch.from_numpy(np.zeros((1, WINDOW_SIZE + 1),dtype=int))
    t2 = torch.from_numpy(np.zeros((1, WINDOW_SIZE + 1),dtype=int))

    for i in range(num):

        t1[0][1:len(input_a[i])+1] = torch.tensor(input_a[i], dtype=torch.float).to(device)
        t2[0][1:WINDOW_SIZE+1] = torch.tensor(output_a[i], dtype=torch.float).to(device)
        
        t1[0][0]=1
        t2[0][0]=1

        src = Variable(t1, requires_grad=False)
        tgt = Variable(t2, requires_grad=False)

        pred = greedy_decode(model, src, src_mask, tgt, max_len=WINDOW_SIZE+1, start_symbol=1, pred=False, g=9) 
        
        if -1 in pred: TP = TP + 1
        else:  FN = FN + 1
        if (i%1000) == 0: print(i)

    A = 100 * (TP + TN)/(TP + TN + FP + FN)
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('True positive (TP): {}, \ntrue negative (TN): {}, \nfalse positive (FP): {}, \nfalse negative (FN): {}, \nAccuracy: {:.3f}%, \nPrecision: {:.3f}%, \nRecall: {:.3f}%, \nF1-measure: {:.3f}%'.format(TP, TN, FP, FN, A, P, R, F1))

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))

    return 

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def generate(name):
    hdfs = set()
    inputs = []
    outputs = []
    num_sessions = 0
    # hdfs = []

    with open('Dataset/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            ##for i in range(len(line) - WINDOW_SIZE):
            inputs.append(line[0: WINDOW_SIZE])
            outputs.append(line[WINDOW_SIZE:WINDOW_SIZE*2])           
            
            if len(outputs[-1]) != WINDOW_SIZE: 
                for _ in range(WINDOW_SIZE - len(outputs[-1])): outputs[-1] = outputs[-1] + (0,)
    
    print(len(inputs))
    print(len(outputs))

    return inputs, outputs