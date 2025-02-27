import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

from utils.kotokenize import *

class DecoderBlock(nn.Module) :
    def __init__(self, context_length, embedding_size, head_size, num_heads):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(context_length, embedding_size, head_size, num_heads)
        self.ffn = FeedForward(embedding_size, embedding_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x) :
        
        # (Batch size, context_length, head_size) -> (Batch size, context_length, embedding_size) 
        x = x + self.self_attention_block(self.ln1(x))
        
        # (Batch size, context_length, embedding_size) -> (Batch size, context_length, embedding_size)
        x = x + self.ffn(self.ln2(x)) 
        
        return x

class FeedForward(nn.Module) :
    def __init__(self, in_size, out_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 4*out_size),
            nn.ReLU(),
            nn.Linear(4*out_size, out_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x) :
        return self.net(x) # (Batch size, context_length, embedding_size)

class MultiHeadAttention(nn.Module) :
    def __init__(self, context_length, embedding_size, head_size, num_heads, dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(context_length, embedding_size, head_size//num_heads) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) :

        # (Batch_size, context_length, head_size) -> (Batch_size, context_length, head_size)
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # (Batch_size, context_length, head_size) -> (Batch_size, context_length, embedding_size)
        x = self.dropout(self.proj(x))

        return x 

class Head(nn.Module) :

    def __init__(self, context_length, embedding_size, head_size, dropout=0.1) :
        super().__init__()
        self.context_length = context_length
        self.embedding_size = embedding_size
        self.head_size = head_size

        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x) :
        
        B,T,C = x.shape

        k = self.key(x)     # (Batch_size, context_length, Head_size)
        q = self.query(x)   # (Batch_size, context_length, Head_size)
        v = self.value(x)   # (Batch_size, context_length, Head_size)

        attention_matrix = q @ k.transpose(-2,-1) * (C)**(-1/2) # (Batch_size, context_length, Head_size) @ (Batch_size, Head_size, context_length) -> (Batch_size, context_length, context_length)
        attention_matrix = attention_matrix.masked_fill(self.tril[:T,:T]==0, float('-inf')) # 앞 토큰도 사용하려면 이 줄을 주석처리. 현재의 context_length 사이즈로만 잘라냄
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        attention_matrix = self.dropout(attention_matrix)

        out = attention_matrix @ v # (Batch_size, context_length, context_length) @ (Batch_size, context_length, Head_size) -> # (Batch_size, context_length, head_size)
        return out

class AttentionLanguageModel(nn.Module) :
    def __init__(self, vocab_size, embedding_size, context_length, head_size, num_heads, num_blocks):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length # T
        self.embedding_size = embedding_size # C
        self.head_size = head_size
        self.num_heads = num_heads
        
        self.token_embedding_table = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_size
            )
        
        self.position_embedding_table = nn.Embedding(
            num_embeddings = self.context_length,
            embedding_dim = self.embedding_size,
            )
        
        # (Batch size, context_length, embedding_size) -> (Batch size, context_length, embedding_size)
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(context_length, embedding_size, head_size, num_heads) for _ in range(num_blocks)]
        )

        self.ln_f = nn.LayerNorm(embedding_size)
        # self.self_attention_head = MultiHeadAttention(self.context_length, self.embedding_size, self.head_size, self.num_heads) # (Batch size, context_length, embedding size) -> (Batch size, context_length, head_size)
        # self.ffn = FeedForward(self.head_size) # (Batch size, context_length, head_size) -> (Batch size, context_length, head_size) 
        
        # (Batch size, context_length, embedding_size) -> (Batch size, context_length, vocab_size)
        self.lm_head = nn.Linear(self.embedding_size, self.vocab_size) 


    # 임베딩 통과 -> Cross Entropy 계산
    def forward(self, idx, targets=None) :
        device = idx.device
        
        B,T = idx.shape

        # (Batch size, context_length) -> (Batch size, context_length, embedding_size)
        tok_emb = self.token_embedding_table(idx) 

        # (context_length, embedding_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # 해당 time step의 x의 context 길이에 맞게 pos_emb를 추출

        # (Batch size, context_length, embedding_size), pos_emb에서 broadcasting 발생
        x = tok_emb + pos_emb 

        # (Batch size, context_length, embedding_size) -> (Batch size, context_length, embedding_size)
        x = self.decoder_blocks(x)

        # (Batch size, context_length, embedding_size) -> (Batch size, context_length, vocab_size)
        logits = self.lm_head(x) 

        # target이 있을 경우에만 loss 계산
        if targets is None :
            loss = None
        else :
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)     # vs (Batch size, context_length)

        return logits, loss

    # 생성 함수
    def generate(self, idx, max_new_tokens, auto=False, tokenizer=None, use_argmax=False) :
        
        for _ in range(max_new_tokens) :
            
            # 토큰을 뒤에서부터 context_length만큼 잘라줌 (그 이상을 받으면 positional embedding 불가)
            idx_cond = idx[:, -self.context_length:]

            # 모델에 통과
            logits, loss = self(idx_cond)                                    # (Batch size, context_length, vocab_size)

            # context length의 마지막만 슬라이싱
            logits = logits[:, -1, :]                                   # (Batch size, vocab_size)
            
            # 각각 문장들마다 확률값을 뽑음
            probs = F.softmax(logits, dim=-1)                           # (Batch size, vocab_size)
            
            # 최대값을 하나 뽑음
            if use_argmax :
                idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)          # (Batch size, 1)
            else :
                idx_next = torch.multinomial(probs, num_samples=1)          # (Batch size, 1)

            # input값과 합침
            idx = torch.cat((idx, idx_next), dim=1)                     # (Batch size, context_length + 1)

            if auto :
                os.system("cls")
                print(tokenizer.decode(idx[0].tolist()))

        return idx


def parse_arguments() :

    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--data_dir', type=str, default="./input.txt")
    
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--head_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--eval_step', type=int, default=500)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    
    args = parser.parse_args()

    return args


if __name__ == "__main__" :

    args = parse_arguments()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # build tokenizer
    tokenizer = KorCharTokenizer(args.data_dir)
    
    # model
    VOCAB_SIZE = len(tokenizer)
    model = AttentionLanguageModel(VOCAB_SIZE, args.embedding_size, args.context_length, args.head_size, args.num_heads, args.num_blocks)
    model.to(DEVICE)

    # data prep
    with open(args.data_dir, "r") as f :
        full_text = f.read()
    data = torch.tensor(tokenizer.encode(full_text))
    train_data, val_data = train_val_split(data, 0.9)
    
    # 생성 - 학습 전
    sample_text = " "
    idx = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long).unsqueeze(0).to(DEVICE)
    output = model.generate(idx, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0].tolist()))

    # 학습 세팅
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Logging
    losses = []

    # 학습
    for steps in tqdm(range(args.epoch), desc= "TRAINING STEP") :
        
        xb, yb = get_batch("train", train_data, val_data, args.context_length, args.batch_size)
        
        logits, loss = model(xb.to(DEVICE), yb.to(DEVICE))
        optimizer.zero_grad(set_to_none=True) # zeroing all the gradients from previous step (초기화)
        loss.backward()     # 역전파
        losses.append(loss.item())
        optimizer.step()    # 옵티마이저 다음 스텝으로
        
        if steps % args.eval_step == 0 :
            print("Loss on  {} / {} step : {}".format(steps, args.epoch, loss.item()))

    print("Final Loss : {}".format(loss.item()))

    # 생성 - 학습 후
    print(tokenizer.decode(model.generate(idx, max_new_tokens=args.max_new_tokens)[0].tolist()))

    # Plotting
    plt.plot(losses)
    plt.title("Loss by epoch")
    plt.savefig("./attention_loss.png")