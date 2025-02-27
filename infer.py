import torch
import argparse
import numpy as np
import random

from utils.kotokenize import *
from utils.attention import AttentionLanguageModel
from transformers import BertTokenizerFast


def parse_arguments() :

    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--model_path', type=str, default="models/")
    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--vocab_path', type=str, default="models/")

    parser.add_argument('--character_level_token', type=int, default=0)

    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--head_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=6)

    parser.add_argument('--sample_text', type=str, default=" ")
    parser.add_argument('--auto_gen', type=bool, default=False)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    
    args = parser.parse_args()
    return args

def load_checkpoint(filename, model, optimizer=None):  
    checkpoint = torch.load(filename)  
    model.load_state_dict(checkpoint['model_state_dict'])  
    if optimizer is not None :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']  
    print(f"Checkpoint loaded from {filename}; Last epoch: {epoch}, Last loss: {loss}")  
    return epoch, loss

if __name__ == "__main__" :

    SEED = 486
    random.seed(SEED)
    np.random.seed(SEED)

    args = parse_arguments()

    print("===========================================")
    print("Now Inferring from {} model...".format(args.model_path))
    print("===========================================")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # build tokenizer
    if args.character_level_token :
        tokenizer = KorCharTokenizer(args.data_path)
        VOCAB_SIZE = len(tokenizer)
    else :
        tokenizer = BertTokenizerFast.from_pretrained(args.vocab_path, strip_accents=False, lowercase=False)
        VOCAB_SIZE = tokenizer.vocab_size
    
    # model
    model = AttentionLanguageModel(VOCAB_SIZE, args.embedding_size, args.context_length, args.head_size, args.num_heads, args.num_blocks)
    epoch, loss = load_checkpoint(args.model_path, model)
    print("Model Loaded! Loss {} on Epoch {}".format(loss, epoch))
    model.to(DEVICE)
    model.eval()
    
    # 생성
    sample_text = args.sample_text
    idx = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long).unsqueeze(0).to(DEVICE)
    output = model.generate(idx, max_new_tokens=args.max_new_tokens, auto=args.auto_gen, tokenizer=tokenizer)
    # output = model.generate(idx, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0].tolist()))