import torch
import argparse
import matplotlib.pyplot as plt

from utils.kotokenize import *
from utils.attention import AttentionLanguageModel
from tokenizers import BertWordPieceTokenizer

def parse_arguments() :

    parser = argparse.ArgumentParser(description='Argparse')

    # Data and Model
    parser.add_argument('--data_path', type=str, default="input.txt")
    parser.add_argument('--out_folder', type=str, default="models")
    
    # Tokenizing
    parser.add_argument('--character_level_token', type=int, default=1)

    # Model Hyperparameters
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--head_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=6)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument('--eval_step', type=int, default=1000)
    parser.add_argument('--save_step', type=int, default=2000)
    
    args = parser.parse_args()
    return args


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth') :
    checkpoint = {  
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'loss': loss,  
    }  
    torch.save(checkpoint, filename)  
    print(f"Checkpoint saved to {filename}")  


if __name__ == "__main__" :

    # SEED = 486
    # random.seed(SEED)
    # np.random.seed(SEED)

    args = parse_arguments()

    print("===========================================")
    print("Now Training model...")
    print("===========================================")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # build tokenizer
    if args.character_level_token :
        tokenizer = KorCharTokenizer(args.data_path)
        VOCAB_SIZE = len(tokenizer)
    else :
        tokenizer = BertWordPieceTokenizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=True, # False로 하면 자소분리됨
            lowercase=True # 대소문자 구분 (한국어는 상관 x)
            )
        tokenizer.train(
            files=args.data_path,
            vocab_size=3000,
            limit_alphabet=500,
            min_frequency=2,
            )
        tokenizer.save_model(("{}/").format(args.out_folder))
        VOCAB_SIZE = tokenizer.get_vocab_size()

    print(tokenizer.get_vocab_size())
    
    # model
    model = AttentionLanguageModel(VOCAB_SIZE, args.embedding_size, args.context_length, args.head_size, args.num_heads, args.num_blocks)
    model.to(DEVICE)
    model.train()

    # data prep
    with open(args.data_path, "r", encoding="utf-8") as f :
        full_text = f.read()
        
    if args.character_level_token :
        data = torch.tensor(tokenizer.encode(full_text))
    else :
        data = torch.tensor(tokenizer.encode(full_text).ids)
    
    train_data, val_data = train_val_split(data, 0.9)
    
    # 생성 - 학습 전
    # sample_text = " "
    # idx = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long).unsqueeze(0).to(DEVICE)
    # output = model.generate(idx, max_new_tokens=args.max_new_tokens)
    # print(tokenizer.decode(output[0].tolist()))

    # 학습 세팅
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Logging
    # losses = []

    # 학습
    for steps in tqdm(range(1,args.epoch+1), desc= "TRAINING STEP") :
        
        xb, yb = get_batch("train", train_data, val_data, args.context_length, args.batch_size)
        
        logits, loss = model(xb.to(DEVICE), yb.to(DEVICE))
        optimizer.zero_grad(set_to_none=True) # zeroing all the gradients from previous step (초기화)
        loss.backward()     # 역전파
        # losses.append(loss.item())
        optimizer.step()    # 옵티마이저 다음 스텝으로
        
        # Logging % Save
        if steps % args.eval_step == 0 :
            
            model.eval()
            xb, yb = get_batch("eval", train_data, val_data, args.context_length, args.batch_size)
            logits_eval, loss_eval = model(xb.to(DEVICE), yb.to(DEVICE))
            model.train()
            print("Loss on  {} / {} step ----- \tTraining : {} \tEval :{}".format(steps, args.epoch, loss.item(), loss_eval.item()))

        if steps % args.save_step == 0 :
            save_checkpoint(model, optimizer, steps, loss.item(), "{}/checkpoint_{}.pth".format(args.out_folder, steps))
            print("Model Saved!")

    print("Final Loss : {}".format(loss.item()))

    # 생성 - 학습 후
    # print(tokenizer.decode(model.generate(idx, max_new_tokens=args.max_new_tokens)[0].tolist()))

    # Plotting
    # plt.plot(losses)
    # plt.title("Loss by epoch")
    # plt.savefig("./attention_loss.png")


### Done ====================================
# TODO : Model Save % Load
# TODO : eval loss 계산해 출력
# TODO : Decoder Block size 변경하기
# TODO : 이 상황에서는 argmax를 해도 되지 않나? -> 생성 전략의 차이. https://www.packtpub.com/en-us/learning/how-to-tutorials/exploring-token-generation-strategies?srsltid=AfmBOopu9q3ZAiCC7hCPNsscB_Zk-Gkj3_hKdiQFkwaEVbdSl0CWQxHF
### =========================================

# TODO : Tokenizer 변경하기
# TODO : wandb 연결하기
# TODO : Trainer에 옮겨야지