import torch
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

from utils.kotokenize import *
from utils.attention import AttentionLanguageModel

from transformers import Trainer, TrainingArguments
from datasets import Dataset


class KorCharTokenizer(Dataset) :
    def __init__(self, path=""):
        super().__init__()
        self.path = path
        self.stoi = {}
        self.itos = {}
        self.all_kor_chars = "./all_kor_chars.txt"
        self.special_chars = " 0123456789~!@#$%^&*()[]{}:;,./<>?/*-+=_`"
        self._build_vocab()
    
    def _build_vocab(self) :
        
        # 텍스트 추출 (경로 주어지지 않은 경우 모든 한국어 음소)
        if self.path == "" :
            self.path = self.all_kor_chars

        with open(self.path, "r", encoding="cp949") as f :
            text = f.read()

        # 특수문자 추가
        text += self.special_chars

        # 중복 제거, 정렬
        chars = sorted(list(set(text)))

        # character 단위 양방향 변환 딕셔너리
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

    def __len__(self) :
        return len(self.stoi)

    def get_vocab_size(self) -> int :
        return len(self.stoi)

    # encode 함수
    # TODO : encode 함수에 매칭되는 게 없을 경우 unk 토큰 처리
    def encode(self, s : str) -> list :
        return [self.stoi[c] for c in s]

    # decode 함수
    def decode(self, l : list) -> str :
        return ''.join([self.itos[i] for i in l])


    # 이거랑 잘 merge. =============================================================
    # =============================================================================
    # =============================================================================
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
        self.addi_feat_df = self.dataset.drop(columns=["id","age","gender","stt","target","file", "question"])
        self.inputs = []
        self.labels = []

        for idx in tqdm(range(len(self.dataset))):

            try :
                tokenized = preprocessing(self.dataset["stt"][idx], self.dataset["age"][idx], self.dataset["gender"][idx], 300)
                self.inputs.append(tokenized)
                self.labels.append(self.dataset['target'][idx])
            except :
                pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'addi_feat' : self.addi_feat_df.loc[idx].values,
            'labels': self.labels[idx],
        }
    
    # input_ids 모델에 dict형태로 줘야됨. 이런 형태의 데이터셋으로 만들어줘야 하고,
    # 모델 안에서도 dict를 call하는 key가 맞아야 하는지? 순서만 맞으면 되는지???
    def forward(self, input_ids, attention_mask, addi_feat, labels=None) :



def parse_arguments() :

    parser = argparse.ArgumentParser(description='Argparse')

    # parser.add_argument('--model_path', type=str, default="./models/checkpoint_")
    parser.add_argument('--data_path', type=str, default="./input.txt")
    parser.add_argument('--output_dir', type=str, default="./output")
    
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--head_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--save_step', type=int, default=2000)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    
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
    # print("Now Training {} model...".format(args.model_path))
    print("===========================================")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # build tokenizer
    tokenizer = KorCharTokenizer(args.data_path)
    
    # model
    VOCAB_SIZE = len(tokenizer)
    model = AttentionLanguageModel(VOCAB_SIZE, args.embedding_size, args.context_length, args.head_size, args.num_heads, args.num_blocks)
    model.to(DEVICE)
    model.train()

    # data prep
    with open(args.data_path, "r") as f :
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

    # 학습
    for steps in tqdm(range(args.epoch), desc= "TRAINING STEP") :
        
        xb, yb = get_batch("train", train_data, val_data, args.context_length, args.batch_size)
        
        logits, loss = model(xb.to(DEVICE), yb.to(DEVICE))
        optimizer.zero_grad(set_to_none=True) # zeroing all the gradients from previous step (초기화)
        loss.backward()     # 역전파
        optimizer.step()    # 옵티마이저 다음 스텝으로
        
        # Logging % Save
        if steps % args.eval_step == 0 :
            
            model.eval()
            xb, yb = get_batch("eval", train_data, val_data, args.context_length, args.batch_size)
            logits_eval, loss_eval = model(xb.to(DEVICE), yb.to(DEVICE))
            model.train()
            print("Loss on  {} / {} step ----- \tTraining : {} \tEval :{}".format(steps, args.epoch, loss.item(), loss_eval.item()))

        if steps % args.save_step == 0 :
            save_checkpoint(model, optimizer, steps, loss.item(), "./models/checkpoint_{}.pth".format(steps))
            print("Model Saved!")

    print("Final Loss : {}".format(loss.item()))

    # 생성 - 학습 후
    print(tokenizer.decode(model.generate(idx, max_new_tokens=args.max_new_tokens)[0].tolist()))




    training_args = TrainingArguments(

        output_dir=args.output_dir,
        
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',

        logging_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,

        save_total_limit = 20,

        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        
        # warmup_ratio= 0.1,
        # adam_beta1 = 0.9,
        # adam_beta2 = 0.999,
        # adam_epsilon=1e-08,
        weight_decay=0.01,
        # lr_scheduler_type='linear',

        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()


### Done ====================================
# TODO : Model Save % Load
# TODO : eval loss 계산해 출력
### =========================================

# TODO : Tokenizer 변경하기 -> huggingface wordpiece tokenizer 

# TODO : 이 상황에서는 argmax를 해도 되지 않나? -> 

# TODO : 1) 모델을 들어올려서 2) Trainer에 옮겨야지
    # TODO : 데이터셋 feeding하는 주는 클래스를 변경. 이건 Tokenizer 작업과 같이 해야 할 수도 있겠음
    # TODO : 모델 안의 forward() 파라미터 전부 `input_ids`랑 `label` 이런 형태의 dict로 줄 수 있게 바꾸기
# TODO : wandb 연결하기
